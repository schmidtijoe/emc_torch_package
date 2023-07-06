""" want to use torch gradient decent to optimize pulse shape"""
import numpy as np
import plotly.offline
import torch
from emc_sim import functions, options, prep
import logging
import plotly.graph_objects as go
import plotly.subplots as psub
import tqdm.auto
from collections import OrderedDict

logModule = logging.getLogger(__name__)


# loss function
def loss_pulse_optim(tmp_magnetization: torch.tensor, target: torch.tensor,
                     p_x: torch.tensor, p_y: torch.tensor, g: torch.tensor,
                     lam_power: float = 1e-2, lam_grad: float = 1.0, lam_smooth: float = 1e-2):
    # magnetization is tensor [n_t2, n_b1, n_samples, 4]
    # target is tensor [n_samples, 4]
    # want individual b1 profiles to match target as closely as possible
    # average over t2
    if tmp_magnetization.shape.__len__() > 3:
        tmp_magnetization = torch.mean(tmp_magnetization, dim=0)
    # calculate error
    shape_err = torch.zeros(1, device=target.device)
    for b1_idx in range(tmp_magnetization.shape[0]):
        shape_err += torch.nn.MSELoss()(torch.norm(tmp_magnetization[b1_idx, :, :2], dim=-1), target[:, 0])
        shape_err += torch.nn.MSELoss()(tmp_magnetization[b1_idx, :, 2], target[:, 1])
    # want to enforce boundaries
    # make pulse abs power low
    power_err = lam_power * torch.exp(torch.mean(p_x**2 + p_y**2))

    # make gradient amplitude low
    grad_amp_err = lam_grad * torch.exp(torch.max(g**2) * 1e-3 + 1e-15)
    # enforce smootheness
    p_g = torch.sum(torch.abs(torch.gradient(p_x)[0])) + torch.sum(torch.abs(torch.gradient(p_y)[0]))
    g_g = torch.sum(torch.abs(torch.gradient(g)[0]))
    smootheness_err = lam_smooth * (p_g + g_g)

    loss = shape_err + power_err + grad_amp_err + smootheness_err
    return loss, shape_err, power_err, grad_amp_err, smootheness_err


def func_to_calculate(pulse_x: torch.tensor, pulse_y: torch.tensor, grad: torch.tensor,
                      b1_vals: torch.tensor, t2_vals: torch.tensor,
                      dt_us: float, sampling_axis: torch.tensor,
                      init_mag: torch.tensor, target: torch.tensor):
    matrix_propagation = functions.matrix_effect_grad_pulse_multi_dim(
        t2_tensor=t2_vals, b1_tensor=b1_vals, pulse_x=pulse_x, pulse_y=pulse_y, grad=grad,
        dt_us=dt_us, sample_axis=sampling_axis,
        t1_s=torch.tensor(1.5).to(t2_vals.device)
    )

    mag_prop = torch.einsum('ijklm, kl -> ijkm', matrix_propagation, init_mag)

    return loss_pulse_optim(mag_prop, target, p_x=pulse_x, p_y=pulse_y, g=grad), mag_prop


def setup_and_run(sim_params: options.SimulationParameters, grad_pulse: prep.GradPulse):
    # gpu device
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    logModule.info(f"torch device: {device}")
    # set b1_value range
    n_b1s = 6
    b1_values = torch.linspace(0.5, 1.5, n_b1s).to(device)
    # set standard t2
    t2_values = torch.tensor([0.035, 0.05]).to(device)
    # get initial magnetization
    tmp_data = options.SimTempData(sim_params=sim_params)
    initial_magnetization = tmp_data.magnetization_propagation.to(device)
    sample_axis = tmp_data.sample_axis.to(device)
    # get sampling time -> divide puls sampling steps into 1/4h of the original
    dt_us = torch.tensor(grad_pulse.dt_sampling_steps).to(device)

    # want to optimize for a pulse shape that is agnostic to b1 changes
    slice_thickness = torch.tensor(0.7).to(device)  # [mm]
    target_shape = torch.zeros((sample_axis.shape[0], 2)).to(device)  # store absolute value of
    # transverse magnetization in first, z mag in second
    target_shape[torch.abs(sample_axis) < 1e-3 * slice_thickness / 2, 0] = 1.0
    target_shape[:, 1] = initial_magnetization[:, -1]
    target_shape[torch.abs(sample_axis) < 1e-3 * slice_thickness / 2, 1] = 0.0

    # initial guess
    # take maximum amplitude from 90 degree pulse as 1
    p_max = torch.max(torch.abs(grad_pulse.data_pulse))
    g_max = 35.0        # mT/m  take this as maximum gradient

    # want to map the tensors between 0 and 1
    px = torch.rand(size=(grad_pulse.data_pulse.shape[0],), requires_grad=True, device=device)
    py = torch.rand(size=(grad_pulse.data_pulse.shape[0],), requires_grad=True, device=device)
    # for gradients 1 is max grad = 40 mT/m
    g = torch.rand(size=(int(grad_pulse.data_grad.shape[0] / 10),), requires_grad=True, device=device)

    # set optimizer
    optimizer = torch.optim.SGD([px, py, g], lr=0.5, momentum=0.5)
    steps = 200
    loss_tracker = [[], [], [], [], []]

    torch.autograd.set_detect_anomaly(True)

    with tqdm.auto.trange(steps) as t:
        t.set_description(f"progress")
        t.set_postfix(ordered_dict=OrderedDict({"loss": -1, "power": torch.sum(torch.abs(px + 1j * py)).item(),
                                                "g max": torch.max(torch.abs(g)).item()}))
        for i in t:
            optimizer.zero_grad()
            # set input
            g_input = -g_max * torch.nn.Sigmoid()(g.repeat_interleave(10))
            px_input = p_max * torch.nn.Sigmoid()(px)
            py_input = p_max * torch.nn.Sigmoid()(py)
            loss_tuple, mag_prop = func_to_calculate(
                pulse_x=px_input, pulse_y=py_input, grad=g_input, b1_vals=b1_values, t2_vals=t2_values, dt_us=dt_us,
                sampling_axis=sample_axis, init_mag=initial_magnetization, target=target_shape
            )
            for l_idx in range(loss_tuple.__len__()):
                loss_tracker[l_idx].append(loss_tuple[l_idx])

            loss = loss_tuple[0]
            loss.backward()
            optimizer.step()
            t.set_postfix(ordered_dict=OrderedDict({"loss": loss.item(),
                                                    "power": torch.sum(torch.sqrt(px**2 + py**2)).item(),
                                                    "g max": torch.max(torch.abs(g)).item()}))
            if i % 25 == 1:
                plot_run(i, px_input, py_input, g_input)
                plot_mag_prop(mag_prop, sample_axis=sample_axis, target_shape=target_shape, run=i)

    fig = go.Figure()
    names = ["loss", "shape", "power", "grad", "smooth"]
    for loss_list_idx in range(loss_tracker.__len__()):
        l = torch.tensor(loss_tracker[loss_list_idx]).detach().cpu()
        fig.add_trace(
            go.Scatter(x=torch.arange(loss_tracker[loss_list_idx].__len__()),
                       y=l, name=names[loss_list_idx])
        )

    plotly.offline.plot(fig, filename=f"optim/loss.html")

    return px, py, g


def plot_mag_prop(mag_prop_in: torch.tensor, sample_axis: torch.tensor, target_shape: torch.tensor, run: int):
    x_ax = sample_axis.cpu()
    target = target_shape.clone().detach().cpu()
    mag_prop = mag_prop_in.clone().detach().cpu()
    mag = mag_prop[0, :, :, 0] + 1j * mag_prop[0, :, :, 1]
    fig = psub.make_subplots(rows=mag_prop_in.shape[1], cols=1)
    for k in range(mag_prop_in.shape[1]):
        fig.add_trace(
            go.Scatter(x=x_ax, y=torch.abs(mag[k]), name=f"b1 val {k+1} - mag"),
            row=k+1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_ax, y=torch.angle(mag[k]) / torch.pi, name=f"b1 val {k+1} -phase"),
            row=k+1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_ax, y=mag_prop[0, k, :, 2], name=f"b1 val {k+1} - z - mag"),
            row=k+1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_ax, y=target[:, 0], fill="tozeroy", name="target - mag"),
            row=k+1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_ax, y=target[:, 1], name="target - z mag"),
            row=k+1, col=1
        )
    plotly.offline.plot(fig, filename=f"optim/optim_magnetization_profile_run_{run}.html")


def optimize(sim_params: options.SimulationParameters, grad_pulse: prep.GradPulse):
    optim_pulse_x, optim_pulse_y, optim_grad = setup_and_run(sim_params=sim_params, grad_pulse=grad_pulse)


def plot_run(run: int, px: torch.tensor, py: torch.tensor, g: torch.tensor):
    g_plot = g.clone().detach().cpu().repeat_interleave(10)
    px_plot = px.clone().detach().cpu()
    py_plot = py.clone().detach().cpu()
    x_axis = torch.arange(py_plot.shape[0])
    fig = psub.make_subplots(2, 1)
    fig.add_trace(
        go.Scatter(x=x_axis, y=g_plot, name="g"),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_axis, y=px_plot, name="px"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_axis, y=py_plot, name="py"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_axis, y=torch.sqrt(px_plot ** 2 + py_plot ** 2), fill="tozeroy", name="p abs"),
        row=1, col=1
    )
    plotly.offline.plot(fig, filename=f"optim/optimization_grad_pulse_run_{run}.html")
