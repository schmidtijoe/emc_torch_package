""" want to use torch gradient decent to optimize pulse shape"""
import torch
from emc_sim import functions, options, prep
from pulse_optim import plotting
import logging
import tqdm.auto
from collections import OrderedDict

logModule = logging.getLogger(__name__)


# loss function
def loss_pulse_optim(tmp_magnetization: torch.tensor,
                     target_mag: torch.tensor, target_phase: torch.tensor, target_z: torch.tensor,
                     p_x: torch.tensor, p_y: torch.tensor, g: torch.tensor, g_re:torch.tensor,
                     lam_power: float = 1, lam_grad: float = 1, lam_smooth: float = 1, lam_ramps: float = 1):
    # magnetization is tensor [n_t2, n_b1, n_samples, 4]
    # target is tensor [n_samples, 4]
    # want individual b1 profiles to match target as closely as possible
    # average over t2
    while tmp_magnetization.shape.__len__() > 3:
        tmp_magnetization = torch.mean(tmp_magnetization, dim=0)
    # calculate error - want to get MSE loss across all slice profiles - magnitude, phase and z (b1 in first dim)
    # and then sum
    shape_err = torch.sum(torch.nn.MSELoss()(torch.norm(tmp_magnetization[:, :, :2], dim=-1), target_mag)) + \
                torch.sum(torch.nn.MSELoss()(tmp_magnetization[:, :, 2], target_z)) + \
                torch.sum(torch.nn.MSELoss()(torch.angle(tmp_magnetization[:, :, 0] + 1j * tmp_magnetization[:, :, 1]),
                                             target_phase))
    # try to make all O(0) order then using the lambdas is much more straight forward
    # want to enforce boundaries
    # make pulse abs power low
    power_err = torch.sum(p_x ** 2 + p_y ** 2) / p_x.shape[0] * 1e7

    # make gradient amplitude low
    grad_amp_err = (torch.sum(g ** 2) + torch.sum(g_re**2)) * 1e-7
    # enforce smootheness - operate along the pulse dimension
    p_g = torch.sum(torch.abs(torch.gradient(p_x, dim=-1)[0])) + torch.sum(torch.abs(torch.gradient(p_y, dim=-1)[0]))
    g_g = torch.sum(torch.abs(torch.gradient(g)[0])) + torch.sum(torch.abs(torch.gradient(g_re)[0]))
    smootheness_err = p_g * 10 + g_g * 1e-3

    # enforce easy ramps
    ramps_p = 1e9 * (torch.sum(p_x[:, 0]**2 + p_x[:, -1]**2 + p_y[:, 0]**2 + p_y[:, -1]**2))
    ramps_g = 1e-2 * (g[0] + g[-1] + g_re[0] + g_re[-1])**2
    ramp_err = ramps_p + ramps_g

    loss = shape_err + lam_power * power_err + lam_grad * grad_amp_err + lam_smooth * smootheness_err + lam_ramps * ramp_err
    return loss, shape_err, power_err, grad_amp_err, smootheness_err, ramp_err


def func_to_calculate(pulse_x: torch.tensor, pulse_y: torch.tensor,
                      grad: torch.tensor, grad_rephase: torch.tensor,
                      sim_data: options.SimulationData, dt_s: float,
                      target_mag: torch.tensor, target_phase: torch.tensor, target_z: torch.tensor):
    # pulse gradient propagation
    matrix_propagation_pulse = functions.propagate_gradient_pulse_relax(
        sim_data=sim_data, pulse_x=pulse_x, pulse_y=pulse_y, grad=grad,
        dt_s=dt_s)
    # rephasing propagation
    matrix_propagation_rephase = functions.propagate_gradient_pulse_relax(
        pulse_x=torch.zeros((pulse_x.shape[0], grad_rephase.shape[0])).to(sim_data.device),
        pulse_y=torch.zeros((pulse_x.shape[0], grad_rephase.shape[0])).to(sim_data.device), grad=grad_rephase,
        dt_s=dt_s*10, sim_data=sim_data
    )
    init_mag = sim_data.magnetization_propagation
    # propagator dim [t1: i, t2: j, b1: k, samples: l, 4: m, 4: n], mag_vector init dim [samples: l, 4: n]
    mag_prop_pulse = torch.einsum('ijklmn, ln -> ijklm', matrix_propagation_pulse, init_mag)
    # after first propagation mag vector has same dim except last [t1: i, t2: j, b1: k, samples: l, 4: n] ijklm
    mag_prop_rephase = torch.einsum('ijklmn, ijkln -> ijklm', matrix_propagation_rephase, mag_prop_pulse)
    return loss_pulse_optim(tmp_magnetization=mag_prop_rephase,
                            target_mag=target_mag, target_phase=target_phase, target_z=target_z,
                            p_x=pulse_x, p_y=pulse_y, g=grad, g_re=grad_rephase), mag_prop_rephase


def setup_and_run(sim_params: options.SimulationParameters):
    name = "run4"
    # gpu device
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    logModule.info(f"torch device: {device}")
    # set rng
    torch.manual_seed(0)
    # set b1_value range
    n_b1s = 6
    sim_params.settings.b1_list = torch.linspace(0.5, 1.5, n_b1s).tolist()
    # set standard t2
    sim_params.settings.t2_list = [35, 50]  # in ms

    # set pulse original
    grad_pulse = prep.GradPulse.prep_single_grad_pulse(
        params=sim_params, excitation_flag=True, grad_rephase_factor=0.0)
    # get initial magnetization
    sim_data = options.SimulationData.from_sim_parameters(sim_params=sim_params, device=device)
    initial_magnetization = sim_data.magnetization_propagation
    sample_axis = sim_data.sample_axis

    # get sampling time -> divide puls sampling steps into 1/4h of the original
    dt_s = torch.tensor(grad_pulse.dt_sampling_steps * 1e-6).to(device)

    # want to optimize for a pulse shape that is agnostic to b1 changes
    slice_thickness = torch.tensor(0.7).to(device)  # [mm]
    # define target shapes for magnetization values (magnitude phase and z) , after including rephasing
    # want step function in magnitude x direction
    target_shape_mag_abs = torch.zeros((n_b1s, sample_axis.shape[0])).to(device)
    target_shape_mag_abs[:, torch.abs(sample_axis) < 1e-3 * slice_thickness / 2] = 1.0
    # want flat phase
    target_shape_mag_phase = torch.zeros((n_b1s, sample_axis.shape[0])).to(device)
    # want "anti" step functions in z direction
    target_shape_mag_z = torch.zeros((n_b1s, sample_axis.shape[0])).to(device)
    target_shape_mag_z[:] = initial_magnetization[None, :, -1]
    target_shape_mag_z[:, torch.abs(sample_axis) < 1e-3 * slice_thickness / 2] = torch.zeros(1, device=device)

    # initial guess
    # take maximum amplitude from 90 degree pulse as 1
    p_max = torch.max(torch.abs(torch.concatenate((grad_pulse.data_pulse_x, grad_pulse.data_pulse_y))))
    g_max = 35.0  # mT/m  take this as maximum gradient

    # want to map the tensors between 0 and 1
    px = torch.rand(size=grad_pulse.data_pulse_x.shape, requires_grad=True, device=device)
    py = torch.rand(size=grad_pulse.data_pulse_y.shape, requires_grad=True, device=device)
    # for gradients 1 is max grad = 40 mT/m
    g = torch.rand(size=(int(grad_pulse.data_grad.shape[0] / 10),), requires_grad=True, device=device)
    g_rephase = torch.rand(size=(50,), requires_grad=True, device=device)

    # set optimizer
    optimizer = torch.optim.SGD([px, py, g], lr=0.2, momentum=0.5)
    steps = 50
    loss_tracker = [[], [], [], [], [], []]

    # torch.autograd.set_detect_anomaly(True)

    with tqdm.auto.trange(steps) as t:
        t.set_description(f"progress")
        t.set_postfix(ordered_dict=OrderedDict({"loss": -1, "power": torch.sum(torch.sqrt(px**2 + py**2)).item(),
                                                "g max": torch.max(torch.abs(g)).item()}))
        for i in t:
            optimizer.zero_grad()
            # set input
            g_input = -g_max * torch.nn.Sigmoid()(g.repeat_interleave(10))
            gr_input = g_max * torch.nn.Sigmoid()(g_rephase)  # sampled on different raster!! (dt * 10),
            # solved in func_to_calculate
            px_input = p_max * torch.nn.Sigmoid()(px)
            py_input = p_max * torch.nn.Sigmoid()(py)
            loss_tuple, mag_prop = func_to_calculate(
                pulse_x=px_input, pulse_y=py_input, grad=g_input, grad_rephase=gr_input,
                sim_data=sim_data, dt_s=dt_s,
                target_mag=target_shape_mag_abs, target_phase=target_shape_mag_phase, target_z=target_shape_mag_z
            )
            for l_idx in range(loss_tuple.__len__()):
                loss_tracker[l_idx].append(loss_tuple[l_idx])

            loss = loss_tuple[0]
            loss.backward()
            optimizer.step()
            t.set_postfix(ordered_dict=OrderedDict({"loss": loss.item(),
                                                    "power": torch.sum(torch.sqrt(px ** 2 + py ** 2)).item(),
                                                    "g max": torch.max(torch.abs(g)).item()}))

    plotting.plot_run(i, px_input, py_input, g_input, gr_input, name=name)
    plotting.plot_mag_prop(mag_prop, sample_axis=sample_axis,
                  target_mag=target_shape_mag_abs, target_phase=target_shape_mag_phase, target_z=target_shape_mag_z,
                  run=i, b1_vals=sim_data.b1_vals, name=name)

    plotting.plot_losses(loss_tracker=loss_tracker, name=name)

    return px, py, g


def main():
    parser, prog_args = options.createCommandlineParser()

    sim_params = options.SimulationParameters.from_cmd_args(prog_args)
    # set logging level after possible config file read
    if sim_params.config.debug_flag:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=level)

    try:
        setup_and_run(sim_params=sim_params)
    except Exception as e:
        print(e)
        parser.print_usage()


if __name__ == '__main__':
    main()
