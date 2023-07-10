import plotly.offline
import torch
import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.express.colors as pxc
from emc_sim import options as eso
from pulse_optim import options


def plot_losses(loss_tracker: list, config: options.ConfigOptimization):
    fig = go.Figure()
    names = ["loss", "shape", "power", "grad", "smooth", "ramps"]
    for loss_list_idx in range(loss_tracker.__len__()):
        l = torch.tensor(loss_tracker[loss_list_idx]).detach().cpu()
        fig.add_trace(
            go.Scatter(x=torch.arange(loss_tracker[loss_list_idx].__len__()),
                       y=l, name=names[loss_list_idx])
        )
    # append to filename
    stem = config.optim_save_path.stem
    filename = config.optim_save_path.with_name(f"{stem}_loss")
    plotly.offline.plot(fig, filename=filename.with_suffix(".html").as_posix())


def plot_mag_prop(sim_data: eso.SimulationData,
                  target_mag: torch.tensor, target_phase: torch.tensor, target_z: torch.tensor, run: int,
                  config: options.ConfigOptimization):
    x_ax = sim_data.sample_axis.clone().detach().cpu()
    target_mag_plot = target_mag.clone().detach().cpu()
    target_phase_plot = target_phase.clone().detach().cpu()
    target_z_plot = target_z.clone().detach().cpu()
    mag_prop = sim_data.magnetization_propagation[0, 0].clone().detach().cpu()
    b1s = sim_data.b1_vals.clone().detach().cpu()
    mag = mag_prop[:, :, 0] + 1j * mag_prop[:, :, 1]
    colors = pxc.sample_colorscale("viridis", torch.linspace(0.1, 0.9, 6).tolist())
    fig = psub.make_subplots(rows=b1s.shape[0], cols=1, subplot_titles=[f"{b1s[i]:.2f}" for i in range(b1s.shape[0])])
    for k in range(b1s.shape[0]):
        fig.add_trace(
            go.Scatter(x=x_ax, y=torch.abs(mag[k]), name=f"b1 val {b1s[k]:.2f} - mag",
                       marker={"color": colors[0]}),
            row=k + 1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_ax, y=torch.angle(mag[k]) / torch.pi, name=f"b1 val {b1s[k]:.2f} -phase",
                       marker={"color": colors[1]}),
            row=k + 1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_ax, y=mag_prop[k, :, 2], name=f"b1 val {b1s[k]:.2f} - z - mag",
                       marker={"color": colors[2]}),
            row=k + 1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_ax, y=target_mag_plot[k, :], fill="tozeroy", name="target - mag",
                       marker={"color": colors[3]}),
            row=k + 1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_ax, y=target_z_plot[k, :], name="target - z mag",
                       marker={"color": colors[4]}),
            row=k + 1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_ax, y=target_phase_plot[k, :], name="target - phase", marker={"color": colors[5]}),
            row=k + 1, col=1
        )
    # append to filename
    stem = config.optim_save_path.stem
    filename = config.optim_save_path.with_name(f"{stem}_mag_profile_{run}")
    plotly.offline.plot(fig, filename=filename.with_suffix(".html").as_posix())


def plot_grad_pulse_optim_run(run: int, px: torch.tensor, py: torch.tensor, g: torch.tensor, gr: torch.tensor,
                              config: options.ConfigOptimization):
    pick_b1 = int(px.shape[0] / 2)
    shape_plot = px.shape[1] + 10 * gr.shape[0]
    g_plot = torch.zeros(shape_plot)
    g_plot[:px.shape[1]] = g.clone().detach().cpu()
    g_plot[px.shape[1]:] = gr.clone().detach().cpu().repeat_interleave(10)

    px_plot = torch.zeros(shape_plot)
    px_plot[:px.shape[1]] = px[pick_b1].clone().detach().cpu()
    py_plot = torch.zeros(shape_plot)
    py_plot[:px.shape[1]] = py[pick_b1].clone().detach().cpu()

    x_axis = torch.arange(shape_plot)
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
    # append to filename
    stem = config.optim_save_path.stem
    filename = config.optim_save_path.with_name(f"{stem}_optim_grad_pulse_step-{run}")
    plotly.offline.plot(fig, filename=filename.with_suffix(".html").as_posix())
