import plotly.offline
import torch
import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.express.colors as pxc


def plot_losses(loss_tracker: list, name: str = ""):
    fig = go.Figure()
    names = ["loss", "shape", "power", "grad", "smooth", "ramps"]
    for loss_list_idx in range(loss_tracker.__len__()):
        l = torch.tensor(loss_tracker[loss_list_idx]).detach().cpu()
        fig.add_trace(
            go.Scatter(x=torch.arange(loss_tracker[loss_list_idx].__len__()),
                       y=l, name=names[loss_list_idx])
        )
    plotly.offline.plot(fig, filename=f"optim/{name}_loss.html")


def plot_mag_prop(mag_prop_in: torch.tensor, sample_axis: torch.tensor,
                  target_mag: torch.tensor, target_phase: torch.tensor, target_z: torch.tensor, run: int,
                  b1_vals: torch.tensor, name: str = ""):
    x_ax = sample_axis.cpu()
    target_mag_plot = target_mag.clone().detach().cpu()
    target_phase_plot = target_phase.clone().detach().cpu()
    target_z_plot = target_z.clone().detach().cpu()
    mag_prop = mag_prop_in.clone().detach().cpu()
    b1s = b1_vals.clone().detach().cpu()
    mag = mag_prop[0, 0, :, :, 0] + 1j * mag_prop[0, 0, :, :, 1]
    colors = pxc.sample_colorscale("viridis", torch.linspace(0.1, 0.9, 6))
    fig = psub.make_subplots(rows=b1s.shape[0], cols=1, subplot_titles=[f"{b1s[i]:.2f}" for i in range(b1s.shape[0])])
    for k in range(b1s.shape[0]):
        fig.add_trace(
            go.Scatter(x=x_ax, y=torch.abs(mag[k]), name=f"b1 val {b1s[k]:.2f} - mag", color=colors[0]),
            row=k + 1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_ax, y=torch.angle(mag[k]) / torch.pi, name=f"b1 val {b1s[k]:.2f} -phase", color=colors[1]),
            row=k + 1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_ax, y=mag_prop[0, k, :, 2], name=f"b1 val {b1s[k]:.2f} - z - mag", color=colors[2]),
            row=k + 1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_ax, y=target_mag_plot[k, :], fill="tozeroy", name="target - mag", color=colors[3]),
            row=k + 1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_ax, y=target_z_plot[k, :], name="target - z mag", color=colors[4]),
            row=k + 1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_ax, y=target_phase_plot[k, :], name="target - phase", color=colors[5]),
            row=k + 1, col=1
        )
    plotly.offline.plot(fig, filename=f"optim/optim_magnetization_profile_{name}_{run}.html")


def plot_run(run: int, px: torch.tensor, py: torch.tensor, g: torch.tensor, gr: torch.tensor, name: str = ""):
    g_plot = torch.zeros(px.shape[0] + 10 * gr.shape[0])
    g_plot[:px.shape[0]] = g.clone().detach().cpu().repeat_interleave(10)
    g_plot[px.shape[0]:] = gr.clone().detach().cpu().repeat_interleave(10)

    px_plot = torch.zeros(px.shape[0] + gr.shape[0])
    px_plot[:px.shape[0]] = px.clone().detach().cpu()
    py_plot = torch.zeros(px.shape[0] + gr.shape[0])
    py_plot[:px.shape[0]] = py.clone().detach().cpu()

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
    plotly.offline.plot(fig, filename=f"optim/optimization_grad_pulse_run_{name}_{run}.html")

