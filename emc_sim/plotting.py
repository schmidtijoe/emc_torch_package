import plotly.graph_objects as go
import plotly.subplots as psub
from plotly.express.colors import sample_colorscale
import plotly
from emc_sim import options
import torch
import numpy as np
import logging

logModule = logging.getLogger(__name__)


def plot_emc_sim_data(sim_data: options.SimulationData):
    emc_mag = sim_data.emc_signal_mag.clone().detach().cpu()
    emc_phase = sim_data.emc_signal_phase.clone().detach().cpu()
    fig = psub.make_subplots(rows=2, cols=1)
    x_ax = torch.arange(1, 1+sim_data.emc_signal_mag.shape[0])
    fig.add_trace(
        go.Scatter(x=x_ax, y=emc_mag,
                   name=f"mag_emc"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_ax, y=emc_phase,
                   name=f"phase_emc"),
        row=2, col=1
    )
    fig.update_layout(legend_title_text="EMC simulated curves")
    fig.update_xaxes(title_text="# Echo")
    fig.update_yaxes(title_text="Signal")

    plotly.offline.plot(fig, filename=f'./tests/emc_signal.html')


def plot_magnetization(sim_data: options.SimulationData, id: int):
    plot_mag = sim_data.magnetization_propagation[0, 0, 0].clone().detach().cpu()
    axis = sim_data.sample_axis.clone().detach().cpu()
    fig = psub.make_subplots(rows=2, cols=1)
    labels = ["x", "y", "z", "e", "abs xy"]
    colors = sample_colorscale('viridis', np.linspace(0.9, 0.1, 5))
    fig.add_trace(
        go.Scatter(x=axis, y=torch.norm(plot_mag[:, :2], dim=1),
                   name=f"mag_{labels[-1]} init",
                   line=dict(color=colors[-1]), fill='tozeroy'),
        row=1, col=1
    )
    for k in range(4):
        fig.add_trace(
            go.Scatter(x=axis, y=plot_mag[:, k],
                       name=f"mag_{labels[k]} init",
                       line=dict(color=colors[k])),
            row=1, col=1
        )
    plotly.offline.plot(fig, filename=f'./tests/grad_pulse_propagation_{id}.html')


def prep_plot_running_mag(rows: int, cols: int):
    fig = psub.make_subplots(rows=rows, cols=cols)
    return fig


def plot_running_mag(fig: go.Figure, sim_data: options.SimulationData, id: int):
    plot_mag = sim_data.magnetization_propagation[0, 0, 0].clone().detach().cpu()
    axis = sim_data.sample_axis.clone().detach().cpu()
    labels = ["x", "y", "z", "e", "mag xy", "phase xy"]
    colors = sample_colorscale('viridis', np.linspace(0.9, 0.1, 6))
    fig.add_trace(
        go.Scatter(x=axis, y=torch.norm(plot_mag[:, :2], dim=1),
                   name=f"{labels[-2]}",
                   line=dict(color=colors[-1]), fill='tozeroy'),
        row=id+1, col=1
    )
    fig.add_trace(
        go.Scatter(x=axis, y=torch.angle(
            plot_mag[:, 0] + 1j * plot_mag[:, 1]) / torch.pi,
                   name=f"{labels[-1]} [$\pi$]",
                   line=dict(color=colors[-2]), fill='tozeroy'),
        row=id+1, col=1
    )
    for k in range(4):
        fig.add_trace(
            go.Scatter(x=axis, y=plot_mag[:, k],
                       name=f"mag_{labels[k]}",
                       line=dict(color=colors[k])),
            row=id+1, col=1
        )
    return fig


def display_running_plot(fig):
    plotly.offline.plot(fig, filename=f'./optim/grad_pulse_propagation.html')
