import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as psub
from plotly.express.colors import sample_colorscale
import plotly.express as px
from emc_sim import options
import torch
import numpy as np
import logging
import pathlib as plib
log_module = logging.getLogger(__name__)


def plot_grad_pulse(px: torch.tensor, py: torch.tensor, g: torch.tensor, b1_vals: torch.tensor,
                    out_path: plib.Path | str, name: str):
    x_ax = torch.arange(px.shape[1])
    p_cplx = px + 1j * py
    p_abs = torch.abs(p_cplx)
    p_phase = torch.angle(p_cplx) / torch.pi
    fig = psub.make_subplots(
        rows=2, cols=1,
        specs=[[{"secondary_y": True}], [{}]]
    )
    for k in range(px.shape[0]):
        fig.add_trace(
            go.Scatter(x=x_ax, y=p_abs[k], name=f"p magnitude, b1: {b1_vals[k]:.2f}"),
            row=1, col=1,
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=x_ax, y=p_phase[k], name=f"p phase [pi], b1: {b1_vals[k]:.2f}"),
            row=1, col=1,
            secondary_y=True
        )
    fig.add_trace(
        go.Scatter(x=x_ax, y=g, name="slice grad", fill="tozeroy"),
        row=2, col=1
    )
    fig['layout']['title']['text'] = "Pulse - Gradient"
    fig.update_xaxes(title_text="sampling point")
    fig['layout']['yaxis']['title']['text'] = "magnitude [a.u.]"
    fig['layout']['yaxis2']['title']['text'] = "phase [pi]"
    fig['layout']['yaxis3']['title']['text'] = "gradient [mT/m]"

    out_path = plib.Path(out_path).absolute()
    fig_file = out_path.joinpath(f"plot_grad_pulse_{name}").with_suffix(".html")
    log_module.info(f"writing file: {fig_file.as_posix()}")
    fig.write_html(fig_file.as_posix())


def plot_signal_traces(sim_data: options.SimulationData):
    # plot at most 5 values
    num_t2s = min(sim_data.t2_vals.shape[0], 5)
    num_b1s = min(sim_data.b1_vals.shape[0], 5)
    num_echoes = min(sim_data.emc_signal_mag.shape[-2], 2)
    t2_vals = sim_data.t2_vals[:num_t2s]
    b1_vals = sim_data.b1_vals[:num_b1s]
    # dims signal tensor [t1s, t2s, b1s, echoes, sim sampling pts]
    mag_plot_x = torch.real(sim_data.signal_tensor[0, :, :, :num_echoes].clone().detach().cpu())
    mag_plot_y = torch.imag(sim_data.signal_tensor[0, :, :, :num_echoes].clone().detach().cpu())
    mag_plot_abs = torch.sqrt(mag_plot_y**2 + mag_plot_x**2)
    mag_plot_phase = torch.angle(sim_data.signal_tensor[0, :, :, :num_echoes].clone().detach().cpu()) / torch.pi
    # setup figure
    fig = psub.make_subplots(
        rows=num_t2s, cols=num_b1s,
        subplot_titles=[f"T2: {t2_*1e3:.1f} ms, B1: {b1_:.1f}" for t2_ in t2_vals for b1_ in b1_vals],
        specs=[[{"secondary_y": True}] * b1_vals] * t2_vals
    )
    x_ax = torch.arange(1, 1 + sim_data.signal_tensor.shape[-1])
    for idx_t2 in range(num_t2s):
        for idx_b1 in range(num_b1s):
            for idx_echo in range(num_echoes):
                trace_abs = mag_plot_abs[idx_t2, idx_b1, idx_echo] / torch.max(mag_plot_abs[idx_t2, idx_b1, idx_echo])
                trace_phase = mag_plot_phase[idx_t2, idx_b1, idx_echo]
                fig.add_trace(
                    go.Scatter(x=x_ax, y=mag_plot_x[idx_t2, idx_b1, idx_echo], name=f"mag_x echo {idx_echo+1}"),
                    row=1+idx_t2, col=1+idx_b1,
                    secondary_y=False
                )
                fig.add_trace(
                    go.Scatter(x=x_ax, y=mag_plot_y[idx_t2, idx_b1, idx_echo], name=f"mag_y echo {idx_echo+1}"),
                    row=1+idx_t2, col=1+idx_b1,
                    secondary_y=False
                )
                fig.add_trace(
                    go.Scatter(x=x_ax, y=trace_abs, fill="tozeroy", name=f"mag_signal echo {idx_echo+1}"),
                    row=1+idx_t2, col=1+idx_b1,
                    secondary_y=False
                )
                fig.add_trace(
                    go.Scatter(x=x_ax, y=trace_phase,
                               name=f"phase_signal echo {idx_echo+1}"),
                    row=1+idx_t2, col=1+idx_b1,
                    secondary_y=True
                )
    fig.update_layout(legend_title_text="simulated signal curves")
    fig.update_xaxes(title_text="virtual ADC sampling pt")
    fig.update_yaxes(title_text="magnitude [normalized a.u.]", secondary_y=False)
    fig.update_yaxes(title_text="phase [$pi]", secondary_y=True)

    fig.write_html(f'./tests/signal_traces.html')


def plot_emc_sim_data(sim_data: options.SimulationData):
    # plot at most 5 values
    num_t2s = min(sim_data.t2_vals.shape[0], 5)
    num_b1s = min(sim_data.b1_vals.shape[0], 5)
    t2_vals = sim_data.t2_vals[:num_t2s]
    b1_vals = sim_data.b1_vals[:num_b1s]
    # dims signal tensor [t1s, t2s, b1s, echoes, sim sampling pts]
    emc_mag = sim_data.emc_signal_mag[0].clone().detach().cpu()
    emc_phase = sim_data.emc_signal_phase[0].clone().detach().cpu()
    # setup figure
    fig = psub.make_subplots(
        rows=num_t2s, cols=num_b1s,
        subplot_titles=[f"T2: {t2_ * 1e3:.1f} ms, B1: {b1_:.1f}" for t2_ in t2_vals for b1_ in b1_vals],
        specs=[[{"secondary_y": True} for b1_ in b1_vals] for t2_ in t2_vals]
    )
    x_ax = torch.arange(1, 1 + sim_data.emc_signal_mag.shape[-1])
    for idx_t2 in range(num_t2s):
        for idx_b1 in range(num_b1s):
            fig.add_trace(
                go.Scatter(x=x_ax, y=emc_mag[idx_t2, idx_b1],
                           name=f"mag_emc"),
                row=1+idx_t2, col=1+idx_b1,
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=x_ax, y=emc_phase[idx_t2, idx_b1] / torch.pi,
                           name=f"phase_emc"),
                row=1+idx_t2, col=1+idx_b1,
                secondary_y=True
            )
    fig.update_layout(legend_title_text="EMC simulated curves")
    fig.update_xaxes(title_text="# Echo")
    fig.update_yaxes(title_text="Signal magnitude", secondary_y=False)
    fig.update_yaxes(title_text="Signal phase [$pi]", secondary_y=True)

    fig.write_html(f'./tests/emc_signal.html')


def plot_magnetization(mag_profile_df: pd.DataFrame, out_path: plib.Path | str, animate: bool = False,
                       slice_thickness_mm: float = 0.0, name: str = ""):
    if animate:
        fig = px.line(
            data_frame=mag_profile_df, x="sample_axis", y="mag_profile", color="dim", animation_frame="name"
        )
    else:
        fig = px.line(
            data_frame=mag_profile_df, x="sample_axis", y="mag_profile", color="dim",
            facet_col="name", facet_col_wrap=2
        )
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        if slice_thickness_mm > 1e-3:
            fig.add_vrect(x0=-slice_thickness_mm * 1e-3 / 2, x1=slice_thickness_mm * 1e-3 / 2,
                          annotation_text="desired slice", annotation_position="bottom right",
                          fillcolor="purple", opacity=0.25, line_width=0)
    # labels = ["x", "y", "z", "e", "abs xy"]
    # colors = sample_colorscale('viridis', np.linspace(0.9, 0.1, 5))
    # fig.add_trace(
    #     go.Scatter(x=axis, y=torch.norm(plot_mag[:, :2], dim=1),
    #                name=f"mag_{labels[-1]} init",
    #                line=dict(color=colors[-1]), fill='tozeroy'),
    #     row=1, col=1
    # )
    # for k in range(4):
    #     fig.add_trace(
    #         go.Scatter(x=axis, y=plot_mag[:, k],
    #                    name=f"mag_{labels[k]} init",
    #                    line=dict(color=colors[k])),
    #         row=1, col=1
    #     )

    out_path = plib.Path(out_path).absolute()
    fig_file = out_path.joinpath(f"plot_magnetization_propagation_{name}").with_suffix(".html")
    log_module.info(f"writing file: {fig_file.as_posix()}")
    fig.write_html(fig_file.as_posix())


def prep_plot_running_mag(rows: int, cols: int, t2: float, b1: float):
    fig = psub.make_subplots(rows=rows, cols=cols)
    fig.update_layout({"title": f"magnetization propagation, T2: {t2*1e3:.1f} ms, B1: {b1:.2f}"})
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
        row=id + 1, col=1
    )
    fig.add_trace(
        go.Scatter(x=axis, y=torch.angle(
            plot_mag[:, 0] + 1j * plot_mag[:, 1]) / torch.pi,
                   name=f"{labels[-1]} [$\pi$]",
                   line=dict(color=colors[-2]), fill='tozeroy'),
        row=id + 1, col=1
    )
    for k in range(4):
        fig.add_trace(
            go.Scatter(x=axis, y=plot_mag[:, k],
                       name=f"mag_{labels[k]}",
                       line=dict(color=colors[k])),
            row=id + 1, col=1
        )
    return fig


def display_running_plot(fig, name: str = ""):
    fig.write_html(f'./tests/{name}_grad_pulse_propagation.html')
