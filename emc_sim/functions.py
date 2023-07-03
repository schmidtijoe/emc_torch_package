import torch
import plotly.graph_objects as go
import plotly.subplots as psub
from plotly.express.colors import sample_colorscale
import plotly
from scipy import stats
import numpy as np
from emc_sim import options

gamma = 42577478.518


def pulseCalibrationIntegral(pulse: torch.tensor,
                             deltaT: float,
                             simParams: options.SimulationParameters,
                             simTempData: options.SimTempData,
                             pulseNumber: int = 0) -> torch.tensor:
    """
    Calibrates pulse waveform for given flip angle, adds phase if given
    """
    # normalize
    b1Pulse = pulse / torch.norm(pulse)
    # integrate (discrete steps) total flip angle achieved with the normalized pulse
    flipAngleNormalizedB1 = torch.sum(torch.abs(b1Pulse) * simParams.sequence.gamma_pi) * deltaT * 1e-6
    if simTempData.excitation_flag:
        angleFlip = simParams.sequence.excitation_angle
        phase = simParams.sequence.excitation_phase / 180.0 * torch.pi
    else:
        # excitation pulse always 0th pulse
        angleFlip = simParams.sequence.refocus_angle[pulseNumber-1]
        phase = simParams.sequence.refocus_phase[pulseNumber-1] / 180.0 * torch.pi
    angleFlip *= torch.pi / 180 * simTempData.run.b1  # calculate with applied actual flip angle offset
    b1PulseCalibrated = b1Pulse * (angleFlip / flipAngleNormalizedB1) * torch.exp(torch.tensor(1j * phase))
    return b1PulseCalibrated


def propagte_grad_pulse(mag_tensor: torch.tensor, grad_t: torch.tensor, pulse_t: torch.tensor, dt_us: float,
                        sample_axis: torch.tensor, t1_s: float, t2_s: float) -> torch.tensor:
    dt_s = dt_us * 1e-6
    for idx_t in range(grad_t.shape[0]):
        mag_tensor = matrix_propagation_grad_pulse(mag_tensor, grad_t=grad_t[idx_t], pulse_t=pulse_t[idx_t], dt_s=dt_s,
                                                   sample_axis=sample_axis, t1_s=t1_s, t2_s=t2_s)
    return mag_tensor


def matrix_propagation_relaxation(mag_tensor: torch.tensor, dt_us: float, t1_s: float, t2_s: float) -> torch.tensor:
    t_dt = dt_us*1e-6
    e1 = np.exp(-t_dt / t1_s)
    e2 = np.exp(-t_dt / t2_s)

    relax_matrix = torch.tensor([
        [e2, 0, 0, 0], [0, e2, 0, 0], [0, 0, e1, 1 - e1], [0, 0, 0, 1]
    ])
    return torch.einsum('ij, ki -> kj', relax_matrix, mag_tensor)


def matrix_propagation_grad_pulse(mag_tensor: torch.tensor, grad_t: float, pulse_t: complex, dt_s: float,
                                  sample_axis: torch.tensor, t1_s: float, t2_s: float) -> torch.tensor:
    global gamma
    num_samples = sample_axis.shape[0]
    rot_axis_vector = torch.zeros((num_samples, 3))

    rot_axis_vector[:, 0] = pulse_t.real
    rot_axis_vector[:, 1] = pulse_t.imag
    rot_axis_vector[:, 2] = grad_t * sample_axis * 1e-3  # grad in mT/m
    # norm
    rot_angle = torch.norm(rot_axis_vector, dim=-1)
    rot_axis_vector = torch.divide(rot_axis_vector, rot_angle[:, None])
    # angle
    rot_angle *= torch.tensor(dt_s * gamma * 2 * torch.pi)
    co = torch.cos(rot_angle)[:, None]
    si = torch.sin(rot_angle)[:, None]
    # extract spatially varying part
    x = mag_tensor[:, :-1]
    # propagate arbitrary rotation
    A = rot_axis_vector * torch.einsum('ik, ik -> i', rot_axis_vector, x)[:, None]
    u_cross_x = torch.cross(rot_axis_vector, x)
    rot_propped = A + co * torch.cross(u_cross_x, rot_axis_vector) + si * u_cross_x
    mag_tensor[:, :3] = rot_propped
    return matrix_propagation_relaxation(mag_tensor, dt_us=1e6 * dt_s, t1_s=t1_s, t2_s=t2_s)


def generate_sample(axis: torch.tensor, extent: float):
    _sample = torch.from_numpy(stats.gennorm(24).pdf(axis / extent * 1.1) + 1e-6)
    _sample = torch.divide(_sample, torch.max(_sample))
    return _sample


if __name__ == '__main__':

    t1 = 1.5
    t2 = 0.035
    dt = 10e-6

    z_extend = 0.01  # m
    sample_num = 1200
    sample_axis = torch.linspace(-z_extend, z_extend, sample_num)
    grad_data = torch.from_numpy(np.load("../tests/grad_data.npy"))
    pulse_data = torch.from_numpy(np.load("../tests/pulse_data.npy"))

    sample = generate_sample(sample_axis, z_extend)

    magnetization = torch.zeros((sample_num, 4))
    for k in range(2):
        # set equilibrium and z start magnetization to sample values
        magnetization[:, 2 + k] = sample

    fig = psub.make_subplots(rows=2, cols=1)
    labels = ["x", "y", "z", "e", "abs xy"]
    colors = sample_colorscale('viridis', np.linspace(0.9, 0.1, 5))
    fig.add_trace(
            go.Scatter(x=sample_axis, y=torch.norm(magnetization[:, :2], dim=1),
                       name=f"mag_{labels[-1]} init",
                       line=dict(color=colors[-1]), fill='tozeroy'),
            row=1, col=1
        )
    for k in range(4):
        fig.add_trace(
            go.Scatter(x=sample_axis, y=magnetization[:, k],
                       name=f"mag_{labels[k]} init",
                       line=dict(color=colors[k])),
            row=1, col=1
        )

    mag_gpr = propagte_grad_pulse(
        mag_tensor=magnetization, grad_t=grad_data, pulse_t=pulse_data,
        t1_s=t1, t2_s=t2, dt_s=dt, sample_axis=sample_axis)

    fig.add_trace(
        go.Scatter(x=sample_axis, y=torch.norm(mag_gpr[:, :2], dim=1),
                   name=f"mag_{labels[-1]} init",
                   line=dict(color=colors[-1]), fill='tozeroy'),
        row=2, col=1
    )

    for k in range(4):
        fig.add_trace(
            go.Scatter(x=sample_axis, y=mag_gpr[:, k],
                       name=f"mag_{labels[k]} - after pulse",
                       line=dict(color=colors[k])),
            row=2, col=1
        )
    plotly.offline.plot(fig, filename='../tests/grad_pulse_propagation.html')
