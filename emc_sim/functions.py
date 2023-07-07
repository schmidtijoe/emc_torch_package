import torch
from scipy import stats
import numpy as np
from emc_sim import options, prep



def pulseCalibrationIntegral(pulse: torch.tensor,
                             deltaT: float,
                             simParams: options.SimulationParameters,
                             excitation: bool,
                             pulseNumber: int = 0) -> torch.tensor:
    """
    Calibrates pulse waveform for given flip angle, adds phase if given
    """
    # get different b1s to simulate for
    b1_vals = torch.tensor(simParams.settings.b1_list)
    # normalize
    b1Pulse = pulse / torch.norm(pulse)
    # integrate (discrete steps) total flip angle achieved with the normalized pulse
    flipAngleNormalizedB1 = torch.sum(torch.abs(b1Pulse * simParams.sequence.gamma_pi)) * deltaT * 1e-6
    if excitation:
        angleFlip = simParams.sequence.excitation_angle
        phase = simParams.sequence.excitation_phase / 180.0 * torch.pi
    else:
        # excitation pulse always 0th pulse
        angleFlip = simParams.sequence.refocus_angle[pulseNumber - 1]
        phase = simParams.sequence.refocus_phase[pulseNumber - 1] / 180.0 * torch.pi
    angleFlip *= torch.pi / 180 * b1_vals  # calculate with applied actual flip angle offset
    b1PulseCalibrated = b1Pulse[None, :] * (angleFlip[:, None] / flipAngleNormalizedB1) * \
                        torch.exp(torch.tensor(1j * phase))
    return b1PulseCalibrated


def propagate_matrix_mag_vector(propagation_matrix: torch.tensor, sim_data: options.SimulationData):
    """
    we setup the simulation around propagation matrices with dimensions:
        [t1s: i, t2s: j, b1s: k, samples: l, 4: m, 4: n]
    The propagation vector consequently propagates in 4d for all dictionary entries
        [t1s: i, t2s: j, b1s:k, samples: l, 4: n]
    this function just scripts the matrix multiplication
    """
    sim_data.magnetization_propagation = torch.einsum('ijklmn, ijkln -> ijklm',
                                                      propagation_matrix,
                                                      sim_data.magnetization_propagation)
    return sim_data


def propagate_gradient_pulse_relax(
        pulse_x: torch.tensor,
        pulse_y: torch.tensor,
        grad: torch.tensor,
        sim_data: options.SimulationData,
        dt_s: torch.tensor) -> torch.tensor:
    # get pulse matrices per step
    propagation_matrix_steps = matrix_propagation_grad_pulse_multidim(
        pulse_x=pulse_x, pulse_y=pulse_y, grad=grad, dt_s=dt_s, sim_data=sim_data)
    # iterate through the pulse grad shape and multiply the matrix propagators
    # dims [num_t1s, num_t2s, num_b1s, num_samples, num_pulse steps, 4, 4]
    propagation_matrix = torch.eye(4)[None, None, None, None].to(sim_data.device)
    for i in range(propagation_matrix_steps.shape[4]):
        propagation_matrix = torch.matmul(propagation_matrix, propagation_matrix_steps[:, :, :, :, i])
    return propagation_matrix


def setup_rot_mats(angle: torch.tensor):
    if angle.shape.__len__() < 1:
        a_shape = 1
    else:
        a_shape = angle.shape
    device = angle.device
    t_0 = torch.zeros(a_shape).to(device)
    t_1 = torch.ones(a_shape).to(device)
    return a_shape, t_0, t_1


def rotation_matrix_x(angle: torch.tensor):
    a_shape, t_0, t_1 = setup_rot_mats(angle)
    rot_matrix = torch.stack([
        torch.stack([t_1, t_0, t_0, t_0]),
        torch.stack([t_0, torch.cos(angle), -torch.sin(angle), t_0]),
        torch.stack([t_0, torch.sin(angle), torch.cos(angle), t_0]),
        torch.stack([t_0, t_0, t_0, t_1])
    ])
    rot_matrix = torch.movedim(rot_matrix, -1, 0)
    if a_shape.__len__() > 1:
        rot_matrix = torch.movedim(rot_matrix, -1, 0)
    return rot_matrix


def rotation_matrix_y(angle: torch.tensor):
    a_shape, t_0, t_1 = setup_rot_mats(angle)
    rot_matrix = torch.stack([
        torch.stack([torch.cos(angle), t_0, torch.sin(angle), t_0]),
        torch.stack([t_0, t_1, t_0, t_0]),
        torch.stack([-torch.sin(angle), t_0, torch.cos(angle), t_0]),
        torch.stack([t_0, t_0, t_0, t_1])
    ])
    rot_matrix = torch.movedim(rot_matrix, -1, 0)
    if a_shape.__len__() > 1:
        rot_matrix = torch.movedim(rot_matrix, -1, 0)
    return rot_matrix


def rotation_matrix_z(angle: torch.tensor):
    a_shape, t_0, t_1 = setup_rot_mats(angle)
    rot_matrix = torch.stack([
        torch.stack([torch.cos(angle), -torch.sin(angle), t_0, t_0]),
        torch.stack([torch.sin(angle), torch.cos(angle), t_0, t_0]),
        torch.stack([t_0, t_0, t_1, t_0]),
        torch.stack([t_0, t_0, t_0, t_1])
    ])
    rot_matrix = torch.movedim(rot_matrix, -1, 0)
    if a_shape.__len__() > 1:
        rot_matrix = torch.movedim(rot_matrix, -1, 0)
    return rot_matrix


def matrix_propagation_grad_pulse_multidim(
        sim_data: options.SimulationData,
        pulse_x: torch.tensor, pulse_y: torch.tensor, grad: torch.tensor, dt_s: torch.tensor) -> torch.tensor:
    """ Calculate the propagation matrix per time step dt, with one amplitude value (complex) for the pulse
    (hard pulse approximation) and an amplitude value for the gradient"""
    # calculate rotation around x
    angle_x = 2 * torch.pi * dt_s * sim_data.gamma * pulse_x
    r_x = rotation_matrix_x(sim_data.b1_vals[:, None] * angle_x)
    # calculate rotation around y
    angle_y = 2 * torch.pi * dt_s * sim_data.gamma * pulse_y
    r_y = rotation_matrix_y(sim_data.b1_vals[:, None] * angle_y)
    # calculate rotation around z
    angle_z = 2 * torch.pi * dt_s * sim_data.gamma * grad[None, :] * 1e-3 * sim_data.sample_axis[:, None]
    r_z = rotation_matrix_z(angle_z)
    # apply all
    r_xy = torch.matmul(r_x, r_y)
    # r_xy dims [num_b1s, num_pulse steps, 4, 4]
    # r_z dims [num_samples, num_pulse_steps, 4, 4]
    propagation_matrix = torch.matmul(r_z[None, :], r_xy[:, None])
    # dims [num_b1s, num_samples, num_pulse steps, 4, 4]
    relaxation_matrix = matrix_propagation_relaxation_multidim(dt_s=dt_s, sim_data=sim_data)
    # dims [num_t1s, num_t2s, 4, 4]
    return torch.matmul(propagation_matrix[None, None, :], relaxation_matrix[:, :, None, None, None])


def matrix_propagation_relaxation_multidim(dt_s: torch.tensor, sim_data: options.SimulationData) -> torch.tensor:
    e1 = torch.exp(-dt_s / sim_data.t1_vals)[:, None].repeat(1, sim_data.t2_vals.shape[0])
    e2 = torch.exp(-dt_s / sim_data.t2_vals)[None, :].repeat(sim_data.t1_vals.shape[0], 1)
    t_0 = torch.zeros(e1.shape, device=sim_data.device)
    t_1 = torch.ones(e1.shape, device=sim_data.device)
    relax_matrix = torch.stack([
        torch.stack([e2, t_0, t_0, t_0]),
        torch.stack([t_0, e2, t_0, t_0]),
        torch.stack([t_0, t_0, e1, t_1 - e1]),
        torch.stack([t_0, t_0, t_0, t_1])
    ])
    relax_matrix = torch.moveaxis(relax_matrix, -1, 0)
    relax_matrix = torch.moveaxis(relax_matrix, -1, 0)
    return relax_matrix


def sample_acquisition(etl_idx: int, sim_params: options.SimulationParameters, sim_data: options.SimulationData,
                       acquisition_matrix_propagator: torch.tensor):
    # acquisition
    for acq_idx in range(sim_params.settings.acquisition_number):
        sim_data = propagate_matrix_mag_vector(acquisition_matrix_propagator, sim_data)
        # remember dims [t1s, t2s, b1s, sample, 4]
        mag_data_cmplx = magnetization[:, :, :, :, 0] + 1j * magnetization[:, :, :, :, 1]
        # signal tensor [t1s, t2s, b1s, etl, acq_num]
        sim_data.signal_tensor[:, :, :, etl_idx, acq_idx] = torch.divide(
            torch.sum(mag_data_cmplx),
            100 * sim_params.settings.length_z / sim_params.settings.sample_number
        )
    return sim_data


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
    # ToDo: test case
