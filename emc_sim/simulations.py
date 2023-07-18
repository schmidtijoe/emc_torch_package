"""
define available sequence simulations

The simulations are setup to calculate the whole matrix propagation matrix for each entity:
e.g.: exciation pulse - relaxation time - refocus 1 - relaxation time - adc step - read
- relaxation time - refocus rest -> loop

This way we can reuse the propagation matrices when they reappear: eg. in the standard mese protocol the calculations
after the first refocussing pulse are identical. The pulses and timings are equal and can be expressed through one
propagation matrix. This then has to be distributed over the changing magnetization vectors and throughout
the acquisition.

The calculation is setup to scale across tensor dimensions. It is able to be run on the GPU. Different inputs can be
tracked with the requires_grad keyword to make them optimizable (magnetization profiles, pulses, etc.).
"""
from emc_sim import options, prep, functions, plotting
import time
import logging
import torch
import tqdm

logModule = logging.getLogger(__name__)


def fid(sim_params: options.SimulationParameters, fft: bool = True):
    """ want to simulate a pulse and readout only for comparing fft with sum approach"""
    # setup device
    device = torch.device("cpu")
    logModule.debug(f"torch device: {device}")
    # setup single values only
    sim_params.settings.t2_list = [50]
    sim_params.settings.t1_list = [1.5]
    sim_params.settings.b1_list = [0.8]
    sim_params.sequence.ETL = 1
    # no crushing - rephasing to calculate
    area_slice_select = sim_params.sequence.gradient_excitation * sim_params.sequence.duration_excitation
    area_rephase = - 0.5 * area_slice_select
    sim_params.sequence.gradient_excitation_rephase = area_rephase / sim_params.sequence.duration_excitation_rephase
    # get sim data
    sim_data = options.SimulationData.from_sim_parameters(sim_params=sim_params, device=device)

    # set up running plot and plot initial magnetization
    if fft:
        rows = 5
    else:
        rows = 3
    plot_idx = 0
    fig = plotting.prep_plot_running_mag(rows, 1, t2=sim_data.t2_vals[0], b1=sim_data.b1_vals[0])
    fig = plotting.plot_running_mag(fig, sim_data, id=plot_idx)
    plot_idx += 1
    # get pulse
    gp_excitation = prep.GradPulse.prep_single_grad_pulse(
        params=sim_params, excitation_flag=True, grad_rephase_factor=1.05
    )
    gp_excitation.plot(sim_data=sim_data)

    # get acquisition
    acquisition = prep.GradPulse.prep_acquisition(params=sim_params)
    # propagate pulse
    # pulse
    sim_data = functions.propagate_gradient_pulse_relax(
        pulse_x=gp_excitation.data_pulse_x, pulse_y=gp_excitation.data_pulse_y,
        grad=gp_excitation.data_grad, sim_data=sim_data,
        dt_s=gp_excitation.dt_sampling_steps * 1e-6
    )
    # plot excitation profile
    fig = plotting.plot_running_mag(fig, sim_data, id=plot_idx)
    plot_idx += 1
    # sample acquisition
    if fft:
        # prephase acquisition in case of fft
        pre_acq = prep.GradPulse.prep_acquisition(params=sim_params)
        pre_acq.data_grad = - pre_acq.data_grad
        pre_acq.duration = pre_acq.duration / 2
        sim_data = functions.propagate_gradient_pulse_relax(
            pulse_x=pre_acq.data_pulse_x, pulse_y=pre_acq.data_pulse_y, grad=pre_acq.data_grad,
            sim_data=sim_data, dt_s=pre_acq.duration * 1e-6
        )
        # plot dephased
        fig = plotting.plot_running_mag(fig, sim_data, id=plot_idx)
        plot_idx += 1
        # acquisition
        sim_data = functions.sample_acquisition(
            etl_idx=0, sim_params=sim_params, sim_data=sim_data,
            acquisition_grad=acquisition.data_grad, dt_s=acquisition.dt_sampling_steps * 1e-6
        )
        # plot after acquisition
        fig = plotting.plot_running_mag(fig, sim_data, id=plot_idx)
        plot_idx += 1
        logModule.debug('Signal array processing fourier')
        image_tensor = torch.fft.ifftshift(
            torch.fft.ifft(sim_data.signal_tensor, dim=-1)
        )
        sim_data.emc_signal_mag = 2 * torch.sum(torch.abs(image_tensor),
                                                dim=-1) / sim_params.settings.acquisition_number
        sim_data.emc_signal_phase = 2 * torch.sum(torch.angle(image_tensor),
                                                  dim=-1) / sim_params.settings.acquisition_number
        plotting.plot_signal_traces(sim_data)
        # rephase acquisition
        sim_data = functions.propagate_gradient_pulse_relax(
            pulse_x=pre_acq.data_pulse_x, pulse_y=pre_acq.data_pulse_y, grad=pre_acq.data_grad,
            sim_data=sim_data, dt_s=pre_acq.duration * 1e-6
        )
    else:
        # propagate half of acquisition time - to equalize with above we need half the time prephaser
        # and half the time for read to arrive at the echo
        relax_matrix = functions.matrix_propagation_relaxation_multidim(
            dt_s=acquisition.duration * 1e-6, sim_data=sim_data
        )
        sim_data = functions.propagate_matrix_mag_vector(relax_matrix, sim_data)
        # sum
        mag = torch.sum(sim_data.magnetization_propagation[0, 0, 0], dim=0)
        sim_data.emc_signal_mag[0, 0, 0] = torch.abs(mag[0] + 1j * mag[1])
        sim_data.emc_signal_phase[0, 0, 0, 0] = torch.angle(mag[0] + 1j * mag[1])
        # again propagate relaxation to arrive at same state (second half of read and rephasing)
        sim_data = functions.propagate_matrix_mag_vector(relax_matrix, sim_data)
    fig = plotting.plot_running_mag(fig, sim_data, id=plot_idx)
    plot_idx += 1
    plotting.display_running_plot(fig, f"fid_simulation_fft-{fft}")
    return sim_data, sim_params


def mese(sim_params: options.SimulationParameters):
    """ want to set up gradients and pulses like in the mese standard protocol
    For this we need all parts that are distinct and then set them up to pulss the calculation through
    """
    logModule.debug(f"Start Simulation")
    t_start = time.time()
    plot_idx = 0
    device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
    logModule.debug(f"torch device: {device}")
    logModule.debug(f"setup simulation data")
    # set up sample and initial magnetization + data carry
    sim_data = options.SimulationData.from_sim_parameters(sim_params=sim_params, device=device)

    # set up running plot and plot initial magnetization
    fig = plotting.prep_plot_running_mag(10, 1, t2=sim_data.t2_vals[0], b1=sim_data.b1_vals[0])
    fig = plotting.plot_running_mag(fig, sim_data, id=plot_idx)
    plot_idx += 1

    # prep pulse grad data - this holds the pulse data and timings
    gp_excitation, gps_refocus, timing, acquisition = prep.prep_gradient_pulse_mese(
        sim_params=sim_params
    )
    gp_excitation.plot(sim_data)
    gps_refocus[0].plot(sim_data)
    gps_refocus[1].plot(sim_data)
    acquisition.plot(sim_data)
    # set devices
    gp_excitation.set_device(device)
    timing.set_device(device)
    for gp in gps_refocus:
        gp.set_device(device)
    acquisition.set_device(device)

    # --- starting sim matrix propagation --- #
    logModule.debug("calculate matrix propagation")
    # excitation
    sim_data = functions.propagate_gradient_pulse_relax(
        pulse_x=gp_excitation.data_pulse_x, pulse_y=gp_excitation.data_pulse_y, grad=gp_excitation.data_grad,
        sim_data=sim_data, dt_s=gp_excitation.dt_sampling_steps * 1e-6
    )
    # plot excitation profile
    fig = plotting.plot_running_mag(fig, sim_data, id=plot_idx)
    plot_idx += 1

    # calculate timing matrices (there are only 4)
    # first refocus
    mat_prop_ref1_pre_time = functions.matrix_propagation_relaxation_multidim(
        dt_s=timing.time_pre_pulse[0] * 1e-6, sim_data=sim_data
    )
    mat_prop_ref1_post_time = functions.matrix_propagation_relaxation_multidim(
        dt_s=timing.time_post_pulse[0] * 1e-6, sim_data=sim_data
    )
    # other refocus
    mat_prop_ref_pre_time = functions.matrix_propagation_relaxation_multidim(
        dt_s=timing.time_pre_pulse[1] * 1e-6, sim_data=sim_data
    )
    mat_prop_ref_post_time = functions.matrix_propagation_relaxation_multidim(
        dt_s=timing.time_post_pulse[1] * 1e-6, sim_data=sim_data
    )
    logModule.debug("loop through refocusing")
    with tqdm.trange(sim_params.sequence.ETL) as t:
        t.set_description(f"ref pulse")
        for loop_idx in t:
            # timing
            if loop_idx == 0:
                m_p_pre = mat_prop_ref1_pre_time
            else:
                m_p_pre = mat_prop_ref_pre_time
            sim_data = functions.propagate_matrix_mag_vector(m_p_pre, sim_data=sim_data)
            # pulse
            sim_data = functions.propagate_gradient_pulse_relax(
                pulse_x=gps_refocus[loop_idx].data_pulse_x, pulse_y=gps_refocus[loop_idx].data_pulse_y,
                grad=gps_refocus[loop_idx].data_grad, sim_data=sim_data,
                dt_s=gps_refocus[loop_idx].dt_sampling_steps * 1e-6
            )
            # timing
            if loop_idx == 0:
                m_p_post = mat_prop_ref1_post_time
            else:
                m_p_post = mat_prop_ref_post_time
            sim_data = functions.propagate_matrix_mag_vector(m_p_post, sim_data=sim_data)

            # acquisition
            sim_data = functions.sample_acquisition(
                etl_idx=loop_idx, sim_params=sim_params, sim_data=sim_data,
                acquisition_grad=acquisition.data_grad, dt_s=acquisition.dt_sampling_steps * 1e-6
            )

            fig = plotting.plot_running_mag(fig, sim_data, id=plot_idx)
            plot_idx += 1

    logModule.debug('Signal array processing fourier')
    image_tensor = torch.fft.ifftshift(
        torch.fft.ifft(sim_data.signal_tensor, dim=-1),
        dim=-1
    )
    sim_data.emc_signal_mag = 2 * torch.sum(torch.abs(image_tensor), dim=-1) / sim_params.settings.acquisition_number
    sim_data.emc_signal_phase = 2 * torch.sum(torch.angle(image_tensor),
                                              dim=-1) / sim_params.settings.acquisition_number

    if sim_params.sequence.ETL % 2 > 0:
        # for some reason we get a shift from the fft when used with odd array length.
        sim_data.emc_signal_mag = torch.roll(sim_data.emc_signal_mag, 1)
        sim_data.emc_signal_phase = torch.roll(sim_data.emc_signal_phase, 1)

    sim_data.time = time.time() - t_start

    plotting.display_running_plot(fig)

    return sim_data, sim_params


def mese_optim(sim_params: options.SimulationParameters, sim_data: options.SimulationData,
               fa_input, gp_excitation, gp_refocusing, timing, acquisition):
    """ want to set up gradients and pulses like in the mese standard protocol
    For this we need all parts that are distinct and then set them up to pulss the calculation through
    """
    # --- starting sim matrix propagation --- #
    logModule.debug("calculate matrix propagation")
    # excitation
    sim_data = functions.propagate_gradient_pulse_relax(
        pulse_x=gp_excitation.data_pulse_x, pulse_y=gp_excitation.data_pulse_y, grad=gp_excitation.data_grad,
        sim_data=sim_data, dt_s=gp_excitation.dt_sampling_steps * 1e-6
    )

    # calculate equal timings only once
    mat_prop_ref1_pre_time = functions.matrix_propagation_relaxation_multidim(
        dt_s=timing.time_pre_pulse[0] * 1e-6, sim_data=sim_data
    )
    mat_prop_ref1_post_time = functions.matrix_propagation_relaxation_multidim(
        dt_s=timing.time_post_pulse[0] * 1e-6, sim_data=sim_data
    )
    mat_prop_ref_pre_time = functions.matrix_propagation_relaxation_multidim(
        dt_s=timing.time_pre_pulse[1] * 1e-6, sim_data=sim_data
    )
    mat_prop_ref_post_time = functions.matrix_propagation_relaxation_multidim(
        dt_s=timing.time_post_pulse[1] * 1e-6, sim_data=sim_data
    )

    logModule.debug("loop through refocusing")
    for loop_idx in range(sim_params.sequence.ETL):
        logModule.debug(f"ref pulse: {loop_idx + 1}")
        # timing
        if loop_idx == 0:
            m_p_pre = mat_prop_ref1_pre_time
        else:
            m_p_pre = mat_prop_ref_pre_time
        sim_data = functions.propagate_matrix_mag_vector(m_p_pre, sim_data=sim_data)

        # pulse
        px = gp_refocusing[loop_idx].data_pulse_x * fa_input[loop_idx]
        py = gp_refocusing[loop_idx].data_pulse_y * fa_input[loop_idx]
        sim_data = functions.propagate_gradient_pulse_relax(
            pulse_x=px, pulse_y=py,
            grad=gp_refocusing[loop_idx].data_grad, sim_data=sim_data,
            dt_s=gp_refocusing[loop_idx].dt_sampling_steps * 1e-6
        )
        # timing
        if loop_idx == 0:
            m_p_post = mat_prop_ref1_post_time
        else:
            m_p_post = mat_prop_ref_post_time
        sim_data = functions.propagate_matrix_mag_vector(m_p_post, sim_data=sim_data)

        # read steps
        # acquisition
        sim_data = functions.sample_acquisition(
            etl_idx=loop_idx, sim_params=sim_params, sim_data=sim_data,
            acquisition_grad=acquisition.data_grad, dt_s=acquisition.dt_sampling_steps * 1e-6
        )

    logModule.debug('Signal array processing fourier')
    image_tensor = torch.fft.ifftshift(
        torch.fft.ifft(sim_data.signal_tensor, dim=-1),
        dim=-1
    )
    sim_data.emc_signal_mag = 2 * torch.sum(torch.abs(image_tensor), dim=-1) / sim_params.settings.acquisition_number
    sim_data.emc_signal_phase = 2 * torch.sum(torch.angle(image_tensor),
                                              dim=-1) / sim_params.settings.acquisition_number

    if sim_params.sequence.ETL % 2 > 0:
        # for some reason we get a shift from the fft when used with odd array length.
        sim_data.emc_signal_mag = torch.roll(sim_data.emc_signal_mag, 1)
        sim_data.emc_signal_phase = torch.roll(sim_data.emc_signal_phase, 1)

    return sim_data, sim_params


def single_pulse(sim_params: options.SimulationParameters):
    """assume T2 > against pulse width"""
    device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
    logModule.debug(f"torch device: {device}")

    # set tensor of k value-tuples to simulate for, here only b1
    n_b1 = 1
    # b1_vals = torch.linspace(0.5, 1.4, n_b1)
    b1_vals = torch.tensor(1.0)
    n_t2 = 1
    # t2_vals_ms = torch.linspace(35, 50, n_t2)
    t2_vals_ms = torch.tensor(50)

    sim_params.settings.sample_number = 500
    sim_params.settings.length_z = 0.005
    sim_params.settings.t2_list = t2_vals_ms.tolist()
    sim_params.settings.b1_list = b1_vals.tolist()
    sim_data = options.SimulationData.from_sim_parameters(sim_params=sim_params, device=device)

    grad_pulse_data = prep.GradPulse.prep_single_grad_pulse(
        params=sim_params, excitation_flag=True, grad_rephase_factor=1.0
    )
    grad_pulse_data.set_device(device)

    plot_idx = 0
    fig = plotting.prep_plot_running_mag(2, 1, 0.05, 1.0)
    # excite only
    fig = plotting.plot_running_mag(fig, sim_data=sim_data, id=plot_idx)
    plot_idx += 1

    # --- starting sim matrix propagation --- #
    logModule.debug("excitation")
    sim_data = functions.propagate_gradient_pulse_relax(
        grad=grad_pulse_data.data_grad, pulse_x=grad_pulse_data.data_pulse_x,
        pulse_y=grad_pulse_data.data_pulse_y, dt_s=grad_pulse_data.dt_sampling_steps * 1e-6, sim_data=sim_data)

    fig = plotting.plot_running_mag(fig, sim_data, id=plot_idx)
    plot_idx += 1
    plotting.display_running_plot(fig)
