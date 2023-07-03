""" define available sequence simulations"""
from emc_sim import options, prep, functions, plotting
import time
import logging
import torch
import tqdm

logModule = logging.getLogger(__name__)


def mese(sim_params: options.SimulationParameters, sim_data: options.SimulationData):
    """ want to set up gradients and pulses like in the mese standard protocol"""
    logModule.debug(f"Start Simulation: params {sim_data.get_run_params()}")
    t_start = time.time()
    plot_idx = 0

    # set up data carry
    tmp_data = options.SimTempData(sim_params=sim_params)
    tmp_data.run = sim_data
    # set up running plot
    fig = plotting.prep_plot_running_mag(10, 1)
    # prep pulse grad data
    gp_excitation, gps_refocus, timing, acquisition = prep.prep_gradient_pulse_mese(
        sim_params=sim_params, sim_tmp_data=tmp_data
    )

    fig = plotting.plot_running_mag(fig, tmp_data, rows=9, cols=1, id=plot_idx)
    plot_idx += 1

    # --- starting sim matrix propagation --- #
    logModule.debug("excitation")
    tmp_data.magnetization_propagation = functions.propagte_grad_pulse(
        mag_tensor=tmp_data.magnetization_propagation,
        grad_t=gp_excitation.data_grad, pulse_t=gp_excitation.data_pulse,
        dt_us=gp_excitation.dt_sampling_steps, sample_axis=tmp_data.sample_axis,
        t1_s=tmp_data.run.t1_s, t2_s=tmp_data.run.t2_s
    )

    fig = plotting.plot_running_mag(fig, tmp_data, rows=9, cols=1, id=plot_idx)
    plot_idx += 1
    for loop_idx in tqdm.trange(sim_params.sequence.ETL):
        # ----- refocusing loop - echo train -----
        logModule.debug(f'run {loop_idx + 1}')

        # delay before pulse
        tmp_data.magnetization_propagation = functions.matrix_propagation_relaxation(
            mag_tensor=tmp_data.magnetization_propagation,
            dt_us=timing.time_pre_pulse[loop_idx],
            t1_s=tmp_data.run.t1_s, t2_s=tmp_data.run.t2_s)

        # pulse
        tmp_data.magnetization_propagation = functions.propagte_grad_pulse(
            mag_tensor=tmp_data.magnetization_propagation,
            grad_t=gps_refocus[loop_idx].data_grad,
            pulse_t=gps_refocus[loop_idx].data_pulse,
            dt_us=gps_refocus[loop_idx].dt_sampling_steps,
            sample_axis=tmp_data.sample_axis,
            t1_s=tmp_data.run.t1_s, t2_s=tmp_data.run.t2_s
        )

        # if simParams.config.debuggingFlag and simParams.config.visualize:
        fig = plotting.plot_running_mag(fig, tmp_data, rows=9, cols=1, id=plot_idx)
        plot_idx += 1

        # delay after pulse
        tmp_data.magnetization_propagation = functions.matrix_propagation_relaxation(
            mag_tensor=tmp_data.magnetization_propagation,
            dt_us=timing.time_post_pulse[loop_idx], t1_s=tmp_data.run.t1_s, t2_s=tmp_data.run.t2_s)

        # acquisition
        for acq_idx in range(sim_params.settings.acquisition_number):
            tmp_data.magnetization_propagation = functions.propagte_grad_pulse(
                mag_tensor=tmp_data.magnetization_propagation,
                grad_t=acquisition.data_grad,
                pulse_t=acquisition.data_pulse,
                dt_us=acquisition.dt_sampling_steps,
                sample_axis=tmp_data.sample_axis, t1_s=tmp_data.run.t1_s, t2_s=tmp_data.run.t2_s
            )

            mag_data_cmplx = tmp_data.magnetization_propagation[:, 0] + 1j * tmp_data.magnetization_propagation[:, 1]
            tmp_data.signal_tensor[loop_idx, acq_idx] = torch.divide(
                torch.sum(mag_data_cmplx),
                100 * sim_params.settings.length_z / sim_params.settings.sample_number
            )

    logModule.debug('Signal array processing fourier')
    image_tensor = torch.fft.fftshift(torch.fft.fft(torch.fft.fftshift(tmp_data.signal_tensor)))
    sim_data.emc_signal_mag = 2 * torch.sum(torch.abs(image_tensor), axis=1) / sim_params.settings.acquisition_number
    sim_data.emc_signal_phase = 2 * torch.sum(torch.angle(image_tensor), axis=1) / sim_params.settings.acquisition_number

    if sim_params.sequence.ETL % 2 > 0:
        # for some reason we get a shift from the fft when used with odd array length.
        sim_data.emc_signal_mag = torch.roll(sim_data.emc_signal_mag, 1)
        sim_data.emc_signal_phase = torch.roll(sim_data.emc_signal_phase, 1)

    sim_data.time = time.time() - t_start

    plotting.display_running_plot(fig)

    return sim_data, sim_params

