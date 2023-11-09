from emc_sim import options, blocks, functions
from emc_sim.simulations.base import Simulation
import torch
import logging
import time
log_module = logging.getLogger(__name__)


class MEGESSE(Simulation):
    def __init__(self, sim_params: options.SimulationParameters,
                 device: torch.device = torch.device("cpu"), num_mag_evol_plot: int = 10):
        super().__init__(sim_params=sim_params, device=device, num_mag_evol_plot=num_mag_evol_plot)

    def _prep(self):
        """ want to set up gradients and pulses like in the mese standard protocol
        For this we need all parts that are distinct and then set them up to pulss the calculation through
        """
        self.gp_excitation = blocks.GradPulse.prep_grad_pulse(
            pulse_type='Excitation',
            pulse_number=0,
            sym_spoil=False,
            params=self.params,
            balanced_read_grads=False
        )

        gp_refocus_1 = blocks.GradPulse.prep_grad_pulse(
            pulse_type='Refocusing_1',
            pulse_number=1,
            sym_spoil=False,
            params=self.params,
            balanced_read_grads=False
        )
        # built list of grad_pulse events, acquisition and timing
        grad_pulses = [gp_refocus_1]
        for r_idx in torch.arange(2, self.params.sequence.etl + 1):
            gp_refocus = blocks.GradPulse.prep_grad_pulse(
                pulse_type='Refocusing',
                pulse_number=r_idx,
                sym_spoil=True,
                params=self.params,
                balanced_read_grads=False
            )
            grad_pulses.append(gp_refocus)

        self.gps_refocus = grad_pulses

        self.gp_acquisition = blocks.GradPulse.prep_acquisition(params=self.params)

        log_module.debug(f"calculate timing")
        self.timing = blocks.Timing.build_fill_timing_mese(self.params)
        if self.params.config.visualize:
            self.gp_excitation.plot(self.data)
            self.gps_refocus[0].plot(self.data)
            self.gps_refocus[1].plot(self.data)
            self.gp_acquisition.plot(self.data)

        # set devices
        self.gp_excitation.set_device(self.device)
        self.timing.set_device(self.device)
        for gp in self.gps_refocus:
            gp.set_device(self.device)
        self.gp_acquisition.set_device(self.device)

    def simulate(self):
        log_module.debug(f"Simulating MEGESSE sequence")
        # t_start = time.time()
        # # --- starting sim matrix propagation --- #
        # log_module.debug("calculate matrix propagation")
        # # excitation
        # sim_data = functions.propagate_gradient_pulse_relax(
        #     pulse_x=gp_excitation.data_pulse_x, pulse_y=gp_excitation.data_pulse_y, grad=gp_excitation.data_grad,
        #     sim_data=sim_data, dt_s=gp_excitation.dt_sampling_steps * 1e-6
        # )
        # # plot excitation profile
        # fig = plotting.plot_running_mag(fig, sim_data, id=plot_idx)
        # plot_idx += 1
        #
        # # sample partial fourier gradient echo readout
        # # ToDo
        #
        # # calculate timing matrices (there are only 4)
        # # first refocus
        # mat_prop_ref1_pre_time = functions.matrix_propagation_relaxation_multidim(
        #     dt_s=timing.time_pre_pulse[0] * 1e-6, sim_data=sim_data
        # )
        # mat_prop_ref1_post_time = functions.matrix_propagation_relaxation_multidim(
        #     dt_s=timing.time_post_pulse[0] * 1e-6, sim_data=sim_data
        # )
        # # other refocus
        # mat_prop_ref_pre_time = functions.matrix_propagation_relaxation_multidim(
        #     dt_s=timing.time_pre_pulse[1] * 1e-6, sim_data=sim_data
        # )
        # mat_prop_ref_post_time = functions.matrix_propagation_relaxation_multidim(
        #     dt_s=timing.time_post_pulse[1] * 1e-6, sim_data=sim_data
        # )
        # log_module.debug("loop through refocusing")
        # with tqdm.trange(sim_params.sequence.etl) as t:
        #     t.set_description(f"ref pulse")
        #     for loop_idx in t:
        #         # timing
        #         if loop_idx == 0:
        #             m_p_pre = mat_prop_ref1_pre_time
        #         else:
        #             m_p_pre = mat_prop_ref_pre_time
        #         sim_data = functions.propagate_matrix_mag_vector(m_p_pre, sim_data=sim_data)
        #         # pulse
        #         sim_data = functions.propagate_gradient_pulse_relax(
        #             pulse_x=gps_refocus[loop_idx].data_pulse_x, pulse_y=gps_refocus[loop_idx].data_pulse_y,
        #             grad=gps_refocus[loop_idx].data_grad, sim_data=sim_data,
        #             dt_s=gps_refocus[loop_idx].dt_sampling_steps * 1e-6
        #         )
        #         # timing
        #         if loop_idx == 0:
        #             m_p_post = mat_prop_ref1_post_time
        #         else:
        #             m_p_post = mat_prop_ref_post_time
        #         sim_data = functions.propagate_matrix_mag_vector(m_p_post, sim_data=sim_data)
        #
        #         # ToDo acquisition gre
        #         sim_data = functions.sample_acquisition(
        #             etl_idx=loop_idx, sim_params=sim_params, sim_data=sim_data,
        #             acquisition_grad=acquisition.data_grad, dt_s=acquisition.dt_sampling_steps * 1e-6
        #         )
        #         fig = plotting.plot_running_mag(fig, sim_data, id=plot_idx)
        #         plot_idx += 1
        #         # ToDo acquisition se
        #         sim_data = functions.sample_acquisition(
        #             etl_idx=loop_idx, sim_params=sim_params, sim_data=sim_data,
        #             acquisition_grad=acquisition.data_grad, dt_s=acquisition.dt_sampling_steps * 1e-6
        #         )
        #         fig = plotting.plot_running_mag(fig, sim_data, id=plot_idx)
        #         plot_idx += 1
        #         # ToDo acquisition gre
        #         sim_data = functions.sample_acquisition(
        #             etl_idx=loop_idx, sim_params=sim_params, sim_data=sim_data,
        #             acquisition_grad=acquisition.data_grad, dt_s=acquisition.dt_sampling_steps * 1e-6
        #         )
        #         fig = plotting.plot_running_mag(fig, sim_data, id=plot_idx)
        #         plot_idx += 1
        #
        #         fig = plotting.plot_running_mag(fig, sim_data, id=plot_idx)
        #         plot_idx += 1
        #
        # log_module.debug('Signal array processing fourier')
        # image_tensor = torch.fft.ifftshift(
        #     torch.fft.ifft(sim_data.signal_tensor, dim=-1),
        #     dim=-1
        # )
        # sim_data.emc_signal_mag = 2 * torch.sum(torch.abs(image_tensor), dim=-1) / sim_params.settings.acquisition_number
        # sim_data.emc_signal_phase = 2 * torch.sum(torch.angle(image_tensor),
        #                                           dim=-1) / sim_params.settings.acquisition_number
        #
        # if sim_params.sequence.etl % 2 > 0:
        #     # for some reason we get a shift from the fft when used with odd array length.
        #     sim_data.emc_signal_mag = torch.roll(sim_data.emc_signal_mag, 1)
        #     sim_data.emc_signal_phase = torch.roll(sim_data.emc_signal_phase, 1)
        #
        # sim_data.time = time.time() - t_start
        #
        # plotting.display_running_plot(fig)
        pass
