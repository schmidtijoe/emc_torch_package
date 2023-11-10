from emc_torch import options, blocks, functions, plotting
from emc_torch.simulations.base import Simulation
import torch
import logging
import time
import tqdm

log_module = logging.getLogger(__name__)


class MESE(Simulation):
    def __init__(self, sim_params: options.SimulationParameters):
        super().__init__(sim_params=sim_params)

    def _prep(self):
        log_module.info("\t - MESE sequence")
        # prep pulse grad data - this holds the pulse data and timings
        log_module.info('\t - pulse gradient preparation')
        self.gp_excitation = blocks.GradPulse.prep_grad_pulse_excitation(params=self.params)

        gp_refocus_1 = blocks.GradPulse.prep_grad_pulse_refocus(params=self.params, refocus_pulse_number=0)

        # built list of grad_pulse events, acquisition and timing
        self.gps_refocus = [gp_refocus_1]
        for r_idx in torch.arange(1, self.params.sequence.etl):
            gp_refocus = blocks.GradPulse.prep_grad_pulse_refocus(params=self.params, refocus_pulse_number=r_idx)
            self.gps_refocus.append(gp_refocus)

        log_module.info(f"\t - calculate sequence timing")
        self.timing = blocks.Timing.build_fill_timing_mese(self.params)

        if self.params.config.visualize:
            log_module.info("\t - plot grad pulse data")
            self.gp_excitation.plot(sim_data=self.data, fig_path=self.fig_path)
            self.gps_refocus[0].plot(sim_data=self.data, fig_path=self.fig_path)
            self.gps_refocus[1].plot(sim_data=self.data, fig_path=self.fig_path)
            self.gp_se_acquisition.plot(sim_data=self.data, fig_path=self.fig_path)

    def _set_device(self):
        # set devices
        self.gp_excitation.set_device(self.device)
        self.timing.set_device(self.device)
        for gp in self.gps_refocus:
            gp.set_device(self.device)
        self.gp_se_acquisition.set_device(self.device)

    def _simulate(self):
        """ want to set up gradients and pulses like in the mese standard protocol
        For this we need all parts that are distinct and then set them up to pulse the calculation through
        """
        log_module.info("__ Simulating MESE Sequence __")
        t_start = time.time()
        # --- starting sim matrix propagation --- #
        log_module.debug("calculate matrix propagation")
        # excitation
        self.data = functions.propagate_gradient_pulse_relax(
            pulse_x=self.gp_excitation.data_pulse_x,
            pulse_y=self.gp_excitation.data_pulse_y,
            grad=self.gp_excitation.data_grad,
            sim_data=self.data, dt_s=self.gp_excitation.dt_sampling_steps_us * 1e-6
        )
        if self.params.config.visualize:
            # save excitation profile snapshot
            self.set_magnetization_profile_snap(
                magnetization_profile=self.data.magnetization_propagation,
                snap_name="excitation"
            )

        # calculate timing matrices (there are only 4)
        # first refocus
        mat_prop_ref1_pre_time = functions.matrix_propagation_relaxation_multidim(
            dt_s=self.timing.time_pre_pulse[0] * 1e-6, sim_data=self.data
        )
        mat_prop_ref1_post_time = functions.matrix_propagation_relaxation_multidim(
            dt_s=self.timing.time_post_pulse[0] * 1e-6, sim_data=self.data
        )
        # other refocus
        mat_prop_ref_pre_time = functions.matrix_propagation_relaxation_multidim(
            dt_s=self.timing.time_pre_pulse[1] * 1e-6, sim_data=self.data
        )
        mat_prop_ref_post_time = functions.matrix_propagation_relaxation_multidim(
            dt_s=self.timing.time_post_pulse[1] * 1e-6, sim_data=self.data
        )
        log_module.debug("loop through refocusing")
        with tqdm.trange(self.params.sequence.etl) as t:
            t.set_description(f"processing sequence, refocusing pulse loop")
            for loop_idx in t:
                # timing
                if loop_idx == 0:
                    m_p_pre = mat_prop_ref1_pre_time
                else:
                    m_p_pre = mat_prop_ref_pre_time
                self.data = functions.propagate_matrix_mag_vector(m_p_pre, sim_data=self.data)
                # pulse
                self.data = functions.propagate_gradient_pulse_relax(
                    pulse_x=self.gps_refocus[loop_idx].data_pulse_x,
                    pulse_y=self.gps_refocus[loop_idx].data_pulse_y,
                    grad=self.gps_refocus[loop_idx].data_grad,
                    sim_data=self.data,
                    dt_s=self.gps_refocus[loop_idx].dt_sampling_steps_us * 1e-6
                )
                # timing
                if loop_idx == 0:
                    m_p_post = mat_prop_ref1_post_time
                else:
                    m_p_post = mat_prop_ref_post_time
                self.data = functions.propagate_matrix_mag_vector(m_p_post, sim_data=self.data)

                if self.params.config.visualize:
                    # save profile snapshot after pulse
                    self.set_magnetization_profile_snap(
                        magnetization_profile=self.data.magnetization_propagation,
                        snap_name=f"refocus_{loop_idx+1}_post_pulse"
                    )

                # acquisition
                if self.params.config.signal_fourier_sampling:
                    # sample the acquisition with the readout gradient moved to into the slice direction
                    self.data = functions.sample_acquisition(
                        etl_idx=loop_idx, sim_params=self.params, sim_data=self.data,
                        acquisition_grad=self.gp_se_acquisition.data_grad,
                        dt_s=self.gp_se_acquisition.dt_sampling_steps_us * 1e-6
                    )
                    # basically this gives us a signal evolution in k-space binned into spatial resolution bins
                else:
                    # take the sum of the contributions of the individual spins at central readout echo time
                    self.data = functions.sum_sample_acquisition(
                        etl_idx=loop_idx, sim_params=self.params, sim_data=self.data,
                        acquisition_duration_s=self.params.sequence.duration_acquisition * 1e-6
                    )

                if self.params.config.visualize:
                    # save excitation profile snapshot
                    self.set_magnetization_profile_snap(
                        magnetization_profile=self.data.magnetization_propagation,
                        snap_name=f"refocus_{loop_idx + 1}_post_acquisition"
                    )

        if self.params.config.signal_fourier_sampling:
            log_module.debug('Signal array processing fourier')
            image_tensor = torch.fft.fftshift(
                torch.fft.ifft(
                    torch.fft.ifftshift(self.data.signal_tensor, dim=-1),
                    dim=-1
                ),
                dim=-1
            )
            if self.params.config.visualize:
                # plot slice sampling image tensor (binned slice profile)
                plotting.plot_slice_img_tensor(slice_image_tensor=image_tensor, sim_data=self.data,
                                               out_path=self.fig_path)

            self.data.emc_signal_mag = (2 * torch.abs(torch.sum(image_tensor, dim=-1)) /
                                        self.params.settings.acquisition_number)
            self.data.emc_signal_phase = torch.angle(torch.sum(image_tensor, dim=-1))

        self.data.time = time.time() - t_start
