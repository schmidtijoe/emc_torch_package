from emc_sim.simulations.base import Simulation
from emc_sim import options, blocks, functions
import torch
import logging
import time
import tqdm

log_module = logging.getLogger(__name__)


class MESE(Simulation):
    def __init__(self, sim_params: options.SimulationParameters,
                 device: torch.device = torch.device("cpu"), num_mag_evol_plot: int = 10):
        super().__init__(sim_params=sim_params, device=device, num_mag_evol_plot=num_mag_evol_plot)

    def _prep(self):
        # prep pulse grad data - this holds the pulse data and timings
        log_module.debug('prep module  -  pulse preparation')
        self.gp_excitation = blocks.GradPulse.prep_grad_pulse(
            pulse_type='Excitation',
            pulse_number=0,
            sym_spoil=False,
            params=self.params,
            orig_mese=False
        )

        gp_refocus_1 = blocks.GradPulse.prep_grad_pulse(
            pulse_type='Refocusing_1',
            pulse_number=1,
            sym_spoil=False,
            params=self.params,
            orig_mese=False
        )

        # built list of grad_pulse events, acquisition and timing
        self.gps_refocus = [gp_refocus_1]
        for r_idx in torch.arange(2, self.params.sequence.etl + 1):
            gp_refocus = blocks.GradPulse.prep_grad_pulse(
                pulse_type='Refocusing',
                pulse_number=r_idx,
                sym_spoil=True,
                params=self.params,
                orig_mese=False
            )
            self.gps_refocus.append(gp_refocus)

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
        """ want to set up gradients and pulses like in the mese standard protocol
        For this we need all parts that are distinct and then set them up to pulss the calculation through
        """
        log_module.info("Simulating MESE Sequence")
        t_start = time.time()
        # --- starting sim matrix propagation --- #
        log_module.debug("calculate matrix propagation")
        # excitation
        self.data = functions.propagate_gradient_pulse_relax(
            pulse_x=self.gp_excitation.data_pulse_x,
            pulse_y=self.gp_excitation.data_pulse_y,
            grad=self.gp_excitation.data_grad,
            sim_data=self.data, dt_s=self.gp_excitation.dt_sampling_steps * 1e-6
        )
        # plot excitation profile
        # self.fig = plotting.plot_running_mag(self.fig, self.data, id=self.plot_idx)
        # self.plot_idx += 1

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
            t.set_description(f"ref pulse")
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
                    dt_s=self.gps_refocus[loop_idx].dt_sampling_steps * 1e-6
                )
                # timing
                if loop_idx == 0:
                    m_p_post = mat_prop_ref1_post_time
                else:
                    m_p_post = mat_prop_ref_post_time
                self.data = functions.propagate_matrix_mag_vector(m_p_post, sim_data=self.data)

                # acquisition
                if self.params.config.signal_fourier_sampling:
                    # sample the acquisition with the readout gradient moved to into the slice direction
                    self.data = functions.sample_acquisition(
                        etl_idx=loop_idx, sim_params=self.params, sim_data=self.data,
                        acquisition_grad=self.gp_acquisition.data_grad,
                        dt_s=self.gp_acquisition.dt_sampling_steps * 1e-6
                    )
                    # basically this gives us a signal evolution in k-space binned into spatial resolution bins
                else:
                    # take the sum of the contributions of the individual spins at central readout echo time
                    self.data = functions.sum_sample_acquisition(
                        etl_idx=loop_idx, sim_params=self.params, sim_data=self.data,
                        acquisition_duration_s=self.params.sequence.duration_acquisition * 1e-6
                    )

                # self.fig = plotting.plot_running_mag(self.fig, self.data, id=self.plot_idx)
                # self.plot_idx += 1

        if self.params.config.signal_fourier_sampling:
            log_module.debug('Signal array processing fourier')
            image_tensor = torch.fft.ifftshift(
                torch.fft.ifft(self.data.signal_tensor, dim=-1),
                dim=-1
            )
            self.data.emc_signal_mag = (2 * torch.abs(torch.sum(image_tensor, dim=-1)) /
                                        self.params.settings.acquisition_number)
            self.data.emc_signal_phase = torch.angle(torch.sum(image_tensor, dim=-1))

            if self.params.sequence.etl % 2 > 0:
                # for some reason we get a shift from the fft when used with odd array length.
                self.data.emc_signal_mag = torch.roll(self.data.emc_signal_mag, 1)
                self.data.emc_signal_phase = torch.roll(self.data.emc_signal_phase, 1)

        self.data.time = time.time() - t_start
        #
        # plotting.display_running_plot(self.fig)
