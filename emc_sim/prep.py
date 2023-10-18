""" prepare module to get and cast data needed for simulations"""
from emc_sim import options, functions, plotting
import logging
import torch
import pathlib as plib
import pypsi

log_module = logging.getLogger(__name__)


class GPDetails:
    duration_pulse: torch.tensor
    grad_amp: torch.tensor
    excitation_flag: bool
    grad_crush_rephase: torch.tensor
    duration_crush_rephase: torch.tensor
    pypsi_path: plib.Path

    @classmethod
    def get_gp_details(cls, params: options.SimulationParameters, excitation_flag: bool):
        gp_instance = cls()
        gp_instance.pypsi_path = plib.Path(params.config.pypsi_path).absolute()
        if excitation_flag:
            gp_instance.grad_crush_rephase = torch.tensor(params.sequence.gradient_excitation_rephase)
            gp_instance.duration_crush_rephase = torch.tensor(params.sequence.duration_excitation_rephase)
            gp_instance.duration_pulse = torch.tensor(params.sequence.duration_excitation)
            gp_instance.grad_amp = torch.tensor(params.sequence.gradient_excitation)
        else:
            gp_instance.grad_crush_rephase = torch.tensor(params.sequence.gradient_crush)
            gp_instance.duration_crush_rephase = torch.tensor(params.sequence.duration_crush)
            gp_instance.duration_pulse = torch.tensor(params.sequence.duration_refocus)
            gp_instance.grad_amp = torch.tensor(params.sequence.gradient_refocus)
        return gp_instance


class GradPulse:
    def __init__(self,
                 pulse_type: str = 'Excitation',
                 pulse_number: int = 0,
                 num_sampling_points: int = 512,
                 dt_sampling_steps: float = 5.0,
                 duration: float = 0.0):
        self.pulse_number = pulse_number
        self.pulse_type = pulse_type
        self.num_sampling_points = num_sampling_points
        self.dt_sampling_steps = dt_sampling_steps
        self.data_grad = torch.zeros(0)
        self.data_pulse_x = torch.zeros(0)
        self.data_pulse_y = torch.zeros(0)
        self.excitation_flag: bool = True
        # self.params = params
        # if sim_temp_data is not None:
        #     self.temp_data = sim_temp_data
        # else:
        #     self.temp_data = options.SimTempData(sim_params=params)
        self.duration = duration
        self.gp_details: GPDetails = GPDetails()

        # set and check
        self._set_exci_flag()

    def set_device(self, device: torch.device):
        self.data_pulse_x = self.data_pulse_x.to(device)
        self.data_pulse_y = self.data_pulse_y.to(device)
        self.data_grad = self.data_grad.to(device)

    def _set_exci_flag(self):
        if self.pulse_type == 'Excitation':
            self.excitation_flag = True
        else:
            self.excitation_flag = False

    def _set_float32(self):
        self.data_grad = self.data_grad.to(dtype=torch.float32)
        self.data_pulse_x = self.data_pulse_x.to(dtype=torch.float32)
        self.data_pulse_y = self.data_pulse_y.to(dtype=torch.float32)

    @classmethod
    def prep_grad_pulse(cls, pulse_type: str = 'Excitation', pulse_number: int = 0, sym_spoil: bool = True,
                        params: options.SimulationParameters = options.SimulationParameters(), orig_mese: bool = True):
        if orig_mese:
            # nothing changes
            pre_read_grad_amp = 0.0
        else:
            # if not original sequence schemes, the readouts are prephased and rephased
            # while spoiling takes place, in negative direction
            # calculate readout prephasing area necessary
            pre_read_area = - params.sequence.duration_acquisition * params.sequence.gradient_acquisition / 2.0
            # calculate amplitude added to spoiling, dependent on duration of spoiling
            pre_read_grad_amp = pre_read_area / params.sequence.duration_crush

        # -- prep pulse
        log_module.debug(f'prep pulse {pulse_type}; # {pulse_number}')
        grad_pulse = cls(pulse_type=pulse_type, pulse_number=pulse_number)
        # set flag
        excitation_flag: bool = (pulse_number == 0) & (pulse_type == 'Excitation')
        # get duration & grad pulse details
        gp_details = GPDetails.get_gp_details(params=params, excitation_flag=excitation_flag)
        # read rf
        rf = pypsi.Params.pulse.load(gp_details.pypsi_path)

        if abs(rf.duration_in_us - gp_details.duration_pulse) > 1e-5:
            rf.resample_to_duration(duration_in_us=int(gp_details.duration_pulse))
        pulse = torch.from_numpy(rf.amplitude) * torch.exp(torch.from_numpy(1j * rf.phase))

        # calculate and normalize
        pulse_from_pypsi = functions.pulseCalibrationIntegral(
            pulse=pulse,
            delta_t=rf.get_dt_sampling_in_us(),
            pulse_number=pulse_number,
            sim_params=params,
            excitation=excitation_flag)

        # ToDo: attention: prephasing artificial read!
        if not excitation_flag:
            # add prephasing readout if set - only applies to refocusing pulses with sym spoiling.
            # not applies to excitation pulses with sym spoiling
            gp_details.grad_crush_rephase += pre_read_grad_amp

        if sym_spoil:
            # symmetric spoiling, use spoiling grads after and before
            grad_prewind = gp_details.grad_crush_rephase
            duration_prewind = gp_details.duration_crush_rephase
        else:
            grad_prewind = 0.0
            duration_prewind = 0.0

        # build parts of pulse gradient
        grad, pulse, duration, area_grad = cls.build_pulse_grad_shapes(
            gp_details=gp_details,
            pulse=pulse_from_pypsi,
            duration_pre=duration_prewind,
            grad_amp_pre=grad_prewind
        )
        if params.config.signal_fourier_sampling:
            # We did change the excitation pulse to play a prephasing z gradient for the virtual k-space readout.
            # this relies on perfect 180 deg pulses and hence changes the acquisition phase with b1 offset
            # even though it is virtual only - this makes sense in the original siemens mese scheme as this
            # relies on perfect b1 also for the readout gradient - lets us estimate phase offset thus created
            if grad_pulse.excitation_flag & orig_mese:
                # Simulation is based on moving the acquisition process (hence gradient) artificially to
                # z-axis along the slice
                # Therefore we need to do a couple of things artificially:

                # when acquiring in k-space along the slice we need to move the k-space start to the corner of k-space
                # i.e.: prephase half an acquisition gradient moment, put it into the rephase timing
                gradient_pre_phase = params.sequence.gradient_acquisition * params.sequence.duration_acquisition / \
                                     (2 * params.sequence.duration_excitation_rephase)

                # the crushers are placed symmetrically about the refocusing pulses,
                # hence are cancelling each others k-space phase. We need to make sure that the crushers are balanced.
                # For timing reasons there is no crusher before the first refocusing pulse in the sequence.
                # We move one into the rephase space of the excitation
                # ToDo: does this make sense actually? sequence is designed already to balance the first crusher with
                #  the rephasing -> hence this is already meant to be the sum of both
                gradient_excitation_crush = params.sequence.gradient_crush * params.sequence.duration_crush / \
                                            params.sequence.duration_excitation_rephase

                # When exciting with a slice selective gradient the gradient creates phase offset along the slice axis.
                # We want to rephase this phase offset
                # (as is the original use of the gradient in the acquisition scheme).
                # However, the rephasing gradient is usually used with half the gradient moment area (at 90° pulses),
                # which is not quite accurate.
                # After investigation a manual correction term can be put in here for accuracy * 1.038
                # ToDo: see above, we basically recalculate the rephasing gradient that was there
                #  in the sequence already,
                #  can double check if it has the same amplitude!
                gradient_excitation_phase_rewind = - area_grad / (2 * params.sequence.duration_excitation_rephase)

                # The gradient pulse scheme needs to be re-done with accommodating those changes in the
                # rephase gradient of the excitation
                gp_details.grad_crush_rephase = torch.div(
                    gradient_pre_phase + gradient_excitation_crush + gradient_excitation_phase_rewind,
                    params.sequence.duration_excitation_rephase
                )
                grad, pulse, duration, area_grad = cls.build_pulse_grad_shapes(
                    gp_details=gp_details,
                    pulse=pulse_from_pypsi,
                    duration_pre=duration_prewind,
                    grad_amp_pre=grad_prewind)

        # assign vars
        grad_pulse.num_sampling_points = rf.num_samples
        grad_pulse.dt_sampling_steps = rf.get_dt_sampling_in_us()
        grad_pulse.data_grad = grad
        grad_pulse.data_pulse_x = pulse.real
        grad_pulse.data_pulse_y = pulse.imag
        grad_pulse.duration = rf.duration_in_us

        grad_pulse._set_float32()
        return grad_pulse

    @staticmethod
    def build_pulse_grad_shapes(
            gp_details: GPDetails, pulse: torch.tensor, duration_pre: float, grad_amp_pre: float):
        grad_amp_pre = torch.tensor(grad_amp_pre)
        """ want to build the shapes given slice gradient pre, spoil and slice select and align it to the given pulse"""
        # grad amplitudes are values, pulse is a shape already with complex numbers and
        # distributed across different b1 values -> pulse dim [# b1, # pulse sampling steps]
        if torch.abs(grad_amp_pre) < 1e-5:
            duration_pre = torch.zeros(1)
        else:
            duration_pre = torch.tensor(duration_pre)
        num_sample_pulse = pulse.shape[1]
        dt_us = gp_details.duration_pulse / num_sample_pulse
        # calculate total number of sampling points
        num_sample_pre = torch.nan_to_num(torch.div(
            duration_pre, dt_us, rounding_mode="trunc"
        )).type(torch.int)
        num_sample_crush = torch.nan_to_num(torch.div(
            gp_details.duration_crush_rephase, dt_us, rounding_mode="trunc"
        )).type(torch.int)
        num_sample_total = num_sample_pre + num_sample_pulse + num_sample_crush
        # allocate tensors
        grad_amp = torch.zeros(num_sample_total)
        pulse_amp = torch.zeros((pulse.shape[0], num_sample_total), dtype=torch.complex128)
        # fill
        grad_amp[:num_sample_pre] = grad_amp_pre
        grad_amp[num_sample_pre:num_sample_pre + num_sample_pulse] = gp_details.grad_amp
        grad_amp[num_sample_pre + num_sample_pulse:] = gp_details.grad_crush_rephase
        pulse_amp[:, num_sample_pre:num_sample_pre + num_sample_pulse] = pulse
        t_total = num_sample_total * dt_us
        return grad_amp, pulse_amp, t_total, torch.sum(
            grad_amp[num_sample_pre:num_sample_pre + num_sample_pulse]) * dt_us

    @classmethod
    def prep_single_grad_pulse(cls, params: options.SimulationParameters = options.SimulationParameters(),
                               excitation_flag: bool = True,
                               grad_rephase_factor: float = 1.0):
        # -- prep pulse
        pulse_type: str = 'Excitation'  # just set it
        log_module.debug(f'prep pulse {pulse_type}; # {0}')
        grad_pulse = cls(pulse_type=pulse_type, pulse_number=0)
        # read file
        # get duration & grad pulse defaults
        gp_details = GPDetails.get_gp_details(params=params, excitation_flag=excitation_flag)
        gp_details.grad_crush_rephase *= grad_rephase_factor
        if grad_rephase_factor < 1e-3:
            gp_details.duration_crush_rephase = 0.0
        # read rf
        rf = pypsi.Params.pulse.load(gp_details.pypsi_path)
        duration_pulse = params.sequence.duration_excitation

        if abs(rf.duration_in_us - duration_pulse) > 1e-5:
            # resample pulse
            rf.resample_to_duration(duration_in_us=int(duration_pulse))

        pulse = torch.from_numpy(rf.amplitude) * torch.exp(torch.from_numpy(1j * rf.phase))
        # calculate and normalize
        pulse_from_pypsi = functions.pulseCalibrationIntegral(
            pulse=pulse,
            delta_t=rf.get_dt_sampling_in_us(),
            pulse_number=0,
            sim_params=params,
            excitation=True)

        # build verse pulse gradient
        grad, pulse, duration, area_grad = cls.build_pulse_grad_shapes(
            gp_details=gp_details,
            pulse=pulse_from_pypsi,
            duration_pre=0.0,
            grad_amp_pre=0.0
        )

        # assign vars
        grad_pulse.num_sampling_points = rf.num_samples
        grad_pulse.dt_sampling_steps = rf.get_dt_sampling_in_us()
        grad_pulse.data_grad = grad
        grad_pulse.data_pulse_x = pulse.real
        grad_pulse.data_pulse_y = pulse.imag
        grad_pulse.duration = rf.duration_in_us

        grad_pulse._set_float32()
        return grad_pulse

    @classmethod
    def prep_acquisition(cls, params: options.SimulationParameters = options.SimulationParameters()):
        log_module.debug("prep acquisition")
        dt_sampling_steps = params.sequence.duration_acquisition / params.settings.acquisition_number

        grad_pulse = cls(pulse_type='Acquisition', pulse_number=0, num_sampling_points=1,
                         dt_sampling_steps=dt_sampling_steps)
        # assign data
        grad_pulse.data_grad = torch.linspace(
            params.sequence.gradient_acquisition,
            params.sequence.gradient_acquisition,
            1)
        # need to cast to pulse data dim [b1s, num_steps]
        grad_pulse.data_pulse_x = torch.linspace(0, 0, 1)[None, :]
        grad_pulse.data_pulse_y = torch.linspace(0, 0, 1)[None, :]
        grad_pulse.duration = params.sequence.duration_acquisition
        grad_pulse._set_float32()
        return grad_pulse

    def plot(self, sim_data: options.SimulationData):
        plotting.plot_grad_pulse(
            px=self.data_pulse_x, py=self.data_pulse_y,
            g=self.data_grad, b1_vals=sim_data.b1_vals,
            name=f"{self.pulse_type}-{self.pulse_number}"
        )


class Timing:
    def __init__(self, time_pre_pulse: torch.tensor = torch.zeros(1),
                 time_post_pulse: torch.tensor = torch.zeros(1)):
        self.time_post_pulse = time_post_pulse
        self.time_pre_pulse = time_pre_pulse

    @classmethod
    def build_fill_timing_mese(cls, params: options.SimulationParameters = options.SimulationParameters()):
        """
        Create a timing scheme: save time in [us] in array[2] -> [0] before pulse, [1] after pulse.
        For all refocusing pulses, i.e. ETL times
        Highly Sequence scheme dependent!
        :return: timing array
        """
        # all in [us]
        time_pre_pulse = torch.zeros(params.sequence.etl)
        time_post_pulse = torch.zeros(params.sequence.etl)

        # after excitation - before first refocusing:
        time_pre_pulse[0] = 1000 * params.sequence.esp / 2 - (
                params.sequence.duration_excitation / 2 + params.sequence.duration_excitation_rephase
                + params.sequence.duration_refocus / 2
        )
        # refocusing pulse...
        # after first refocusing
        time_post_pulse[0] = 1000 * params.sequence.esp / 2 - (
                params.sequence.duration_refocus / 2 + params.sequence.duration_crush +
                params.sequence.duration_acquisition / 2
        )

        # in this scheme, equal for all pulses, should incorporate some kind of "menu" for different sequence flavors:
        for pulseIdx in torch.arange(1, params.sequence.etl):
            time_pre_pulse[pulseIdx] = time_post_pulse[0]
            time_post_pulse[pulseIdx] = time_post_pulse[0]
        return cls(time_pre_pulse=time_pre_pulse, time_post_pulse=time_post_pulse)

    @classmethod
    def build_fill_timing_se(cls, params: options.SimulationParameters = options.SimulationParameters()):
        """
        Create a timing scheme: save time in [us] in array[2] -> [0] before pulse, [1] after pulse.
        For SE sequence
        :return: timing array
        """
        # all in [us]
        time_pre_pulse = torch.zeros(1)
        time_post_pulse = torch.zeros(1)
        # after excitation - before refocusing (check for prephaser):
        time_pre_pulse[0] = 1000 * params.sequence.esp / 2 - (
                params.sequence.duration_excitation / 2 + params.sequence.duration_excitation_rephase
                + params.sequence.duration_refocus / 2
        )
        # refocusing pulse...
        # after refocusing
        time_post_pulse[0] = 1000 * params.sequence.esp / 2 - (
                params.sequence.duration_refocus / 2 + params.sequence.duration_crush +
                params.sequence.duration_acquisition / 2
        )
        return cls(time_pre_pulse=time_pre_pulse, time_post_pulse=time_post_pulse)

    def set_device(self, device: torch.device):
        self.time_post_pulse = self.time_post_pulse.to(device)
        self.time_pre_pulse = self.time_pre_pulse.to(device)


def prep_gradient_pulse_mese(
        sim_params: options.SimulationParameters) -> (GradPulse, list, Timing, GradPulse):
    log_module.debug('pulse preparation')
    gp_excitation = GradPulse.prep_grad_pulse(
        pulse_type='Excitation',
        pulse_number=0,
        sym_spoil=False,
        params=sim_params,
        orig_mese=False
    )

    gp_refocus_1 = GradPulse.prep_grad_pulse(
        pulse_type='Refocusing_1',
        pulse_number=1,
        sym_spoil=False,
        params=sim_params,
        orig_mese=False
    )
    # built list of grad_pulse events, acquisition and timing
    grad_pulses = [gp_refocus_1]
    for r_idx in torch.arange(2, sim_params.sequence.etl + 1):
        gp_refocus = GradPulse.prep_grad_pulse(
            pulse_type='Refocusing',
            pulse_number=r_idx,
            sym_spoil=True,
            params=sim_params,
            orig_mese=False
        )
        grad_pulses.append(gp_refocus)

    acquisition = GradPulse.prep_acquisition(params=sim_params)

    log_module.debug(f"calculate timing")
    timing = Timing.build_fill_timing_mese(sim_params)

    return gp_excitation, grad_pulses, timing, acquisition
