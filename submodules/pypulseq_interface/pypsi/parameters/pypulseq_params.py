import simple_parsing as sp
import dataclasses as dc
import typing
import numpy as np
import logging

log_module = logging.getLogger(__name__)


@dc.dataclass
class PypulseqParameters(sp.helpers.Serializable):
    """
    Holding all Sequence Parameters
    """
    name: str = "jstmc"
    version: str = "xx"
    report: bool = sp.field(default=False, alias="-r")
    visualize: bool = sp.field(default=True, alias="-v")

    resolution_fov_read: float = 100  # [mm]
    resolution_fov_phase: float = 100.0  # [%]
    resolution_base: int = 100
    resolution_slice_thickness: float = 1.0  # [mm]
    resolution_slice_num: int = 10
    resolution_slice_gap: int = 20  # %

    number_central_lines: int = 40
    acceleration_factor: float = 2.0

    excitation_rf_fa: float = 90.0
    excitation_rf_phase: float = 90.0  # °
    excitation_rf_time_bw_prod: float = 2.0
    excitation_duration: int = 2500  # [us]
    excitation_grad_moment_pre: float = 1000.0  # Hz/m
    excitation_grad_rephase_factor: float = 1.04  # Correction factor for insufficient rephasing

    refocusing_rf_fa: typing.List = dc.field(default_factory=lambda: [140.0])
    refocusing_rf_phase: typing.List = dc.field(default_factory=lambda: [0.0])  # °
    refocusing_rf_time_bw_prod: float = 2.0
    refocusing_duration: int = 3000  # [us]
    refocusing_grad_slice_scale: float = 1.5  # adjust slice selective gradient sice of refocusing -
    # caution: this broadens the slice profile of the pulse, the further away from 180 fa
    # we possibly get saturation outside the slice
    read_grad_spoiling_factor: float = 0.5
    grad_moment_slice_spoiling: float = 2500.0  # [Hz/m]
    grad_moment_slice_spoiling_end: float = 2500  # [Hz/m]
    interleaved_acquisition: bool = True
    # interfacing with rfpf
    ext_rf_exc: str = ""
    ext_rf_ref: str = ""

    esp: float = 7.6  # [ms] echo spacing
    etl: int = 8  # echo train length
    tr: float = 4500.0  # [ms]

    bandwidth: float = 250.0  # [Hz / px]
    oversampling: int = 2  # oversampling factor
    sample_weighting: float = 0.0  # factor to weight random sampling towards central k-space ->
    # towards 1 we get densely sampled center

    acq_phase_dir: str = "PA"

    def __post_init__(self):
        # resolution, number of fe and pe. we want this to be a multiple of 2 for FFT reasoning (have 0 line)
        self.resolution_n_read = int(np.ceil(self.resolution_base / 2) * 2)  # number of freq encodes
        # if we need to up one freq encode point, we need to update the fov to keep desired voxel resolution
        if np.abs(self.resolution_n_read - self.resolution_base) > 0:
            log_module.info(
                f"updating FOV in read direction from {self.resolution_fov_read:.3f} mm to "
                f"{self.resolution_n_read / self.resolution_base * self.resolution_fov_read:.3f} mm. "
                f"For even frequency encode number")
            self.resolution_fov_read *= self.resolution_n_read / self.resolution_base
            self.resolution_base = self.resolution_n_read
        # calculate number of phase encodes.
        # hence we might update the user defined fov phase percentage to the next higher position
        # phase grads - even number of lines. should end up with a 0 line
        resolution_n_phase = self.resolution_base * self.resolution_fov_phase / 100
        self.resolution_n_phase = int(np.ceil(resolution_n_phase / 2) * 2)
        if np.abs(self.resolution_n_phase - int(resolution_n_phase)) > 0:
            log_module.info(
                f"updating FOV in phase direction from {self.resolution_fov_phase:.2f} % to "
                f"{self.resolution_n_phase / self.resolution_n_read * 100:.2f} % . "
                f"For even phase encode line number")
            self.resolution_fov_phase = self.resolution_n_phase / self.resolution_n_read * 100
        self.resolution_voxel_size_read = self.resolution_fov_read / self.resolution_base  # [mm]
        self.resolution_voxel_size_phase = self.resolution_fov_read / self.resolution_base  # [mm]
        self.delta_k_read = 1e3 / self.resolution_fov_read  # cast to m
        self.delta_k_phase = 1e3 / (self.resolution_fov_read * self.resolution_fov_phase / 100.0)  # cast to m
        self.te = np.arange(1, self.etl + 1) * self.esp  # [ms] echo times
        # there is one gap less than number of slices,
        self.z_extend = self.resolution_slice_thickness * (
                self.resolution_slice_num + self.resolution_slice_gap / 100.0 * (self.resolution_slice_num - 1)
        )  # in mm
        # acc
        self.number_outer_lines = round(
            (self.resolution_n_phase - self.number_central_lines) / self.acceleration_factor)
        # sequence
        self.acquisition_time = 1 / self.bandwidth
        # dwell needs to be on adc raster time, acquisition time is flexible -> leads to small deviations in bandwidth
        # adc raster here hardcoded
        adc_raster = 1e-7
        s_dwell = self.acquisition_time / self.resolution_n_read / self.oversampling  # oversampling
        adcr_dwell = s_dwell / adc_raster  # we want this to be divisible by 2, take next lower even number
        adcr_dwell = int(np.floor(adcr_dwell / 2) * 2)
        # adcr_dwell = int(np.floor(s_dwell / adc_raster))     # round down -> slight changes needed to set on raster,
        # might as well decrease acquisition time with change
        self.dwell = adc_raster * adcr_dwell
        if np.abs(s_dwell - self.dwell) > 1e-9:
            log_module.info(f"setting dwell time on adc raster -> small bw adoptions (set bw: {self.bandwidth:.3f})")
        # update acquisition time and bandwidth
        self.acquisition_time = self.dwell * self.resolution_n_read * self.oversampling
        self.bandwidth = 1 / self.acquisition_time
        log_module.debug(f"Bandwidth: {self.bandwidth:.3f} Hz/px; "
                         f"Readout time: {self.acquisition_time * 1e3:.1f} ms; "
                         f"DwellTime: {self.dwell * 1e6:.1f} us; "
                         f"Number of Freq Encodes: {self.resolution_n_read}")
        # ref list
        if self.refocusing_rf_fa.__len__() != self.refocusing_rf_phase.__len__():
            err = f"provide same amount of refocusing pulse angle ({self.refocusing_rf_fa.__len__()}) " \
                  f"and phases ({self.refocusing_rf_phase.__len__()})"
            log_module.error(err)
            raise AttributeError(err)
        # check for phase values
        for l_idx in range(self.refocusing_rf_phase.__len__()):
            while np.abs(self.refocusing_rf_phase[l_idx]) > 180.0:
                self.refocusing_rf_phase[l_idx] = self.refocusing_rf_phase[l_idx] - \
                                                  np.sign(self.refocusing_rf_phase[l_idx]) * 180.0
            while np.abs(self.refocusing_rf_fa[l_idx]) > 180.0:
                self.refocusing_rf_fa[l_idx] = self.refocusing_rf_fa[l_idx] - np.sign(
                    self.refocusing_rf_fa[l_idx]) * 180.0
        while self.refocusing_rf_fa.__len__() < self.etl:
            # fill up list with last value
            self.refocusing_rf_fa.append(self.refocusing_rf_fa[-1])
            self.refocusing_rf_phase.append(self.refocusing_rf_phase[-1])
        # while self.sliceSpoilingMoment.__len__() < self.ETL:
        #     self.sliceSpoilingMoment.append(self.sliceSpoilingMoment[-1])

        # casting
        self.excitation_rf_rad_fa = self.excitation_rf_fa / 180.0 * np.pi
        self.excitation_rf_rad_phase = self.excitation_rf_phase / 180.0 * np.pi
        self.refocusing_rf_rad_fa = np.array(self.refocusing_rf_fa) / 180.0 * np.pi
        self.refocusing_rf_rad_phase = np.array(self.refocusing_rf_phase) / 180.0 * np.pi
        self.get_voxel_size()
        if self.acq_phase_dir == "PA":
            self.read_dir = 'x'
            self.phase_dir = 'y'
        elif self.acq_phase_dir == "RL":
            self.phase_dir = 'x'
            self.read_dir = 'y'
        else:
            err = 'Unknown Phase direction: chose either PA or RL'
            log_module.error(err)
            raise AttributeError(err)

        # error catches
        if np.any(np.array(self.grad_moment_slice_spoiling) < 1e-7):
            err = f"this implementation needs a spoiling moment supplied: provide spoiling Moment > 0"
            log_module.error(err)
            raise ValueError(err)

    def get_voxel_size(self, write_log: bool = False):
        msg = (
            f"Voxel Size [read, phase, slice] in mm: "
            f"{[self.resolution_voxel_size_read, self.resolution_voxel_size_phase, self.resolution_slice_thickness]}"
        )
        if write_log:
            log_module.info(msg)
        else:
            log_module.debug(msg)
        return self.resolution_voxel_size_read, self.resolution_voxel_size_phase, self.resolution_slice_thickness

    def get_fov(self):
        fov_read = 1e-3 * self.resolution_fov_read
        fov_phase = 1e-3 * self.resolution_fov_read * self.resolution_fov_phase / 100
        fov_slice = self.z_extend * 1e-3
        if self.read_dir == 'x':
            log_module.info(
                f"FOV (xyz) Size [read, phase, slice] in mm: "
                f"[{1e3 * fov_read:.1f}, {1e3 * fov_phase:.1f}, {1e3 * fov_slice:.1f}]")
            return fov_read, fov_phase, fov_slice
        else:
            log_module.info(
                f"FOV (xyz) Size [phase, read, slice] in mm: "
                f"[{1e3 * fov_phase:.1f}, {1e3 * fov_read:.1f}, {1e3 * fov_slice:.1f}]")
            return fov_phase, fov_read, fov_slice
