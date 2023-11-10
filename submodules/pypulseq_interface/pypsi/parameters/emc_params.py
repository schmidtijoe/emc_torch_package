import simple_parsing as sp
import dataclasses as dc
import typing
import logging
import numpy as np
log_module = logging.getLogger(__name__)


@dc.dataclass
class EmcParameters(sp.helpers.Serializable):
    """
    Holding all Parameters necessary to simulate via EMC
    """
    # global parameter gamma [Hz/t]
    gamma_hz: float = 42577478.518

    # echo train length
    etl: int = 16
    # echo spacing [ms]
    esp: float = 9.0
    # bandwidth [Hz/px]
    bw: float = 349
    # gradient mode

    # Excitation, Flip Angle [°]
    excitation_angle: float = 90.0
    # Excitation, Phase [°]
    excitation_phase: float = 90.0
    # Excitation, gradient if rectangular/trapezoid [mt/m]
    gradient_excitation: float = -18.5
    # Excitation, duration of pulse [us]
    duration_excitation: float = 2560.0

    gradient_excitation_rephase: float = -10.51  # [mT/m], rephase
    duration_excitation_rephase: float = 1080.0  # [us], rephase

    # Refocussing, Flip Angle [°]
    refocus_angle: typing.List = sp.field(default_factory=lambda: [140.0])
    # Refocussing, Phase [°]
    refocus_phase: typing.List = sp.field(default_factory=lambda: [0.0])
    # Refocussing, gradient strength if rectangular/trapezoid [mt/m]
    gradient_refocus: float = -36.2
    # Refocussing, duration of pulse [us]
    duration_refocus: float = 3584.0

    gradient_crush: float = -38.7  # [mT/m], crusher
    duration_crush: float = 1000.0  # [us], crushe
    gradient_acquisition: float = 0.0  # set automatically after settings init

    # time for acquisition (of one pixel) * 1e6 <- [(px)s] * 1e6

    def __post_init__(self):
        self.gamma_pi: float = self.gamma_hz * 2 * np.pi
        self.duration_acquisition: float = 1e6 / self.bw  # [us]
        if self.refocus_phase.__len__() != self.refocus_angle.__len__():
            err = f"provide same amount of refocusing pulse angle ({self.refocus_angle.__len__()}) " \
                  f"and phases ({self.refocus_phase.__len__()})"
            log_module.error(err)
            raise AttributeError(err)
        # check for phase values
        for l_idx in range(self.refocus_phase.__len__()):
            while abs(self.refocus_phase[l_idx]) > 180.0:
                self.refocus_phase[l_idx] = self.refocus_phase[l_idx] - np.sign(self.refocus_phase[l_idx]) * 180.0
            while abs(self.refocus_angle[l_idx]) > 180.0:
                self.refocus_angle[l_idx] = self.refocus_angle[l_idx] - np.sign(self.refocus_angle[l_idx]) * 180.0
        while self.refocus_angle.__len__() < self.etl:
            # fill up list with last value
            self.refocus_angle.append(self.refocus_angle[-1])
            self.refocus_phase.append(self.refocus_phase[-1])
