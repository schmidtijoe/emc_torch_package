import simple_parsing as sp
import dataclasses as dc


@dc.dataclass
class ScannerParameters(sp.helpers.Serializable):
    """
    Holding all Scanning System Parameters
    """
    # magnet
    b_0: float = 6.98    # [T]
    # gradients
    max_grad: float = 40.0
    grad_unit: str = 'mT/m'
    max_slew: float = 200.0
    slew_unit: str = 'T/m/s'
    rise_time: int = 0  # watch out, rise time != 0 gives max_slew = 0 in opts method
    grad_raster_time: float = 10e-6

    # rf
    rf_dead_time: float = 100e-6
    rf_raster_time: float = 1e-6
    rf_ringdown_time: float = 30e-6

    # general
    adc_dead_time: float = 20e-6
    gamma: float = 42577478.518  # [Hz/T]
