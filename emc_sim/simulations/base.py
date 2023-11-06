import abc
from emc_sim import options, plotting
import logging
import torch
log_module = logging.getLogger(__name__)


class Simulation(abc.ABC):
    def __init__(self, sim_params: options.SimulationParameters,
                 device: torch.device = torch.device("cpu"), num_mag_evol_plot: int = 10):
        # setup device
        self.device: torch.device = device
        log_module.debug(f"torch device: {device}")
        # simulation variables
        self.params: options.SimulationParameters = sim_params
        self.data: options.SimulationData = options.SimulationData.from_sim_parameters(
            sim_params=self.params, device=self.device
        )

        # setup plotting
        self.plot_idx: int = 0
        if self.params.config.visualize:
            # set up running plot and plot initial magnetization
            self.fig = plotting.prep_plot_running_mag(
                num_mag_evol_plot, 1,
                t2=self.data.t2_vals[0], b1=self.data.b1_vals[0]
            )
            self.fig = plotting.plot_running_mag(self.fig, self.data, id=self.plot_idx)
            self.plot_idx += 1

        # call specific prep
        self._prep()

    @abc.abstractmethod
    def _prep(self):
        """ sequence specific preparation method (gradient pulses and init variants) """
    @abc.abstractmethod
    def simulate(self):
        """ sequence specific definition of the simulation """
        pass


