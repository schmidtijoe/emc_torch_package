import torch
from emc_sim import options


class LossBC:
    def __init__(self, name: str = "", emphasis: float = 1.0):
        self.name: str = name
        self.value: torch.tensor = torch.zeros(1)
        self.emphasis: float = emphasis

    def get_loss(self):
        return self.emphasis * self.value

    def calculate_loss(self, sim_data: options.SimulationData):
        return NotImplementedError


class LossSNR(LossBC):
    def __init__(self, emphasis: float = 1.0):
        super().__init__(name="LossSNR", emphasis=emphasis)

    def calculate_loss(self, sim_data: options.SimulationData):
        # want to maximize curve snr
        total_signal = torch.norm(sim_data.emc_signal_mag, dim=-1)
        # optimizer minimizes the loss, hence take engative
        self.value = - torch.sum(total_signal)


class LossCorr(LossBC):
    def __init__(self, emphasis: float = 1.0):
        super().__init__(name="LossCorr", emphasis=emphasis)

    def calculate_loss(self, sim_data: options.SimulationData):
        self.value = -1


class LossOptimize:
    def __init__(self, lambda_snr: float = 1.0, lambda_corr: float = 1.0):
        self.loss_corr = LossCorr(emphasis=lambda_corr)
        self.loss_snr = LossSNR(emphasis=lambda_snr)
        self.value: torch.tensor = torch.zeros(1)

    def calculate(self, sim_data: options.SimulationData):
        self.loss_corr.calculate_loss(sim_data=sim_data)
        self.loss_snr.calculate_loss(sim_data=sim_data)
        self.value = self.loss_corr.get_loss() + self.loss_snr.get_loss()

    def get_loss(self):
        return self.value

    def get_registered_loss_dict(self) -> dict:
        return {
            self.loss_corr.name: self.loss_corr.value,
            self.loss_snr.name: self.loss_snr.value
        }
