import itertools

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
        self.weight_matrix: torch.tensor = NotImplementedError

    def set_weight_matrix(self, sim_data: options.SimulationData):
        # t1 gets avd out. t2 b1 get combined:
        param_list = [(t2, b1) for t2 in sim_data.t2_vals for b1 in sim_data.b1_vals]
        num_param_pairs = param_list.__len__()
        weight_matrix = torch.tril(torch.ones((num_param_pairs, num_param_pairs)), -1)
        # set t2 inter dependencies
        for t2 in sim_data.t2_vals:
            t2_idxs = []
            for p_idx in range(num_param_pairs):
                t2p, _ = param_list[p_idx]
                if t2p == t2:
                    t2_idxs.append(p_idx)

            for o1_idx, o2_idx in itertools.product(t2_idxs, t2_idxs):
                weight_matrix[o1_idx,o2_idx] = 0.0
        self.weight_matrix = weight_matrix.to(sim_data.device)

    def calculate_loss(self, sim_data: options.SimulationData):
        # want to minimize cross correlation across b1 and t2 values
        emc_mag = sim_data.emc_signal_mag
        emc_phase = sim_data.emc_signal_phase
        # digress T1 out    dims: [t1, t2, b1, echos]
        while emc_mag.shape.__len__() > 3:
            emc_mag = torch.mean(emc_mag, dim=0)
        while emc_phase.shape.__len__() > 3:
            emc_phase = torch.mean(emc_phase, dim=0)
        # flatten matrix
        emc_mag = torch.reshape(emc_mag, (-1, emc_mag.shape[-1]))
        emc_phase = torch.reshape(emc_phase, (-1, emc_phase.shape[-1]))
        # normalize
        norm = torch.norm(emc_mag, dim=-1, keepdim=True)
        emc_mag = torch.div(emc_mag, norm)
        # dims [t2b1, echoes]
        mag_source = emc_mag[:, None, :] - emc_mag
        loss_matrix_mag = torch.einsum("ijk, ijk -> ij", mag_source, mag_source)
        phase_source = emc_phase[:, None, :] - emc_phase
        loss_matrix_phase = torch.einsum("ijk, ijk -> ij", phase_source, phase_source)
        loss = self.weight_matrix * loss_matrix_mag + 0.2 * self.weight_matrix * loss_matrix_phase
        self.value = torch.sum(loss) / emc_mag.shape[0]


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
