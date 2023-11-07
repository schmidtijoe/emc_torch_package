import abc

import pandas as pd
import numpy as np
from emc_sim import options, blocks, plotting
import logging
import torch
import pathlib as plib
log_module = logging.getLogger(__name__)


class Simulation(abc.ABC):
    def __init__(self, sim_params: options.SimulationParameters,
                 device: torch.device = torch.device("cpu"), num_mag_evol_plot: int = 10):
        log_module.info("__ Set-up Simulation __ ")
        # setup device
        self.device: torch.device = device
        log_module.debug(f"torch device: {device}")
        # simulation variables
        self.params: options.SimulationParameters = sim_params
        self.data: options.SimulationData = options.SimulationData.from_sim_parameters(
            sim_params=self.params, device=self.device
        )
        self.fig_path: plib.Path = NotImplemented
        self._fig_magnetization_profile_snaps: list = []

        # setup plotting
        if self.params.config.visualize:
            # set up plotting path
            out_path = plib.Path(self.params.config.save_path).absolute().joinpath("plots/")
            out_path.mkdir(parents=True, exist_ok=True)
            self.fig_path = out_path

            # save initial magnetization to snaps
            self.set_magnetization_profile_snap(
                magnetization_profile=self.data.magnetization_propagation, snap_name="initial"
            )

        # setup acquisition
        self.gp_se_acquisition = blocks.GradPulse.prep_acquisition(params=self.params)
        # call specific prep
        self._prep()

    @abc.abstractmethod
    def _prep(self):
        """ sequence specific preparation method (gradient pulses and init variants) """
    @abc.abstractmethod
    def simulate(self):
        """ sequence specific definition of the simulation """
        pass

    def set_magnetization_profile_snap(self, magnetization_profile: torch.tensor, snap_name: str):
        """ add magnetization profile snapshot to list for plotting """
        self._fig_magnetization_profile_snaps.append(
            {"name": snap_name, "profile": magnetization_profile}
        )

    def plot_magnetization_profiles(self, animate: bool = True):
        # pick middle sim range values
        b1_idx = int(self.data.b1_vals.shape[0] / 2)
        b1_val = f"{self.data.b1_vals[b1_idx].numpy(force=True):.2f}".replace(".", "p")
        t2_idx = int(self.data.t2_vals.shape[0] / 2)
        t2_val = f"{1000*self.data.t2_vals[t2_idx].numpy(force=True):.1f}ms".replace(".", "p")
        t1_idx = int(self.data.t1_vals.shape[0] / 2)
        t1_val = f"{self.data.t1_vals[t1_idx].numpy(force=True):.2f}s".replace(".", "p")

        profiles = []
        dims = []
        names = []
        axs = []
        for entry_idx in range(len(self._fig_magnetization_profile_snaps)):
            entry_dict = self._fig_magnetization_profile_snaps[entry_idx]
            # on inital magnetization no different values are available
            mag_prof = entry_dict["profile"].numpy(force=True)
            t1_choice_idx = np.min([mag_prof.shape[0] - 1, t1_idx])
            t2_choice_idx = np.min([mag_prof.shape[1] - 1, t2_idx])
            b1_choice_idx = np.min([mag_prof.shape[2] - 1, b1_idx])
            mag_prof = mag_prof[t1_choice_idx, t2_choice_idx, b1_choice_idx]
            profiles.extend(np.abs(mag_prof[:, 0] + 1j * mag_prof[:, 1]))
            dims.extend(["abs"] * mag_prof.shape[0])
            axs.extend(self.data.sample_axis.numpy(force=True))
            profiles.extend(np.angle(mag_prof[:, 0] + 1j * mag_prof[:, 1]) / np.pi)
            dims.extend(["angle"] * mag_prof.shape[0])
            axs.extend(self.data.sample_axis)
            profiles.extend(mag_prof[:, 2])
            dims.extend(["z"] * mag_prof.shape[0])
            axs.extend(self.data.sample_axis)

            names.extend([entry_dict["name"]] * 3 * mag_prof.shape[0])

        df = pd.DataFrame({
            "mag_profile": profiles, "dim": dims, "sample_axis": axs, "name": names
        })
        # calculate desired slice thickness from pulse & slice select
        bw = self.params.pulse.bandwidth_in_Hz      # Hz
        grad = self.params.sequence.gradient_excitation     # mT/m
        desired_slice_thickness_mm = np.abs(
            2 * bw / self.params.sequence.gamma_hz / grad / 1e-6
        )
        plotting.plot_magnetization(mag_profile_df=df, animate=animate, name=f"t1-{t1_val}_t2-{t2_val}_b1-{b1_val}",
                                    out_path=self.fig_path, slice_thickness_mm=desired_slice_thickness_mm)

