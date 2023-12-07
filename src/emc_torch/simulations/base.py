import abc

import pandas as pd
import numpy as np
from emc_torch import options, blocks, plotting
import logging
import torch
import pathlib as plib
log_module = logging.getLogger(__name__)


class Simulation(abc.ABC):
    def __init__(self, sim_params: options.SimulationParameters):
        log_module.info("__ Set-up Simulation __ ")
        # setup device
        if sim_params.config.use_gpu and torch.cuda.is_available():
            device = torch.device(f"cuda:{sim_params.config.gpu_device}")
        else:
            device = torch.device("cpu")
        self.device: torch.device = device
        log_module.info(f"\t\t- torch device: {device}")
        # simulation variables
        self.params: options.SimulationParameters = sim_params
        self.data: options.SimulationData = options.SimulationData.from_sim_parameters(
            sim_params=self.params, device=self.device
        )
        self.fig_path: plib.Path = NotImplemented
        self._fig_magnetization_profile_snaps: list = []
        # pick middle sim range values for magnetization profile snaps
        self._fig_t1_idx: int = int(self.data.t1_vals.shape[0] / 2)  # t1
        self._fig_t2_idx: int = int(self.data.t2_vals.shape[0] / 2)  # t2
        self._fig_b1_idx: int = int(self.data.b1_vals.shape[0] / 2)  # b1

        # setup plotting
        if self.params.config.visualize:
            # set up plotting path
            out_path = plib.Path(self.params.config.save_path).absolute().joinpath("plots/")
            out_path.mkdir(parents=True, exist_ok=True)
            self.fig_path = out_path

            # set up indices for which we save the snaps
            # (otherwise we need to transfer the whole t1s * t2s * b1s * sample_num * 4 array
            # every time we save the profile)
            # choose first t1

            # save initial magnetization to snaps
            self.set_magnetization_profile_snap(snap_name="initial")

        # setup acquisition
        self.gp_acquisition = blocks.GradPulse.prep_acquisition(params=self.params)
        # give instance of sequence timings
        self.sequence_timings: blocks.SequenceTimings = blocks.SequenceTimings()
        # sim info
        # set total number of simulated curves
        num_curves = self.data.t2_vals.shape[0] * self.data.b1_vals.shape[0] * self.data.t1_vals.shape[0]
        log_module.info(f"\t\t- total number of entries to simulate: {num_curves}")
        sim_params.settings.total_num_sim = num_curves
        # call specific prep
        self._prep()
        # call timing build
        self._register_sequence_timings()

    @abc.abstractmethod
    def _prep(self):
        """ sequence specific preparation method (gradient pulses and init variants) """

    @abc.abstractmethod
    def _simulate(self):
        """ sequence specific simulation method"""

    @abc.abstractmethod
    def _set_device(self):
        """ sequence specific setting to put relevant tensors on device """

    @abc.abstractmethod
    def _register_sequence_timings(self):
        """ sequence specific timing objects """

    def set_magnetization_profile_snap(self, snap_name: str):
        """ add magnetization profile snapshot to list for plotting """
        t1_choice_idx = min([self.data.magnetization_propagation.shape[0] - 1, self._fig_t1_idx])
        t2_choice_idx = min([self.data.magnetization_propagation.shape[1] - 1, self._fig_t2_idx])
        b1_choice_idx = min([self.data.magnetization_propagation.shape[2] - 1, self._fig_b1_idx])
        mag_profile = self.data.magnetization_propagation[
            t1_choice_idx, t2_choice_idx, b1_choice_idx
        ].clone().detach().cpu()
        self._fig_magnetization_profile_snaps.append({
            "name": snap_name,
            "profile": mag_profile
        })

    def simulate(self):
        """ sequence specific definition of the simulation """
        # set device
        self._set_device()
        # simulate
        self._simulate()

    def plot_magnetization_profiles(self, animate: bool = True):
        # pick middle sim range values
        b1_val = f"{self.data.b1_vals[self._fig_b1_idx].numpy(force=True):.2f}".replace(".", "p")
        t2_val = f"{1000*self.data.t2_vals[self._fig_t2_idx].numpy(force=True):.1f}ms".replace(".", "p")
        t1_val = f"{self.data.t1_vals[self._fig_t1_idx].numpy(force=True):.2f}s".replace(".", "p")

        profiles = []
        dims = []
        names = []
        sample_pts = []
        last_name = ""
        for entry_idx in range(len(self._fig_magnetization_profile_snaps)):
            entry_dict = self._fig_magnetization_profile_snaps[entry_idx]
            name = entry_dict["name"]
            # loop to iterating characters, see if we are on same refocussing
            for character in name:
                # checking if character is numeric,
                # saving index
                if character.isdigit():
                    temp = name.index(character)
                    name = name[:temp+1]
                    break
            if name == last_name:
                dim_extend = "_post_acquisition"
            else:
                dim_extend = ""
            # on inital magnetization no different values are available
            mag_prof = entry_dict["profile"].numpy(force=True)
            profiles.extend(np.abs(mag_prof[:, 0] + 1j * mag_prof[:, 1]))
            dims.extend([f"abs{dim_extend}"] * mag_prof.shape[0])
            profiles.extend(np.angle(mag_prof[:, 0] + 1j * mag_prof[:, 1]) / np.pi)
            dims.extend([f"angle{dim_extend}"] * mag_prof.shape[0])
            profiles.extend(mag_prof[:, 2])
            dims.extend([f"z{dim_extend}"] * mag_prof.shape[0])

            names.extend([name] * 3 * mag_prof.shape[0])
            sample_pts.extend(np.tile(self.data.sample_axis.numpy(force=True) * 1e3, 3))
            last_name = name
        df = pd.DataFrame({
            "profile": profiles, "dim": dims, "axis": sample_pts, "name": names
        })
        # calculate desired slice thickness from pulse & slice select
        bw = self.params.pulse.excitation.bandwidth_in_Hz      # Hz
        grad = self.params.sequence.gradient_excitation     # mT/m
        desired_slice_thickness_mm = np.abs(
            bw / self.params.sequence.gamma_hz / grad / 1e-6
        )
        plotting.plot_magnetization(
            mag_profile_df=df, animate=animate, name=f"t1-{t1_val}_t2-{t2_val}_b1-{b1_val}",
            out_path=self.fig_path, slice_thickness_mm=desired_slice_thickness_mm
        )

    def plot_emc_signal(self):
        plotting.plot_emc_sim_data(sim_data=self.data, out_path=self.fig_path)

    def plot_signal_traces(self):
        plotting.plot_signal_traces(sim_data=self.data, out_path=self.fig_path)