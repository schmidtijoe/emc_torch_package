import pathlib as plib
import nibabel as nib
import torch
import numpy as np
from torch import nn
import tqdm
import json
from emc_torch import DB
from emc_torch.fitting import options, io
import plotly.graph_objects as go
import plotly.subplots as psub
import logging
import scipy.interpolate as scinterp
import yabox as yb
from yabox.algorithms import DE as ybde

log_module = logging.getLogger(__name__)
logging.getLogger('simple_parsing').setLevel(logging.WARNING)


class DictionaryMatchingTv(nn.Module):
    """Custom Pytorch model for gradient optimization of our function
    """

    def __init__(
            self, slice_signal: torch.tensor,
            db_torch_mag: torch.tensor, db_t2s_s: torch.tensor, db_b1s: torch.tensor,
            delta_t_r2p_ms: torch.tensor, device: torch.device = torch.device("cpu"),
            t2_range_s: tuple = (0.0001, 1), r2p_range_Hz: tuple = (0.01, 200),
            b1_range: tuple = (0.2, 1.6), autograd: bool = False, activation: str = "HardSigmoid"):
        super().__init__()
        # save setup as normalized curves
        signal = torch.nan_to_num(
            torch.divide(slice_signal, torch.linalg.norm(slice_signal, dim=-1, keepdim=True)),
            nan=0.0, posinf=0.0
        )
        self.a_fn = self.set_activation(activation=activation)
        self.device = device
        log_module.info(f"set torch device: {device}")
        log_module.info(f"set autograd: {autograd}")
        # want to save some params
        self.batch_size = 5000
        # dimensions within slice
        self.nx: int = signal.shape[0]
        self.ny: int = signal.shape[1]
        # echo train length and corresponding timings for the attenuation factor as delta to next SE
        self.etl: int = db_torch_mag.shape[-1]
        self.delta_t_r2p_s: torch.tensor = delta_t_r2p_ms.to(self.device) * 1e-3

        self.slice_shape: tuple = (self.nx, self.ny)
        self.signal_mask = torch.zeros_like(signal[:, :, 0])
        # mask based on first echo
        self.signal_mask[signal[:, :, 0] > 1e-6] = 1
        # extend to all echoes
        self.signal_mask = self.signal_mask[:, :, None].expand(-1, -1, self.etl).to(dtype=torch.bool)
        # dims [x, y, t]
        # place weights dependent on signal mask - put factor of 1/1000 on signal free areas
        self.weights = torch.ones_like(self.signal_mask[:, :, 0], dtype=torch.float32, device=self.device)
        self.weights[~self.signal_mask[:, :, 0]] = 1e-3

        # reshape into dims [xy, t]
        self.signal = torch.reshape(signal, (-1, self.etl)).to(self.device)
        # database and t2 and b1 values
        self.db_t2s_s: torch.tensor = db_t2s_s
        self.db_t2s_s_unique: torch.tensor = torch.unique(db_t2s_s).to(self.device)
        self.num_t2s: int = self.db_t2s_s_unique.shape[0]
        self.db_b1s: torch.tensor = db_b1s
        self.db_b1s_unique: torch.tensor = torch.unique(db_b1s).to(self.device)
        self.db_delta_b1: torch.tensor = torch.diff(self.db_b1s_unique)[0]
        self.num_b1s: int = self.db_b1s_unique.shape[0]
        self.db_mag: torch.tensor = db_torch_mag.to(self.device)
        # range of parameters to consider
        self.t2_range_ms: tuple = t2_range_s
        self.r2p_range_Hz: tuple = r2p_range_Hz
        self.b1_range: tuple = b1_range
        # want to start with the unregularized match
        self.t2_estimate_init, self.b1_estimate_init = self.estimate_t2_b1_from_se()
        # crop to non noise values
        mask = torch.reshape(self.signal_mask, (-1, self.etl))
        self.signal = torch.reshape(self.signal[mask], (-1, self.etl))
        # want to optimize for T2s and B1 for a slice of the image
        self.db_t2s_s_unique = self.db_t2s_s_unique.to(self.device)
        self.db_b1s_unique = self.db_b1s_unique.to(self.device)
        self.delta_t_r2p_s = self.delta_t_r2p_s.to(self.device)
        # t2p_b1 = torch.distributions.Uniform(0, 1).sample((2, nx, ny)).to(self.device)
        # start with random t2p values
        r2p = torch.distributions.Uniform(0, 1).sample((self.nx, self.ny)).to(self.device)
        r2p = torch.flatten(r2p[self.signal_mask[:, :, 0]])
        # reverse effect of range
        b1 = ((self.b1_estimate_init.to(device) - self.b1_range[0]) /
              (self.b1_range[1] - self.b1_range[0]).to(self.device))
        t2 = ((self.t2_estimate_init.to(device) - self.t2_range_ms[0]) /
              (self.t2_range_ms[1] - self.t2_range_ms[0]).to(self.device))
        t2 = torch.flatten(t2[self.signal_mask[:, :, 0]])
        # register torch parameters - mask r2 and t2 and signal for computational efficiency
        self.estimates_b1: nn.Parameter = nn.Parameter(b1, requires_grad=autograd)
        self.estimates_r2p = nn.Parameter(r2p, requires_grad=autograd)
        self.estimates_t2 = nn.Parameter(t2, requires_grad=autograd)
        # some regularizers
        self.lambda_b1_tv = 1e-3
        self.lambda_t2_dist = 10.0
        self.lambda_t2_l2 = 100.0
        self.lambda_b1_dist = 1.0
        self.lambda_b1_l2 = 100.0
        self.lambda_t2_init = 1e3
        self.t2_init = torch.flatten(self.t2_estimate_init[self.signal_mask[:, :, 0]]).to(self.device)

    def set_activation(self, activation: str):
        if activation == "ReLU":
            a_fn = torch.nn.ReLU()
        elif activation == "Sigmoid":
            a_fn = torch.nn.Sigmoid()
        elif activation == "HardSigmoid":
            a_fn = torch.nn.Hardsigmoid()
        elif activation == "LeakyReLU":
            a_fn = torch.nn.LeakyReLU()
        else:
            err = f"unknown activation function{activation}"
            log_module.error(err)
            raise ValueError(err)
        return a_fn

    def estimate_t2_b1_from_se(self) -> (torch.tensor, torch.tensor):
        se_idx = self.delta_t_r2p_s < 1e-3
        db = self.db_mag[:, se_idx]
        db = torch.nan_to_num(
            torch.divide(db, torch.linalg.norm(db, dim=-1, keepdim=True)),
            posinf=0.0, nan=0.0
        )
        # need to batch data of whole slice
        batch_idx = torch.split(torch.arange(self.signal.shape[0]), self.batch_size)
        batch_data = torch.split(self.signal[:, se_idx], self.batch_size)
        t2_estimate_init = torch.zeros((self.nx * self.ny), dtype=self.db_t2s_s.dtype)
        b1_estimate_init = torch.zeros((self.nx * self.ny), dtype=self.db_b1s.dtype)
        for idx_batch in tqdm.trange(len(batch_idx), desc="match dictionary from SE reads:: unregularized brute force"):
            data_batch = batch_data[idx_batch]
            data_batch = torch.nan_to_num(
                torch.divide(data_batch, torch.linalg.norm(data_batch, dim=-1, keepdim=True)),
                posinf=0.0, nan=0.0
            )
            data_idx = batch_idx[idx_batch]
            # data dims [bs, t], db dims [t2-b1, t]
            l2_t2_b1 = torch.linalg.norm(data_batch[None, :] - db[:, None], dim=-1)
            fit_idx = torch.argmin(l2_t2_b1, dim=0).detach().cpu()
            t2_estimate_init[data_idx] = self.db_t2s_s[fit_idx]
            b1_estimate_init[data_idx] = self.db_b1s[fit_idx]
        t2_estimate_init = torch.reshape(t2_estimate_init, (self.nx, self.ny))
        b1_estimate_init = torch.reshape(b1_estimate_init, (self.nx, self.ny))
        del data_batch, db, l2_t2_b1, se_idx, batch_data
        torch.cuda.empty_cache()
        return t2_estimate_init, b1_estimate_init

    def get_candidate_values(self, identifier: str, activation: str = "Sigmoid"):
        if identifier == "t2":
            values = self.estimates_t2
        elif identifier == "r2p":
            values = self.estimates_r2p
        elif identifier == "b1_2d":
            values = self.estimates_b1
        elif identifier == "b1":
            values = torch.flatten(self.estimates_b1[self.signal_mask[:, :, 0]])
        else:
            err = f"identifier {identifier} unknown."
            log_module.error(err)
            raise ValueError(err)
        a_values = self.a_fn(values)
        return self.scale_to_range(a_values, identifier=identifier)

    def scale_to_range(self, values: torch.tensor, identifier: str):
        if identifier == "t2":
            v_range = self.t2_range_ms
        elif identifier == "r2p":
            v_range = self.r2p_range_Hz
        elif identifier == "b1" or identifier == "b1_2d":
            v_range = self.b1_range
        else:
            err = "identifier not recognized"
            log_module.error(err)
            raise ValueError(err)
        return v_range[0] + (v_range[1] - v_range[0]) * values

    def interpolate_db(self):
        # get flattened t2 and b1 values from current candidate
        t2_vals = self.scale_to_range(torch.flatten(self.estimates[0]), identifier="t2")
        b1_vals = self.scale_to_range(torch.flatten(self.estimates[2]), identifier="b1")
        # want to compute in between which two indexes of the db entries those lay, need indices and weights for this
        t2_idxs = torch.argmin((t2_vals[None, :] - self.db_t2s_s_unique[:, None]) ** 2, dim=0)
        t2_idxs_l = torch.clip(
            t2_idxs - (t2_vals - self.db_t2s_s_unique[t2_idxs] < 0).to(torch.int),
            0, self.db_t2s_s_unique.shape[0] - 1
        )
        t2_idxs_u = torch.clip(t2_idxs_l + 1, 0, self.db_t2s_s_unique.shape[0] - 1)
        t2_w = torch.clip(
            torch.nan_to_num(
                (t2_vals - self.db_t2s_s_unique[t2_idxs_l]) / (
                        self.db_t2s_s_unique[t2_idxs_u] - self.db_t2s_s_unique[t2_idxs_l]),
                nan=0.0, posinf=0.0, neginf=0.0
            ),
            0.0, 1.0
        )[:, None]
        # for b1s
        b1_idxs = torch.argmin((b1_vals[None, :] - self.db_b1s_unique[:, None]) ** 2, dim=0)
        b1_idxs_l = torch.clip(
            b1_idxs - (b1_vals - self.db_b1s_unique[b1_idxs] < 0).to(torch.int),
            0, self.db_b1s_unique.shape[0] - 1
        )
        b1_idxs_u = torch.clip(b1_idxs_l + 1, 0, self.db_b1s_unique.shape[0] - 1)
        b1_w = torch.clip(
            torch.nan_to_num(
                (b1_vals - self.db_b1s_unique[b1_idxs_l]) /
                (self.db_b1s_unique[b1_idxs_u] - self.db_b1s_unique[b1_idxs_l]),
                nan=0.0, posinf=0.0, neginf=0.0
            ),
            0.0, 1.0
        )[:, None]
        # linearly interpolate db entries
        db = torch.reshape(self.db_mag, (self.num_t2s, self.num_b1s, -1))
        db_t2_l_b1_l = db[t2_idxs_l, b1_idxs_l, :]
        db_t2_u_b1_l = db[t2_idxs_u, b1_idxs_l, :]
        db_t2_l_b1_u = db[t2_idxs_l, b1_idxs_u, :]
        db_t2_u_b1_u = db[t2_idxs_u, b1_idxs_u, :]
        # interpolate
        db_inter = (db_t2_l_b1_l * t2_w * b1_w + db_t2_u_b1_l * (1 - t2_w) * b1_w +
                    db_t2_l_b1_u * t2_w * (1 - b1_w) + db_t2_u_b1_u * (1 - t2_w) * (1 - b1_w))
        return db_inter

    def interpolate_db_rxv(self):
        # get t2 values
        t2_vals = torch.flatten(self.scale_to_range(values=self.estimates[0], identifier="t2")).detach().cpu().numpy()
        # get b1 values
        b1_vals = torch.flatten(self.scale_to_range(values=self.estimates[2], identifier="b1")).detach().cpu().numpy()
        # we want to interpolate the database for those values
        db_interp = scinterp.interpn(
            points=(self.db_t2s_s, self.db_b1s),
            values=self.db_mag.detach().cpu().numpy(),
            xi=(t2_vals, b1_vals)
        )
        return torch.from_numpy(db_interp)

    def interpolate_db_t2(self, db_b1_fixed: torch.tensor):
        # get t2 values
        t2_vals = torch.flatten(self.scale_to_range(values=self.estimates[0], identifier="t2")).detach().cpu().numpy()
        # we want to interpolate the database for those values
        db_interp = scinterp.interpn(
            points=self.db_t2s_s,
            values=db_b1_fixed.numpy(),
            xi=t2_vals
        )
        return torch.from_numpy(db_interp)

    def set_parameters(self, t2_t2p_b1: torch.tensor):
        self.estimates_t2 = t2_t2p_b1[0].to(self.device)
        self.estimates_r2p = t2_t2p_b1[1].to(self.device)
        self.estimates_b1 = t2_t2p_b1[2].to(self.device)

        # def sample_and_find_db_idx(self, t2_or_b1: bool):
        #     if t2_or_b1:
        #         # for t2
        #         t2_val = self.t2_range_ms[0] + (self.t2_range_ms[1] - self.t2_range_ms[0]) * self.estimates[0]
        #         t2_scale = self.t2_range_ms[0] + (self.t2_range_ms[1] - self.t2_range_ms[0]) * self.spread[0]
        #         # want to sample according to spread to get variations
        #         t2_val = torch.distributions.Normal(t2_val, t2_scale).sample()
        #         # find closest db entry
        #         t2_idx = torch.argmin((t2_val[:, :, None] - self.db_t2s_ms[None, None, :])**2, dim=-1)
        #         return t2_idx
        #     else:
        #         # for b1
        #         b1_val = self.b1_range[0] + (self.b1_range[1] - self.b1_range[0]) * self.estimates[2]
        #         b1_scale = self.b1_range[0] + (self.b1_range[1] - self.b1_range[0]) * self.spread[1]
        #         b1_val = torch.distributions.Normal(b1_val, b1_scale).sample()
        #         # want to find the next entry in db
        #         b1_idx = torch.argmin((b1_val[:, :, None] - self.db_b1s[None, None, :])**2, dim=-1)
        #         return b1_idx

    def get_etha(self):
        # input is t2p, in dims [x,y], get out attenuation factor based on time t and dims [xy, t]
        flat_r2p = self.get_candidate_values("r2p")
        return torch.exp(-self.delta_t_r2p_s[None, :] * flat_r2p[:, None])

    def get_maps(self):
        # we get the t2 & b1 values from the estimated means, scaled to the range
        # put back into whole volume
        t2 = torch.zeros((self.nx, self.ny), dtype=self.estimates_t2.dtype)
        r2p = torch.zeros((self.nx, self.ny), dtype=self.estimates_r2p.dtype)
        # reshape all
        t2[self.signal_mask[:, :, 0]] = self.get_candidate_values("t2").detach().cpu()
        r2p[self.signal_mask[:, :, 0]] = self.get_candidate_values("r2p").detach().cpu()
        b1 = self.get_candidate_values("b1_2d").detach().cpu()
        return t2, r2p, b1, self.t2_estimate_init, self.b1_estimate_init

    def get_db_from_t2_or_b1(self, identifier: str):
        if identifier not in ["t2", "b1"]:
            err = f"identifier must be one of: [t2, b1]"
            log_module.error(err)
            raise ValueError(err)
        db = torch.reshape(self.db_mag, (self.num_t2s, self.num_b1s, -1))
        vals = self.get_candidate_values(identifier=identifier)
        if identifier == "t2":
            t2_idx = torch.argmin((vals[:, None] - self.db_t2s_s_unique[None, :]) ** 2, dim=-1)
            db = db[t2_idx, :]
        else:
            b1_idx = torch.argmin((vals[:, None] - self.db_b1s_unique[None, :]) ** 2, dim=-1)
            db = db[:, b1_idx]
        return db

    def get_db_for_current_estimates(self):
        db = torch.reshape(self.db_mag, (self.num_t2s, self.num_b1s, -1))
        t2_vals = self.scale_to_range(torch.flatten(self.estimates[0]), identifier="t2")
        b1_vals = self.scale_to_range(torch.flatten(self.estimates[2]), identifier="b1")
        t2_idx = torch.argmin((t2_vals[:, None] - self.db_t2s_s_unique[None, :]) ** 2, dim=-1)
        b1_idx = torch.argmin((b1_vals[:, None] - self.db_b1s_unique[None, :]) ** 2, dim=-1)
        db = db[t2_idx, b1_idx]
        return db

    def forward(self):
        """Implement function to be optimised.
        In this case we get the input Signal per slice
        ||theta * eta - S||_l2 + lambda_b1 || B1 ||_Tv
        """
        # we get the b1 values for the latest estimate
        b1_map = self.get_candidate_values("b1_2d")

        # we calculate the total variation as one objective - weight for signal
        f_1 = torch.gradient(b1_map, dim=(0, 1))
        f_1 = self.lambda_b1_tv * torch.sum(torch.abs(f_1[0]) * self.weights + torch.abs(f_1[1]) * self.weights)
        # we want to cross pull the b1 and t2 values to the respective minimizing condition
        # get the database for the latest b1 candidates
        db_b1 = self.get_db_from_t2_or_b1(identifier="b1")
        # # dims db [t2, xy, t]
        # calculate attenuation factor by eta
        etha = self.get_etha()
        # dims db [t2, xy, t], dims eta = [xy, t]
        db_b1 *= etha[None]
        # need to normalize db
        db_b1 = torch.nan_to_num(
            torch.divide(db_b1, torch.linalg.norm(db_b1, dim=-1, keepdim=True)),
            posinf=0.0, nan=0.0
        )
        # match minimum l2 and find t2 idx
        l2_b1 = torch.linalg.norm(db_b1 - self.signal[None], dim=-1)
        t2_idx = torch.argmin(l2_b1, dim=0)
        t2_fit_vals = self.db_t2s_s_unique[t2_idx]
        # calculate mismatch
        candidate_t2 = self.get_candidate_values("t2")
        mismatch_t2 = torch.sqrt((candidate_t2 - t2_fit_vals) ** 2)
        f_2 = self.lambda_t2_dist * torch.mean(mismatch_t2)
        l2_min_f2 = self.lambda_t2_l2 * torch.mean(l2_b1[t2_idx])
        # pull towards initial
        f_2_init = self.lambda_t2_init * torch.mean(torch.sqrt(
            (candidate_t2 - self.t2_init) ** 2
        ))
        # get the database for the latest t2 candidates
        db_t2 = self.get_db_from_t2_or_b1(identifier="t2")
        # # dims db [xy, b1, t]
        # calculate attenuation factor by eta
        # dims db [xy, b1, t], dims eta = [xy, t]
        db_t2 *= etha[:, None, :]
        # need to normalize db
        db_t2 = torch.nan_to_num(
            torch.divide(db_t2, torch.linalg.norm(db_t2, dim=-1, keepdim=True)),
            posinf=0.0, nan=0.0
        )
        # match minimum l2 and find b1 idx
        l2_t2 = torch.linalg.norm(db_t2 - self.signal[:, None, :], dim=-1)
        b1_idx = torch.argmin(l2_t2, dim=1)
        b1_fit_vals = self.db_b1s_unique[b1_idx]
        candidate_b1 = self.get_candidate_values(identifier="b1")
        # calculate mismatch
        mismatch_b1 = torch.sqrt((candidate_b1 - b1_fit_vals) ** 2)
        f_3 = self.lambda_b1_dist * torch.mean(mismatch_b1)
        l2_min_f3 = self.lambda_b1_l2 * torch.mean(l2_t2[b1_idx])

        return f_1 + f_2 + f_2_init + l2_min_f2 + f_3 + l2_min_f3

    def yb_function(self, flat_t2_t2p_b1: np.ndarray):
        # reshape and torchify
        t2_t2p_b1 = torch.reshape(
            torch.from_numpy(flat_t2_t2p_b1),
            (3, self.nx, self.ny)
        )
        self.set_parameters(t2_t2p_b1=t2_t2p_b1)
        return self.forward().numpy()


def yb_optimization(model: DictionaryMatchingTv):
    loss = model.yb_function
    # optimize maps t2, t2p, b1, each with bounds (0,1), treat it as 1d vector of 3 times nx*ny dims
    bounds = [(0, 1)] * model.nx * model.ny * 3
    de = ybde(loss, bounds=bounds, maxiters=10, popsize=2 * len(bounds))
    for step in tqdm.tqdm(de.geniterator()):
        idx = step.best_idx
        norm_vector = step.population[idx]
        best_params = de.denormalize([norm_vector])
        log_module.info(step.best_fitness, norm_vector, best_params[0])
    return best_params


def optimization(model, optimizer, n=3):
    "Training loop for torch model."
    log_module.info("Optimize model")
    losses = []
    for i in tqdm.trange(n, desc="optimize matching for t2p and b1 Tv regularisation"):
        loss = model()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
    return losses


class BruteForce:
    def __init__(self, slice_signal: torch.tensor,
                 db_torch_mag: torch.tensor, db_t2s_s: torch.tensor, db_b1s: torch.tensor,
                 delta_t_r2p_ms: torch.tensor, device: torch.device = torch.device("cpu"),
                 r2p_range_Hz: tuple = (0.001, 200), r2p_sampling_size: int = 220):
        log_module.info("Brute Force matching algorithm")
        # save some vars
        # torch gpu processing
        self.device = device
        log_module.info(f"Set device: {device}")
        self.batch_size: int = 2000

        self.nx, self.ny, self.etl = slice_signal.shape
        self.delta_t_r2p_s: torch.tensor = 1e-3 * delta_t_r2p_ms
        # database and t2 and b1 values
        self.db_t2s_s: torch.tensor = db_t2s_s
        self.db_t2s_s_unique: torch.tensor = torch.unique(db_t2s_s).to(self.device)
        self.num_t2s: int = self.db_t2s_s_unique.shape[0]
        self.db_b1s: torch.tensor = db_b1s
        self.db_b1s_unique: torch.tensor = torch.unique(db_b1s).to(self.device)
        self.db_delta_b1: torch.tensor = torch.diff(self.db_b1s_unique)[0]
        self.num_b1s: int = self.db_b1s_unique.shape[0]
        self.db_mag: torch.tensor = db_torch_mag.to(self.device)
        self.db_mag_shaped = torch.reshape(db_torch_mag, (self.num_t2s, self.num_b1s, self.etl))

        self.signal = torch.reshape(slice_signal, (-1, self.etl)).to(self.device)

        # get initial estimate via SE data
        self.t2_estimate_init, self.b1_estimate_init = self.estimate_t2_b1_from_se()

        # set up db to include r2p
        self.db_r2ps_unique = torch.linspace(*r2p_range_Hz, r2p_sampling_size)
        self.num_r2ps = r2p_sampling_size
        etha = torch.exp(-self.delta_t_r2p_s[None, :] * self.db_r2ps_unique[:, None])
        self.db_mag_shaped = self.db_mag_shaped[:, None] * etha[None, :, None, :]
        self.db_mag_shaped = torch.nan_to_num(
            torch.divide(
                self.db_mag_shaped,
                torch.linalg.norm(self.db_mag_shaped, dim=-1, keepdim=True)
            ),
            nan=0.0, posinf=0.0
        )
        self.db_r2ps = torch.flatten(self.db_r2ps_unique[None, :, None].expand(self.num_t2s, -1, self.num_b1s))
        self.db_t2s_s = torch.flatten(
            torch.reshape(
                self.db_t2s_s, (self.num_t2s, self.num_b1s)
            )[:, None, :].expand(-1, self.num_r2ps, -1)
        )
        self.db_b1s = torch.flatten(
            torch.reshape(
                self.db_b1s, (self.num_t2s, self.num_b1s)
            )[:, None, :].expand(-1, self.num_r2ps, -1)
        )
        self.db_mag = torch.reshape(self.db_mag_shaped, (-1, self.etl)).to(device)

        self.estimates_t2 = torch.zeros((self.nx * self.ny), dtype=self.db_t2s_s.dtype)
        self.estimates_r2p = torch.zeros((self.nx * self.ny), dtype=self.db_r2ps.dtype)
        self.estimates_b1 = torch.zeros((self.nx * self.ny), dtype=self.db_b1s.dtype)
        # need to reduce batch size due to memory constraints
        self.batch_size: int = 20

    def estimate_t2_b1_from_se(self) -> (torch.tensor, torch.tensor):
        se_idx = self.delta_t_r2p_s < 1e-3
        db = self.db_mag[:, se_idx]
        db = torch.nan_to_num(
            torch.divide(db, torch.linalg.norm(db, dim=-1, keepdim=True)),
            posinf=0.0, nan=0.0
        )
        # need to batch data of whole slice
        batch_idx = torch.split(torch.arange(self.signal.shape[0]), self.batch_size)
        batch_data = torch.split(self.signal[:, se_idx], self.batch_size)
        t2_estimate_init = torch.zeros((self.nx * self.ny), dtype=self.db_t2s_s.dtype)
        b1_estimate_init = torch.zeros((self.nx * self.ny), dtype=self.db_b1s.dtype)
        for idx_batch in tqdm.trange(len(batch_idx), desc="match dictionary from SE reads:: unregularized brute force"):
            data_batch = batch_data[idx_batch]
            data_batch = torch.nan_to_num(
                torch.divide(data_batch, torch.linalg.norm(data_batch, dim=-1, keepdim=True)),
                posinf=0.0, nan=0.0
            )
            data_idx = batch_idx[idx_batch]
            # data dims [bs, t], db dims [t2-b1, t]
            l2_t2_b1 = torch.linalg.norm(data_batch[None, :] - db[:, None], dim=-1)
            fit_idx = torch.argmin(l2_t2_b1, dim=0).detach().cpu()
            t2_estimate_init[data_idx] = self.db_t2s_s[fit_idx]
            b1_estimate_init[data_idx] = self.db_b1s[fit_idx]
        t2_estimate_init = torch.reshape(t2_estimate_init, (self.nx, self.ny))
        b1_estimate_init = torch.reshape(b1_estimate_init, (self.nx, self.ny))
        del data_batch, db, l2_t2_b1, se_idx, batch_data
        torch.cuda.empty_cache()
        return t2_estimate_init, b1_estimate_init

    def estimate_values(self):
        # need to batch data of whole slice
        batch_idx = torch.split(torch.arange(self.signal.shape[0]), self.batch_size)
        batch_data = torch.split(self.signal, self.batch_size)
        for idx_batch in tqdm.trange(len(batch_idx), desc="match dictionary:: unregularized brute force"):
            data_batch = batch_data[idx_batch]
            data_batch = torch.nan_to_num(
                torch.divide(data_batch, torch.linalg.norm(data_batch, dim=-1, keepdim=True)),
                posinf=0.0, nan=0.0
            )
            data_idx = batch_idx[idx_batch]
            # data dims [bs, t], db dims [t2-b1, t]
            l2_t2_b1 = torch.linalg.norm(data_batch[None, :] - self.db_mag[:, None], dim=-1)
            fit_idx = torch.argmin(l2_t2_b1, dim=0).detach().cpu()
            self.estimates_t2[data_idx] = self.db_t2s_s[fit_idx]
            self.estimates_r2p[data_idx] = self.db_r2ps[fit_idx]
            self.estimates_b1[data_idx] = self.db_b1s[fit_idx]

    def get_maps(self):
        t2 = 1e3 * torch.reshape(self.estimates_t2, (self.nx, self.ny))
        r2p = torch.reshape(self.estimates_r2p, (self.nx, self.ny))
        b1 = torch.reshape(self.estimates_b1, (self.nx, self.ny))
        return t2.detach().cpu(), r2p.detach().cpu(), b1.detach().cpu(), self.t2_estimate_init, self.b1_estimate_init


def plot_loss(losses: list, save_path: plib.Path, title: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(x=np.arange(losses.__len__()), y=losses)
    )
    fig.update_layout(width=800, height=500)
    fig_name = save_path.joinpath(title).with_suffix(".html")
    log_module.info(f"write file: {fig_name}")
    fig.write_html(fig_name.as_posix())


def plot_maps(t2: torch.tensor, r2p: torch.tensor, b1: torch.tensor,
              t2_init: torch.tensor, b1_init: torch.tensor,
              save_path: plib.Path, title: str):
    fig = psub.make_subplots(
        rows=1, cols=5, shared_xaxes=True, shared_yaxes=True,
        column_titles=["T2 [ms]", "R2p [Hz]", "B1+", "T2 init [ms]", "B1+ init"],
        horizontal_spacing=0.01,
        vertical_spacing=0
    )
    zmin = [0, 0, 0.2, 0, 0.2]
    data_list = [t2, r2p, b1, t2_init, b1_init]
    for idx_data in range(len(data_list)):
        data = data_list[idx_data].numpy(force=True)

        fig.add_trace(
            go.Heatmap(
                z=data, transpose=True, zmin=zmin[idx_data], zmax=np.max(data),
                showscale=False, colorscale="Magma"
            ),
            row=1, col=1 + idx_data
        )
        if idx_data > 0:
            x = f"x{idx_data + 1}"
        else:
            x = "x"
        fig.update_xaxes(visible=False, row=1, col=1 + idx_data)
        fig.update_yaxes(visible=False, row=1, col=1 + idx_data, scaleanchor=x)

    fig.update_layout(
        width=1000, height=500
    )
    fig_name = save_path.joinpath(title).with_suffix(".html")
    log_module.info(f"write file: {fig_name}")
    fig.write_html(fig_name.as_posix())


def megesse_fit(
        fit_config: options.FitConfig, data_nii: torch.tensor, db_torch_mag: torch.tensor,
        db: DB, name: str, b1_nii=None):
    # data_nii = data_nii[:, :, 17:19, :]
    ep_path = plib.Path(fit_config.echo_props_path).absolute()
    if not ep_path.is_file():
        err = f"echo properties file: {ep_path} not found or not a file."
        log_module.error(err)
        raise FileNotFoundError(err)
    log_module.info(f"loading echo property file: {ep_path}")
    with open(ep_path.as_posix(), "r") as j_file:
        echo_props = json.load(j_file)

    if echo_props.__len__() < db_torch_mag.shape[-1]:
        warn = "echo type list not filled or shorter than database etl. filling with SE type acquisitions"
        log_module.warning(warn)
    while echo_props.__len__() < db_torch_mag.shape[-1]:
        # if the list is too short or insufficiently filled we assume SE acquisitions
        echo_props[echo_props.__len__()] = {"type": "SE", "te_ms": 0.0, "time_to_adj_se_ms": 0.0}
    # possibly need some kind of convenient class to coherently store information
    # need tensor to hold time deltas to SE
    delta_t_ms_to_se = []
    for idx_e in range(echo_props.__len__()):
        delta_t_ms_to_se.append((echo_props[str(idx_e)]["time_to_adj_se_ms"]))
    delta_t_ms_to_se = torch.abs(torch.tensor(delta_t_ms_to_se))

    # get values
    t2_vals = torch.from_numpy(db.pd_dataframe[db.pd_dataframe["echo"] == 1]["t2"].values)
    b1_vals = torch.from_numpy(db.pd_dataframe[db.pd_dataframe["echo"] == 1]["b1"].values)
    # we want to use the torch ADAM optimizer to optimize our function exploiting torchs internal tools
    # implement slice wise
    shape = data_nii.shape[:3]
    t2 = torch.zeros(shape)
    r2p = torch.zeros(shape)
    b1 = torch.zeros(shape)
    t2_init = torch.zeros(shape)
    b1_init = torch.zeros(shape)
    for idx_slice in range(data_nii.shape[2]):
        log_module.info(f"Process slice {idx_slice + 1} of {data_nii.shape[2]}")
        # # try autograd model
        # # set up model for slice
        # slice_optimize_model = DictionaryMatchingTv(
        #     db_torch_mag=db_torch_mag, db_t2s_s=t2_vals, db_b1s=b1_vals, slice_signal=data_nii[:, :, idx_slice],
        #     delta_t_r2p_ms=delta_t_ms_to_se, device=torch.device("cuda:0"),
        #     t2_range_s=(torch.min(t2_vals), torch.max(t2_vals)), autograd=True,
        #     b1_range=(torch.min(b1_vals), torch.max(b1_vals)), r2p_range_Hz=(0.01, 200)
        # )
        # # Instantiate optimizer
        # # torch autograd
        # # opt = torch.optim.Adam(slice_optimize_model.parameters(), lr=0.02)
        # opt = torch.optim.SGD(slice_optimize_model.parameters(), lr=0.02, momentum=0.9)
        # losses = optimization(model=slice_optimize_model, optimizer=opt, n=500)

        # brute force
        slice_optimize_model = BruteForce(
            db_torch_mag=db_torch_mag, db_t2s_s=t2_vals, db_b1s=b1_vals, slice_signal=data_nii[:, :, idx_slice],
            delta_t_r2p_ms=delta_t_ms_to_se, device=torch.device("cuda:0"),
            r2p_range_Hz=(0.01, 100), r2p_sampling_size=200
        )
        slice_optimize_model.estimate_values()

        # yabox
        # yb_best = yb_optimization(slice_optimize_model)
        # this is a 3 * nx * ny vector bound between 0, 1, can use the defined model to translate this to the maps
        # t2_t2p_b1 = torch.reshape(
        #     torch.from_numpy(yb_best[0]),
        #     (3, slice_optimize_model.nx, slice_optimize_model.ny)
        # )
        # slice_optimize_model.set_parameters(t2_t2p_b1=t2_t2p_b1)

        # get slice maps
        (t2[:, :, idx_slice], r2p[:, :, idx_slice], b1[:, :, idx_slice], t2_init[:, :, idx_slice],
         b1_init[:, :, idx_slice]) = slice_optimize_model.get_maps()
        save_path = plib.Path(fit_config.save_path)
        # plot_loss(losses, save_path=save_path, title=f'losses_slice_{idx_slice + 1}')
        plot_maps(t2, r2p, b1, t2_init, b1_init, save_path=save_path, title=f"maps_slice{idx_slice + 1}")

    r2, pd = (None, None)
    return t2.numpy(), r2p.numpy(), r2, b1.numpy(), pd


def mese_fit(
        fit_config: options.FitConfig, data_nii: torch.tensor, name: str,
        db_torch_mag: torch.tensor, db: DB, device: torch.device, b1_nii=None):
    # get scaling of signal as one factor to compute pd
    rho_s = torch.linalg.norm(data_nii, dim=-1, keepdim=True)
    # for a first approximation we fit only the spin echoes
    # make 2d [xyz, t]
    data_nii_input_shape = data_nii.shape
    data_nii = torch.reshape(data_nii, (-1, data_nii.shape[-1]))
    # l2 normalize data - db is normalized
    data_nii = torch.nan_to_num(
        torch.divide(data_nii, torch.linalg.norm(data_nii, dim=-1, keepdim=True)),
        nan=0.0, posinf=0.0
    ).to(device)
    # nii_data = torch.reshape(data_nii_se, (-1, data_nii_input_shape[-1])).to(device)
    num_flat_dim = data_nii.shape[0]
    db_torch_mag = db_torch_mag.to(device)
    # get emc simulation norm
    # db_torch_norm = torch.squeeze(torch.from_numpy(db_norm)).to(device)
    db_torch_norm = torch.linalg.norm(db_torch_mag, dim=-1, keepdim=True)
    db_torch_mag = torch.nan_to_num(
        torch.divide(db_torch_mag, db_torch_norm), posinf=0.0, nan=0.0
    )
    db_torch_norm = torch.squeeze(db_torch_norm)

    batch_size = 3000
    nii_idxs = torch.split(torch.arange(num_flat_dim), batch_size)
    nii_zero = torch.sum(torch.abs(data_nii), dim=-1) < 1e-6
    nii_data = torch.split(data_nii, batch_size, dim=0)
    # make scaling map
    t2_vals = torch.from_numpy(db.pd_dataframe[db.pd_dataframe["echo"] == 1]["t2"].values).to(device)
    b1_vals = torch.from_numpy(db.pd_dataframe[db.pd_dataframe["echo"] == 1]["b1"].values).to(device)

    # b1 penalty of b1 database vs b1 input
    # dim b1 [xyz], db b1 - values for each entry [t1 t2 b1]
    if b1_nii is not None:
        b1_scale = fit_config.b1_tx_scale
        b1_nii = torch.reshape(b1_nii, (num_flat_dim,)).to(device)
        if torch.max(b1_nii) > 10:
            b1_nii = b1_nii / 100
        b1_nii *= b1_scale
        b1_nii = torch.split(b1_nii, batch_size, dim=0)
        use_b1 = True
        b1_weight = fit_config.b1_weight
        name = f"{name}_b1-in-w-{b1_weight}".replace(".", "p")
        if abs(1.0 - b1_scale) > 1e-3:
            name = f"{name}_tx-scale-{b1_scale:.2f}"
    else:
        use_b1 = False
        b1_weight = 0.0
        b1_scale = 1.0

    t2 = torch.zeros(num_flat_dim, dtype=t2_vals.dtype, device=device)
    b1 = torch.zeros(num_flat_dim, dtype=b1_vals.dtype, device=device)
    # get scaling of db curves as one factor to compute pd
    rho_theta = torch.zeros(num_flat_dim, dtype=rho_s.dtype, device=device)

    # need to bin data for memory reasons
    for idx in tqdm.trange(len(nii_data), desc="batch processing"):
        data_batch = nii_data[idx]

        # l2 norm difference of magnitude data vs magnitude database
        # calculate difference, dims db [t2s t1 t2 b1, t], nii-batch [x*y*z*,t]
        l2_norm_diff = torch.linalg.vector_norm(db_torch_mag[:, None] - data_batch[None, :], dim=-1)

        # b1 penalty
        if use_b1:
            b1_batch = b1_scale * b1_nii[idx]
            b1_penalty = torch.sqrt(torch.square(b1_vals[:, None] - b1_batch[None, :]))
        else:
            b1_penalty = 0.0

        evaluation_matrix = b1_weight * b1_penalty + (1.0 - b1_weight) * l2_norm_diff

        # find minimum index in db dim
        min_idx = torch.argmin(evaluation_matrix, dim=0)
        # populate maps
        t2[nii_idxs[idx]] = t2_vals[min_idx]
        b1[nii_idxs[idx]] = b1_vals[min_idx]
        rho_theta[nii_idxs[idx]] = db_torch_norm[min_idx]
    # set t2 0 for signal 0 (eg. for bet) We could in principle use this to reduce computation demands
    # by not needing to compute those entries,
    # however we want to estimate the b1
    t2[nii_zero] = 0.0

    # reshape
    if torch.max(t2) < 5:
        # cast to ms
        t2 = 1e3 * t2
    t2 = torch.reshape(t2, data_nii_input_shape[:-1])
    r2 = torch.nan_to_num(1000.0 / t2, nan=0.0, posinf=0.0)
    t2 = t2.numpy(force=True)
    r2 = r2.numpy(force=True)
    b1 = torch.reshape(b1, data_nii_input_shape[:-1]).numpy(force=True)
    rho_theta[nii_zero] = 0.0
    rho_theta = torch.reshape(rho_theta, data_nii_input_shape[:-1]).cpu()

    pd = torch.nan_to_num(
        torch.divide(
            torch.squeeze(rho_s),
            torch.squeeze(rho_theta)
        ),
        nan=0.0, posinf=0.0
    )
    # we want to calculate histograms for both, and find upper cutoffs of the data values based on the histograms
    # since both might explode
    pd_hist, pd_bins = torch.histogram(pd.flatten(), bins=200)
    # find percentage where 95 % of data lie
    pd_hist_perc = torch.cumsum(pd_hist, dim=0) / torch.sum(pd_hist, dim=0)
    pd_cutoff_value = pd_bins[torch.nonzero(pd_hist_perc > 0.95)[0].item()]
    pd = torch.clamp(pd, min=0.0, max=pd_cutoff_value).numpy(force=True)
    return t2, r2, b1, pd


def main(fit_config: options.FitConfig):
    # set path
    path = plib.Path(fit_config.save_path).absolute()
    log_module.info(f"setup save path: {path.as_posix()}")
    path.mkdir(parents=True, exist_ok=True)

    # load in data
    if fit_config.save_name_prefix:
        fit_config.save_name_prefix += f"_"
    in_path = plib.Path(fit_config.nii_path).absolute()
    stem = in_path.stem
    for suffix in in_path.suffixes:
        stem = stem.removesuffix(suffix)
    name = f"{fit_config.save_name_prefix}{stem}"

    log_module.info("__ Load data")
    data_nii, db, b1_nii, data_affine, b1_affine = io.load_data(fit_config=fit_config)
    data_nii_input_shape = data_nii.shape

    # for now take only magnitude data
    db_mag, db_phase, db_norm = db.get_numpy_array_ids_t()
    # device
    device = torch.device("cuda:0")

    # set echo types
    if not fit_config.echo_props_path:
        warn = "no echo properties given, assuming SE type echo train."
        log_module.warning(warn)
        t2, r2, b1, pd = mese_fit(
            fit_config=fit_config, data_nii=data_nii, name=name, db_torch_mag=torch.from_numpy(db_mag),
            db=db, device=device, b1_nii=b1_nii
        )
        r2p = None
    else:
        t2, r2p, r2, b1, pd = megesse_fit(
            fit_config=fit_config, data_nii=data_nii, db_torch_mag=torch.from_numpy(db_mag),
            db=db, name=name, b1_nii=b1_nii
        )

    # save
    names = [f"t2", f"r2", f"b1", f"pd_like"]
    data = [t2, r2, b1, pd]
    if r2p is not None:
        data.append(r2p)
        names.append("r2p")
    for idx in range(len(data)):
        if data[idx] is None:
            if names[idx] == "r2":
                data[idx] = np.divide(1e3, data[0], where=data[0] > 1e-5, out=np.zeros_like(data[0]))
            else:
                data[idx] = np.zeros_like(data[0])
        save_nii_data(data=data[idx], affine=data_affine, name_prefix=name, name=names[idx], path=path)


def save_nii_data(data, name: str, path: plib.Path, name_prefix: str, affine):
    save_name = f"{name_prefix}_{name}"
    img = nib.Nifti1Image(data, affine=affine)
    file_name = path.joinpath(save_name).with_suffix(".nii")
    logging.info(f"write file: {file_name.as_posix()}")
    nib.save(img, file_name.as_posix())


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    logging.info("_________________________________________________________")
    logging.info("___________________ EMC torch fitting ___________________")
    logging.info("_________________________________________________________")

    parser, prog_args = options.create_cli()

    opts = options.FitConfig.from_cli(prog_args)
    # set logging level after possible config file read
    if opts.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    try:
        main(fit_config=opts)

    except Exception as e:
        logging.exception(e)
        parser.print_usage()
