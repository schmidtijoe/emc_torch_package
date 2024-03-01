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

log_module = logging.getLogger(__name__)
logging.getLogger('simple_parsing').setLevel(logging.WARNING)


class DictionaryMatchingTv(nn.Module):
    """Custom Pytorch model for gradient optimization of our function
    """
    def __init__(
            self, slice_signal: torch.tensor,
            db_torch_mag: torch.tensor, db_t2s_ms: torch.tensor, db_b1s: torch.tensor,
            delta_t_t2p_ms: torch.tensor, nx: int, ny: int, device: torch.device = torch.device("cpu"),
            lambda_b1: float = 0.5, t2_range_ms: tuple = (1, 1000), t2p_range_ms: tuple = (1, 500),
            b1_range: tuple = (0.2, 1.6)):
        super().__init__()
        # save setup as normalized curves
        signal = torch.nan_to_num(
            torch.divide(slice_signal, torch.linalg.norm(slice_signal, dim=-1, keepdim=True)),
            nan=0.0, posinf=0.0
        )
        self.device = device
        log_module.info(f"set torch device: {device}")
        # reshape in dims [xy, t]
        self.signal = torch.reshape(signal, (-1, signal.shape[-1])).to(self.device)
        # want to save some params
        # dimensions within slice
        self.nx: int = nx
        self.ny: int = ny
        self.slice_shape: tuple = (nx, ny)
        # database and t2 and b1 values
        self.db_t2s_ms: torch.tensor = db_t2s_ms
        self.num_t2s: int = db_t2s_ms.shape[0]
        self.db_b1s: torch.tensor = db_b1s
        self.num_b1s: int = db_b1s.shape[0]
        self.db_mag: torch.tensor = torch.reshape(db_torch_mag, (self.num_t2s, self.num_b1s, -1)).to(self.device)
        # echo train length and corresponding timings for the attenuation factor as delta to next SE
        self.etl: int = db_torch_mag.shape[-1]
        self.delta_t_t2p_ms: torch.tensor = delta_t_t2p_ms
        # fit weighting of Tv B1
        self.lambda_b1: float = lambda_b1
        # range of parameters to consider
        self.t2_range_ms: tuple = t2_range_ms
        self.t2p_range_ms: tuple = t2p_range_ms
        self.b1_range: tuple = b1_range
        # want to optimize for T2s and B1 for a slice of the image
        t2p_b1 = torch.distributions.Uniform(0, 1).sample((2, nx, ny)).to(self.device)
        # initialize weights with random numbers
        # make weights torch parameters
        self.estimates: nn.Parameter = nn.Parameter(t2p_b1)
        self.t2_estimate: torch.tensor = torch.zeros((nx, ny))

    def scale_to_range(self, values: torch.tensor, identifier: str):
        if identifier == "t2":
            v_range = self.t2_range_ms
        elif identifier == "t2p":
            v_range = self.t2p_range_ms
        elif identifier == "b1":
            v_range = self.b1_range
        else:
            err = "identifier not recognized"
            log_module.error(err)
            raise ValueError(err)
        return v_range[0] + (v_range[1] - v_range[0]) * values

    def interpolate_db(self):
        # get t2 values
        t2_vals = torch.flatten(self.scale_to_range(values=self.estimates[0], identifier="t2")).detach().numpy()
        # get b1 values
        b1_vals = torch.flatten(self.scale_to_range(values=self.estimates[2], identifier="b1")).detach().numpy()
        # we want to interpolate the database for those values
        db_interp = scinterp.interpn(
            points=(self.db_t2s_ms, self.db_b1s),
            values=self.db_mag.numpy(),
            xi=(t2_vals, b1_vals)
        )
        return torch.from_numpy(db_interp)


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
        flat_t2p = torch.flatten(self.estimates[0])
        # scale to ms range
        flat_t2p = self.scale_to_range(values=flat_t2p, identifier="t2p")
        return torch.exp(-self.delta_t_t2p_ms[None, :] / flat_t2p[:, None])

    def get_maps(self):
        # we get the t2 & b1 values from the estimated means, scaled to the range
        # t2 = self.scale_to_range(self.estimates[0], identifier="t2")
        b1 = self.scale_to_range(self.estimates[2], identifier="b1")
        t2p = self.scale_to_range(self.estimates[1], identifier="t2p")
        # reshape all
        # t2 = torch.reshape(t2, (self.nx, self.ny))
        t2p = torch.reshape(t2p, (self.nx, self.ny))
        b1 = torch.reshape(b1, (self.nx, self.ny))
        return self.t2_estimate, t2p, b1

    def get_db_from_b1(self, b1_vals: torch.tensor):
        b1_idx = torch.argmin((b1_vals[:, None] - self.db_b1s[None, :])**2, dim=-1)
        db = self.db_mag[:, b1_idx]
        return db

    def forward(self):
        """Implement function to be optimised.
        In this case we get the input Signal per slice
        ||theta * eta - S||_l2 + lambda_b1 || B1 ||_Tv
        """
        # we get the b1 values for the latest estimate
        b1 = self.scale_to_range(self.estimates[1], identifier="b1")
        # we calculate the total variation as one objective
        f_2 = torch.gradient(b1, dim=(0, 1))
        f_2 = torch.sum(torch.abs(f_2[0]) + torch.abs(f_2[1]))
        # we get the database for the closest matching b1 entries
        db = self.get_db_from_b1(torch.flatten(b1))
        # calculate attenuation factor by eta
        # dims db [t2, xy, t], dims eta = [xy, t]
        s_db = db * self.get_etha()[None]
        # need to normalize data and db
        s_db = torch.nan_to_num(
            torch.divide(s_db, torch.linalg.norm(s_db, dim=-1, keepdim=True)),
            posinf=0.0, nan=0.0
        )
        # find t2 indexes
        l2_min = torch.linalg.norm(s_db - self.signal[None], dim=-1)
        t2_idx = torch.argmin(l2_min, dim=0)
        # set estimate
        self.t2_estimate = torch.reshape(self.db_t2s_ms[t2_idx], (self.nx, self.ny))
        # calculate l2 value as second objective to optimize t2p
        f_1 = torch.linalg.norm(l2_min, dim=-1)
        f_1 = torch.sum(f_1)

        return f_1 + f_2


def optimization(model, optimizer, n=3):
    "Training loop for torch model."
    losses = []
    for i in tqdm.trange(n):
        loss = model()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return losses


def plot_loss(losses: list, save_path: plib.Path, title: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(x=np.arange(losses.__len__()), y=losses)
    )
    fig.update_layout(width=800, height=500)
    fig_name = save_path.joinpath(title).with_suffix(".html")
    log_module.info(f"write file: {fig_name}")
    fig.write_html(fig_name.as_posix())

def plot_maps(t2: torch.tensor, t2p: torch.tensor, b1:torch.tensor, save_path: plib.Path, title: str):
    fig = psub.make_subplots(
        rows=1, cols=3, shared_xaxes=True, shared_yaxes=True,
        column_titles=["T2 [ms]", "T2p [ms]", "B1+"],
        horizontal_spacing=0.01,
        vertical_spacing=0
    )
    zmin = [0, 0, 0.2]
    zmax = [100, 100, 1.6]
    for idx_data in range(3):
        data = [t2, t2p, b1][idx_data].numpy(force=True)

        fig.add_trace(
            go.Heatmap(
                z=data, transpose=True, zmin=zmin[idx_data], zmax=zmax[idx_data],
                showscale=False, colorscale="Magma"
            ),
            row=1, col=1+idx_data
        )
        if idx_data > 0:
            x = f"x{idx_data+1}"
        else:
            x = "x"
        fig.update_xaxes(visible=False, row=1, col=1+idx_data)
        fig.update_yaxes(visible=False, row=1, col=1+idx_data, scaleanchor=x)

    fig.update_layout(
        width=1000, height=500
    )
    fig_name = save_path.joinpath(title).with_suffix(".html")
    log_module.info(f"write file: {fig_name}")
    fig.write_html(fig_name.as_posix())


def megesse_fit(
        fit_config: options.FitConfig, data_nii: torch.tensor, db_torch_mag: torch.tensor,
        db: DB, name: str, b1_nii=None):
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
    t2_vals = torch.from_numpy(db.pd_dataframe[db.pd_dataframe["echo"] == 1]["t2"].unique())
    b1_vals = torch.from_numpy(db.pd_dataframe[db.pd_dataframe["echo"] == 1]["b1"].unique())
    # we want to use the torch ADAM optimizer to optimize our function exploiting torchs internal tools
    # implement slice wise
    for idx_slice in range(data_nii.shape[2]):
        log_module.info(f"Process slice {idx_slice + 1} of {data_nii.shape[2]}")
        # set up model for slice
        slice_optimize_model = DictionaryMatchingTv(
            db_torch_mag=db_torch_mag, db_t2s_ms=t2_vals*1e3, db_b1s=b1_vals, slice_signal=data_nii[:, :, idx_slice],
            delta_t_t2p_ms=delta_t_ms_to_se, nx=data_nii.shape[0], ny=data_nii.shape[1],
            t2_range_ms=(1e3*torch.min(t2_vals), 1e3*torch.max(t2_vals)),
            b1_range=(torch.min(b1_vals), torch.max(b1_vals))
        )
        # Instantiate optimizer
        opt = torch.optim.SGD(slice_optimize_model.parameters(), lr=0.01, momentum=0.9)
        losses = optimization(model=slice_optimize_model, optimizer=opt, n=300)
        # get slice maps
        t2, t2p, b1 = slice_optimize_model.get_maps()
        save_path = plib.Path(fit_config.save_path)
        plot_loss(losses, save_path=save_path, title=f'losses_slice_{idx_slice+1}')
        plot_maps(t2, t2p, b1, save_path=save_path, title=f"maps_slice{idx_slice+1}")

    t2, t2p, r2, b1, pd = (None, None, None, None, None)
    return t2, t2p, r2, b1, pd


def mese_fit(
        fit_config: options.FitConfig, data_nii: torch.tensor, name:str,
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
        t2p = None
    else:
        t2, t2p, r2, b1, pd = megesse_fit(
            fit_config=fit_config, data_nii=data_nii, db_torch_mag=torch.from_numpy(db_mag),
            db=db, name=name, b1_nii=b1_nii
        )

    # save
    names = [f"t2", f"r2", f"b1", f"pd_like"]
    data = [t2, r2, b1, pd]
    if t2p is not None:
        data.append(t2p)
        names.append("t2p")
    for idx in range(len(data)):
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
