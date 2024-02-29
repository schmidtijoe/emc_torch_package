import pathlib as plib
import nibabel as nib
import torch
import tqdm
import json
from emc_torch import DB
from emc_torch.fitting import options, io
import logging

log_module = logging.getLogger(__name__)
logging.getLogger('simple_parsing').setLevel(logging.WARNING)


def megesse_fit(
        fit_config: options.FitConfig, data_nii: torch.tensor, db_torch_mag: torch.tensor,
        db: DB, name:str, b1_nii=None):
    ep_path = plib.Path(fit_config.echo_props_path).absolute()
    if not ep_path.is_file():
        err = f"echo properties file: {ep_path} not found or not a file."
        log_module.error(err)
        raise FileNotFoundError(err)
    log_module.info(f"loading echo property file: {ep_path}")
    with open(ep_path.as_posix(), "r") as j_file:
        echo_props = json.load(j_file)
    fit_se = False
    if echo_props.__len__() < db_torch_mag.shape[-1]:
        warn = "echo type list not filled or shorter than database etl. filling with SE type acquisitions"
        log_module.warning(warn)
    while echo_props.__len__() < db_torch_mag.shape[-1]:
        # if the list is too short or insufficiently filled we assume SE acquisitions
        echo_props[echo_props.__len__()] = {"type": "SE", "te_ms": 0.0, "time_to_adj_se_ms": 0.0}
    # possibly need some kind of convenient class to coherently store information
    # get number of SE and GRE
    idx_se = []
    idx_gre = []
    for idx_e in range(echo_props.__len__()):
        if echo_props[str(idx_e)]["type"] == "SE":
            idx_se.append(idx_e)
        elif echo_props[str(idx_e)]["type"] == "GRE":
            idx_gre.append(idx_e)
        else:
            err = "unknown echo type in separating SE and GRE data"
            log_module.error(err)
            raise AttributeError(err)
    num_se = len(idx_se)
    num_gre = len(idx_gre)

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

    # we could just sample another value grid
    # t2_p_resolution = 50
    # t2_p_values_ms = torch.linspace(1, 100, t2_p_resolution)
    # t2_p_effect = torch.zeros((t2_p_resolution, db_mag.shape[-1]))
    # for idx_e in range(db_mag.shape[-1]):
    #     # calculate t2p effect on curve signal value
    #     t_eff_ms = abs(echo_props[str(idx_e)]["time_to_adj_se_ms"])
    #     t2_p_effect[:, idx_e] = torch.exp(-(t_eff_ms / t2_p_values_ms))

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
