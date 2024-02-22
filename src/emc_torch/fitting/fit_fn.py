import pathlib as plib
import nibabel as nib
import torch
import tqdm
from emc_torch.fitting import options, io
import logging

log_module = logging.getLogger(__name__)
logging.getLogger('simple_parsing').setLevel(logging.WARNING)


def fit(
        nii_path: str, save_path: str,
        database_file: str, b1_file: str = "", b1_weight: float = 0.5,
        save_name_prefix: str = "",
        b1_tx_scale: float = 1.0,
        visualize: bool = True,
        debug: bool = False,
        use_gpu: bool = True,
        gpu_device: int = 0):
    fit_config = options.FitConfig(
        nii_path=nii_path,
        database_file=database_file,
        b1_file=b1_file,
        b1_weight=b1_weight,
        save_path=save_path,
        save_name_prefix=save_name_prefix,
        b1_tx_scale=b1_tx_scale,
        visualize=visualize,
        debug=debug,
        use_gpu=use_gpu,
        gpu_device=gpu_device
    )
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    logging.info("_________________________________________________________")
    logging.info("___________________ EMC torch fitting ___________________")
    logging.info("_________________________________________________________")

    if fit_config.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    try:
        main(fit_config=fit_config)

    except Exception as e:
        logging.exception(e)


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
    if fit_config.use_gpu:
        device = torch.device(f"cuda:{fit_config.gpu_device}")
    else:
        device = torch.device("cpu")

    # make 2d [xyz, t]
    nii_scale = torch.linalg.norm(data_nii, dim=-1, keepdim=True)
    # l2 normalize data - db is normalized
    data_nii = torch.nan_to_num(
        torch.divide(data_nii, nii_scale),
        nan=0.0, posinf=0.0
    )
    nii_data = torch.reshape(data_nii, (-1, data_nii_input_shape[-1])).to(device)
    num_flat_dim = nii_data.shape[0]
    db_torch_mag = torch.from_numpy(db_mag).to(device)
    db_torch_norm = torch.squeeze(torch.from_numpy(db_norm).to(device))
    batch_size = 3000
    nii_idxs = torch.split(torch.arange(nii_data.shape[0]), batch_size)
    nii_zero = torch.sum(torch.abs(nii_data), dim=-1) < 1e-6
    nii_data = torch.split(nii_data, batch_size, dim=0)
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
            name = f"{name}_tx-scale-{b1_scale:.2f}".replace(".", "p")
    else:
        use_b1 = False
        b1_weight = 0.0
        b1_scale = 1.0

    t2 = torch.zeros(num_flat_dim, dtype=t2_vals.dtype, device=device)
    b1 = torch.zeros(num_flat_dim, dtype=b1_vals.dtype, device=device)
    s0 = torch.zeros(num_flat_dim, dtype=nii_scale.dtype, device=device)

    # need to bin data for memory reasons
    for idx in tqdm.trange(len(nii_data), desc="batch processing"):
        data_batch = nii_data[idx]

        # l2 norm difference of magnitude data vs magnitude database
        # calculate difference, dims db [t1 t2 b1, t], nii-batch [x*y*z*,t]
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
        s0[nii_idxs[idx]] = db_torch_norm[min_idx]
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
    s0[nii_zero] = 0.0
    s0 = torch.reshape(s0, data_nii_input_shape[:-1]).cpu()
    pd_like = torch.nan_to_num(
        torch.divide(
            torch.squeeze(nii_scale),
            torch.squeeze(s0)
        ),
        nan=0.0, posinf=0.0
    )
    double_weighting = torch.squeeze(nii_scale) * torch.squeeze(s0)
    # we want to calculate histograms for both, and find upper cutoffs of the data values based on the histograms
    # since both might explode
    dw_hist, dw_bins = torch.histogram(double_weighting.flatten(), bins=200)
    pd_hist, pd_bins = torch.histogram(pd_like.flatten(), bins=200)
    # find percentage where 99.5 % of data lie
    dw_hist_perc = torch.cumsum(dw_hist, dim=0) / torch.sum(dw_hist, dim=0)
    pd_hist_perc = torch.cumsum(pd_hist, dim=0) / torch.sum(pd_hist, dim=0)

    dw_cutoff_value = dw_bins[torch.nonzero(dw_hist_perc > 0.995)[0].item()]
    pd_cutoff_value = pd_bins[torch.nonzero(pd_hist_perc > 0.99)[0].item()]

    double_weighting = torch.clamp(double_weighting, min=0.0, max=dw_cutoff_value).numpy(force=True)
    pd_like = torch.clamp(pd_like, min=0.0, max=pd_cutoff_value).numpy(force=True)

    # save
    names = [f"t2", f"r2", f"b1", f"pd_like_scaling", "d_norm_w"]
    data = [t2, r2, b1, pd_like, double_weighting]
    for idx in range(len(data)):
        save_name = f"{name}_{names[idx]}"
        img = nib.Nifti1Image(data[idx], affine=data_affine)
        file_name = path.joinpath(save_name).with_suffix(".nii")
        logging.info(f"write file: {file_name.as_posix()}")
        nib.save(img, file_name.as_posix())


def l2_diff_einsum(tensor_a: torch.tensor, tensor_b: torch.tensor):
    # dims a [n,t], b[m, t]
    diff = tensor_a[:, None] - tensor_b[None, :]
    return torch.sqrt(torch.einsum("nmt, nmt -> nm", diff, diff))


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
