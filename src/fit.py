import pathlib as plib
import nibabel as nib
import torch
import tqdm
from emc_torch.fit import options, io
import logging

log_module = logging.getLogger(__name__)
logging.getLogger('simple_parsing').setLevel(logging.WARNING)


def main(fit_config: options.FitConfig):
    # set path
    path = plib.Path(fit_config.save_path).absolute()
    log_module.info(f"setup save path: {path.as_posix()}")
    path.mkdir(parents=True, exist_ok=True)

    # load in data
    log_module.info("__ Load data")
    data_nii, db, b1_nii, data_affine, b1_affine = io.load_data(fit_config=fit_config)
    data_nii_input_shape = data_nii.shape

    # for now take only magnitude data
    db_mag, _ = db.get_numpy_array_ids_t()
    # device
    device = torch.device("cuda:0")

    # make 2d [xyz, t]
    nii_data = torch.reshape(data_nii, (-1, data_nii_input_shape[-1])).to(device)
    num_flat_dim = nii_data.shape[0]
    db_torch_mag = torch.from_numpy(db_mag).to(device)
    batch_size = 3000
    nii_idxs = torch.split(torch.arange(nii_data.shape[0]), batch_size)
    nii_data = torch.split(nii_data, batch_size, dim=0)

    t2_vals = torch.from_numpy(db.pd_dataframe[db.pd_dataframe["echo"] == 1]["t2"].values).to(device)
    b1_vals = torch.from_numpy(db.pd_dataframe[db.pd_dataframe["echo"] == 1]["b1"].values).to(device)

    # b1 penalty of b1 database vs b1 input
    # dim b1 [xyz], db b1 - values for each entry [t1 t2 b1]
    if b1_nii is not None:
        b1_nii = torch.reshape(b1_nii / 100, (num_flat_dim,)).to(device)
        b1_nii = torch.split(b1_nii, batch_size, dim=0)
        use_b1 = True
        b1_weight = fit_config.b1_weight
    else:
        use_b1 = False
        b1_weight = 0.0

    t2 = torch.zeros(num_flat_dim, dtype=t2_vals.dtype, device=device)
    b1 = torch.zeros(num_flat_dim, dtype=b1_vals.dtype, device=device)
    # need to bin data for memory reasons
    for idx in tqdm.trange(len(nii_data), desc="batch processing"):
        data_batch = nii_data[idx]

        # l2 norm difference of magnitude data vs magnitude database
        # calculate difference, dims db [t1 t2 b1, t], nii-batch [x*y*z*,t]
        l2_norm_diff = torch.linalg.vector_norm(db_torch_mag[:, None] - data_batch[None, :], dim=-1)

        # b1 penalty
        if use_b1:
            b1_batch = b1_nii[idx]
            b1_penalty = torch.sqrt(torch.square(b1_vals[:, None] - b1_batch[None, :]))
        else:
            b1_penalty = 0.0

        evaluation_matrix = b1_weight * b1_penalty + (1.0 - b1_weight) * l2_norm_diff

        # find minimum index in db dim
        min_idx = torch.argmin(evaluation_matrix, dim=0)
        # populate maps
        t2[nii_idxs[idx]] = t2_vals[min_idx]
        b1[nii_idxs[idx]] = b1_vals[min_idx]

    # reshape
    t2 = torch.reshape(t2, data_nii_input_shape[:-1])
    r2 = torch.nan_to_num(1000.0 / t2, nan=0.0, posinf=0.0)
    t2 = t2.numpy(force=True)
    r2 = r2.numpy(force=True)
    b1 = torch.reshape(b1, data_nii_input_shape[:-1]).numpy(force=True)

    # save
    names = [f"t2_inb1-{bool(fit_config.b1_file)}_pen-{b1_weight:.2f}",
             f"r2_inb1-{bool(fit_config.b1_file)}_pen-{b1_weight:.2f}",
             f"b1_inb1-{bool(fit_config.b1_file)}_pen-{b1_weight:.2f}"]
    data = [t2, r2, b1]
    for idx in range(3):
        name = names[idx].replace(".", "p")
        img = nib.Nifti1Image(data[idx], affine=data_affine)
        file_name = path.joinpath(f"{fit_config.save_name}_{name}").with_suffix(".nii")
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
