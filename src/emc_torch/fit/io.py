import typing

import nibabel as nib
import pathlib as plib
import logging

import numpy as np
import torch
from emc_torch import DB
from .options import FitConfig
log_module = logging.getLogger(__name__)


def check_name(name: str):
    if not name:
        err = "no filename given to save file to"
        log_module.error(err)
        raise AttributeError(err)


def load_data(fit_config: FitConfig) -> (torch.tensor, DB, torch.tensor, torch.tensor, torch.tensor):
    """
    load in data files from configuration

    returns: nii data, database, b1 data, nii affine, b1 affine
    """
    # load in data
    nii_data, nii_affine = load_nii_data(fit_config.nii_path)
    nii_input_shape = nii_data.shape
    # dont normalize for extracting noise!

    # b1
    if fit_config.b1_file:
        b1_nii, b1_affine = load_nii_data(fit_config.b1_file)
    else:
        b1_nii = None
        b1_affine = None

    # database
    db = load_database(fit_config.database_file)
    return nii_data, db, b1_nii, nii_affine, b1_affine


def load_nii_data(file_path_nii: str | plib.Path) -> (torch.tensor, torch.tensor):
    # ensure path is plib.Path, in case of str input
    file_path_nii = plib.Path(file_path_nii).absolute()
    # check if file
    if not file_path_nii.is_file():
        err = f"File : {file_path_nii.as_posix()} not found!"
        log_module.error(err)
        raise FileNotFoundError(err)
    if ".nii" not in file_path_nii.suffixes:
        err = f"File : {file_path_nii.as_posix()} not a .nii file."
        log_module.error(err)
        raise AttributeError(err)
    # load
    log_module.info(f"Loading Nii File: {file_path_nii.as_posix()}")
    nii_img = nib.load(file_path_nii.as_posix())
    nii_data = torch.from_numpy(nii_img.get_fdata())
    nii_affine = torch.from_numpy(nii_img.affine)
    return nii_data, nii_affine


def load_database(file_path_db: str | plib.Path) -> DB:
    # ensure path is plib.Path, in case of str input
    file_path_db = plib.Path(file_path_db).absolute()
    # check if file
    if not file_path_db.is_file():
        err = f"File : {file_path_db.as_posix()} not found!"
        log_module.error(err)
        raise FileNotFoundError(err)
    # load
    log_module.info(f"Loading DB File: {file_path_db.as_posix()}")
    db = DB.load(file_path_db.as_posix())
    return db


def save_nii(data: typing.Union[nib.Nifti1Image, np.ndarray, torch.tensor], file_path: str | plib.Path, name: str,
             affine: np.ndarray = None):
    check_name(name=name)
    file_path = file_path.joinpath(name).with_suffix(".nii")
    if torch.is_tensor(data):
        data = data.numpy(force=True)
    if isinstance(data, np.ndarray):
        if affine is None:
            err = "Provide Affine to save .nii"
            log_module.error(err)
            raise AttributeError(err)
        img = nib.Nifti1Image(data, affine)
    else:
        img = data
    log_module.info(f"Writing File: {file_path.as_posix()}")
    nib.save(img, file_path.as_posix())

