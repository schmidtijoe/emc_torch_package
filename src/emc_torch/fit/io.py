import nibabel as nib
import pathlib as plib
import logging
import torch
from emc_torch import DB

log_module = logging.getLogger(__name__)


def load_data(file_path_nii: str | plib.Path) -> (torch.tensor, torch.tensor):
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
