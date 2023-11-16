from . import plotting, options
from .db_class import DB

__all__ = ["plotting", "options", "DB"]

# make sure all sources are seen -> submodule
import sys
import pathlib
path_content_root = pathlib.Path(__file__).absolute().parent.parent
pulse_path = path_content_root.joinpath("pypulseq_interface/")
dmri_path = path_content_root.joinpath("autodmri")

sys.path.append(pulse_path.as_posix())
