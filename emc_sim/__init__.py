# make sure all sources are seen -> submodule
import sys
import pathlib
pulse_path = pathlib.Path(__file__).absolute().parent.parent.joinpath("pypulseq_interface/")
sys.path.append(pulse_path.as_posix())

