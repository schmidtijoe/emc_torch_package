"""
We want a class structure that provides all of the necessary information for:
- info of scanner specs
- details used for sequence creation
- details needed for raw data processing
- details needed for sequence simulation via EMC
- info needed to calculate and / or store k-space trajectories -> gridding via kbnufft
- saving and loading sampling patterns
- interfacing pulse files - store and plot pulses used or feed in pulse shapes
"""
import logging
import pathlib as plib
import typing
import numpy as np
import pandas as pd
import simple_parsing as sp
import simple_parsing.helpers.serialization as sphs
import dataclasses as dc
from pypsi import parameters

log_module = logging.getLogger(__name__)


@dc.dataclass
class Config(sp.helpers.Serializable):
    config_file: str = sp.field(default="", alias=["-c"])
    output_path: str = sp.field(default="./test/", alias=["-o"])
    visualize: bool = sp.field(default=True, alias=["-v"])


@dc.dataclass
class XConfig(sp.helpers.Serializable):
    # loading extra files
    pypulseq_config_file: str = sp.field(default=None, alias="-ppf")
    pulse_file: str = sp.field(default=None, alias="-pf")
    sampling_k_traj_file: str = sp.field(default=None, alias="-skf")
    emc_info_file: str = sp.field(default=None, alias="-emcf")
    raw_data_details_file: str = sp.field(default=None, alias="-rddf")
    scanner_specs_file: str = sp.field(default=None, alias="-ssf")


@dc.dataclass
class Params(sp.helpers.Serializable):
    config: Config = Config()
    emc: parameters.EmcParameters = parameters.EmcParameters()
    pypulseq: parameters.PypulseqParameters = parameters.PypulseqParameters()
    pulse: parameters.RFParameters = parameters.RFParameters()
    sampling_k_traj: parameters.SamplingKTrajectoryParameters = parameters.SamplingKTrajectoryParameters()
    recon: parameters.ReconParameters = parameters.ReconParameters()
    specs: parameters.ScannerParameters = parameters.ScannerParameters()

    def __post_init__(self):
        self._d_to_set: dict = {
            "pypulseq_config_file": "pypulseq",
            "pulse_file": "pulse",
            "sampling_k_traj_file": "sampling_k_traj",
            "emc_info_file": "emc",
            "raw_data_details_file": "recon",
            "scanner_specs_file": "specs",
        }

    def display_sequence_configuration(self):
        # build dataframe for visualization of most important data
        names = [
            "",
            "__ sequence configuration__ ",
            "name", "version", "report",
            "resolution_fov_read", "resolution_fov_phase",
            "resolution_base", "resolution_slice_thickness", "resolution_slice_num", "resolution_slice_gap",
            "resolution_voxel_size_read", "resolution_voxel_size_phase",
            "acceleration_factor", "etl", "tr", "bandwidth", "acq_phase_dir",
            "__ system specifications __",
            "b_0", "max_grad", "max_slew"
        ]
        units = [
            "unit",
            "",
            "", "", "",
            "mm", "%",
            "", "mm", "", "%",
            "mm", "mm",
            "", "", "ms", "Hz/px", "",
            "",
            "T", "mT/m", "T/m/s"
        ]
        attr_pypulseq = self.pypulseq.to_dict()
        attr_specs = self.specs.to_dict()
        vals = [""] * len(names)
        vals[0] = "value"
        for n_idx in range(len(names)):
            name = names[n_idx]
            if name in attr_pypulseq.keys():
                vals[n_idx] = attr_pypulseq[name]
            if name in attr_specs.keys():
                vals[n_idx] = attr_specs[name]
            # add voxel size, not in dict from post init
            if name == "resolution_voxel_size_read" or name == "resolution_voxel_size_phase":
                vals[n_idx] = f"{self.pypulseq.__getattribute__(name):.3f}"

        d = {
            "names": names, "vals": vals, "units": units
        }
        # set index column blank
        idx = [""] * len(names)
        df = pd.DataFrame(d, index=idx)
        # set index row blank
        df.columns = ["", "", ""]
        # display via logging
        log_module.info(df)

    def save_as_subclasses(self, path: typing.Union[str, plib.Path]):
        # ensure path
        path = plib.Path(path).absolute()
        # check if exists or make
        if path.suffixes:
            path = path.parent
        path.mkdir(parents=True, exist_ok=True)
        # save
        for f_name, att_name in self._d_to_set.items():
            suffix = ".json"
            if att_name == "pulse" or att_name == "sampling_k_traj":
                suffix = ".pkl"
            save_file = path.joinpath(f_name).with_suffix(suffix)
            log_module.info(f"write file: {save_file.as_posix()}")
            subclass = self.__getattribute__(att_name)
            if suffix == ".pkl":
                subclass.save(save_file)
            else:
                subclass.save_json(save_file, indent=2)

    @classmethod
    def from_cli(cls, args: sp.ArgumentParser.parse_args):
        # create instance, fill config arguments
        instance = cls(config=args.config)
        # check if config file exists and laod
        c_file = plib.Path(args.config.config_file).absolute()
        if c_file.is_file():
            log_module.info(f"loading config file: {c_file.as_posix()}")
            instance = cls.load(c_file)
        # check for extra file input
        instance._load_extra_argfile(extra_files=args.extra_files)
        return instance

    def visualize(self):
        self.sampling_k_traj.plot_sampling_pattern(output_path=self.config.output_path)
        self.sampling_k_traj.plot_k_space_trajectories(output_path=self.config.output_path)
        self.pulse.plot(output_path=self.config.output_path)

    def _load_extra_argfile(self, extra_files: XConfig):
        # check through all arguments
        for mem, f_name in extra_files.__dict__.items():
            # check if provided
            if f_name is not None:
                # get member name of Config class and set file name value
                # convert to plib Path
                path = plib.Path(f_name)
                if path.is_file():
                    log_module.info(f"load file: {path.as_posix()}")
                    # get corresponding member
                    mem_name = self._d_to_set.get(mem)
                    if mem_name is not None:
                        mem = extra_files.__getattribute__(mem_name)
                        mem.load(path)
                        self.__setattr__(mem_name, mem)
                elif not path.is_file():
                    err = f"{path} is not a file. exiting..."
                    log_module.error(err)
                    raise FileNotFoundError(err)


# set serializable encoders
@sphs.encode.register
def encode_ndarray(obj: np.ndarray):
    """ encode np ndarray as lists """
    return obj.tolist()


@sphs.encode.register
def encode_pandas_dataframe(obj: pd.DataFrame):
    """ encode pandas dataframe as dict """
    return obj.to_dict()


# set serializable decoders
sphs.register_decoding_fn(np.ndarray, np.array)
sphs.register_decoding_fn(pd.DataFrame, pd.DataFrame.from_dict)


def create_cli() -> (sp.ArgumentParser, sp.ArgumentParser.parse_args):
    parser = sp.ArgumentParser(prog="pypsi")
    parser.add_arguments(Config, dest="config")
    parser.add_arguments(XConfig, dest="extra_files")
    args = parser.parse_args()
    return parser, args


if __name__ == '__main__':
    parser, args = create_cli()

    params = Params.from_cli(args=args)
    s_file = plib.Path("./default_config/pypsi.pkl").absolute()
    s_file.parent.mkdir(parents=True, exist_ok=True)
    params.save(s_file.as_posix())

    params.save_as_subclasses(s_file.parent.as_posix())

    params = Params.load(s_file.as_posix())
    log_module.info("success")
