import simple_parsing as sp
import dataclasses as dc
import pandas as pd
import logging
import pathlib as plib
log_module = logging.getLogger(__name__)


@dc.dataclass
class FitConfig(sp.Serializable):
    """
        Configuration for Fitting
        """
    # provide Configuration file (.json)
    config_file: str = sp.field(alias="-c", default="")
    # set filepath to nii file
    nii_path: str = sp.field(alias="-i", default="")
    # set path to save database and used config
    save_path: str = sp.field(alias="-s", default="")
    # set name of file
    save_name_prefix: str = sp.field(alias="-o", default="")
    # set filename of database
    database_file: str = sp.field(alias="-db", default="")
    # set filename of b1 map (optional)
    b1_file: str = sp.field(alias="-b1", default="")
    # set weighting of b1 (optional), sets the weight of the b1 penalty for the db entries to match input b1
    b1_weight: float = sp.field(alias="-b1w", default=0.5)
    # set transmit field offset if used in scan (optional). adjustment of the transmit voltage in semc
    # would scale the B1 map acquired with other modalities
    b1_tx_scale: float = sp.field(alias="-b1tx", default=1.0)
    # use gpu
    use_gpu: bool = sp.field(alias="-gpu", default=True)
    # gpu device
    gpu_device: int = sp.field(alias="-gpud", default=0)

    # flags
    # visualization
    visualize: bool = sp.field(alias="-v", default=True)
    # debug
    debug: bool = sp.field(alias="-d", default=False)

    @classmethod
    def from_cli(cls, args: sp.ArgumentParser.parse_args):
        instance = args.config
        if instance.config_file:
            c_path = plib.Path(instance.config_file).absolute()
            if not c_path.is_file():
                err = f"Config File set: {c_path.as_posix()} not found!"
                log_module.error(err)
                raise FileNotFoundError(err)
            instance = cls.load(c_path.as_posix())
        instance.display()
        return instance

    def display(self):
        # display via logging
        df = pd.Series(self.to_dict())
        # concat empty entry to start of series for nicer visualization
        df = pd.concat([pd.Series([""], index=["___ Config ___"]), df])
        # display
        log_module.info(df)


def create_cli() -> (sp.ArgumentParser, sp.ArgumentParser.parse_args):
    """
        Build the parser for arguments
        Parse the input arguments.
        """
    parser = sp.ArgumentParser(prog='emc_torch_fit')
    parser.add_arguments(FitConfig, dest="config")
    args = parser.parse_args()
    return parser, args
