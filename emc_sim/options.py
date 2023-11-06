"""
Implementation of cmd line options and configuration / runtime classes
"""
from dataclasses import dataclass
import simple_parsing as sp
import logging
import torch
import typing
import pathlib as plib
import pandas as pd
from scipy import stats
import pypsi

log_module = logging.getLogger(__name__)


@dataclass
class SimulationConfig(sp.Serializable):
    """
        Configuration for simulation
        """
    # provide Configuration file (.json)
    config_file: str = sp.field(alias=["-c"], default="")
    # provide separate sequence params
    emc_seq_config: str = sp.field(alias=["-esc"], default="")
    # set path to save database and used config
    save_path: str = sp.field(alias=["-s"], default="")
    # set filename of database
    database_name: str = sp.field(alias=["-db"], default="database_test.pkl")
    # set filepath to interface
    pypsi_path: str = sp.field(alias=["-p"], default="./tests/pypsi_mese_test.pkl")
    # set simulation type
    sim_type: str = sp.field(alias="-t", default="mese", choices=["mese", "megesse", "fid", "single"])

    # set signal echo processing -> this enables sampling the signal over the 1d slice dimension
    # substituting the readout and using identical readout time etc.
    # when turned off the spin contributions are summed across the profile
    signal_fourier_sampling: bool = sp.field(alias="-sfs", default=True)

    # set flag to visualize pulse profiles and sequence scheme
    visualize: bool = True
    # toggle debugging log
    debug_flag: bool = True

    # toggle multithreading
    # multiprocessing: bool = False
    # give number of CPUs to leave unused when multiprocessing
    # mpHeadroom: int = 16
    # give desired number of CPUs to use
    # mpNumCpus: int = 1

    def __post_init__(self):
        # if self.multiprocessing:
        #     # we take at most the maximum free cpus but leave some headroom for other users
        #     self.mpNumCpus = mp.cpu_count() - self.mpHeadroom
        #     self.mpNumCpus = np.max([mp.cpu_count() - self.mpHeadroom, 4])
        #     # we take at least 4 cpus (kind of catches misconfiguration of the headroom parameter)
        # else:
        #     self.mpNumCpus = 1
        # self.mpNumCpus = int(self.mpNumCpus)
        pass

    def display(self):
        # display via logging
        df = pd.Series(self.to_dict())
        # concat empty entry to start of series for nicer visualization
        df = pd.concat([pd.Series([""], index=["___ Config ___"]), df])
        # display
        log_module.info(df)


@dataclass
class SimulationSettings(sp.Serializable):
    """
        Optional settings for simulation eg. spatial resolution
        """
    sample_number: int = 1000  # no of sampling points along slice profile
    length_z: float = 0.005  # [m] length extension of z-axis spanned by sample -> total length 2*lengthZ (-:+)
    acquisition_number: int = 50  # number of bins across slice sample -> effectively sets spatial resolution
    # resolution = 2 * lengthZ / acquisitionNumber

    t1_list: typing.List = sp.field(default_factory=lambda: [1.5])  # T1 to simulate [s]
    t2_list: typing.List = sp.field(default_factory=lambda: [[25, 30, 0.5], [30, 35, 1]])  # T2 to simulate [ms]
    b1_list: typing.List = sp.field(default_factory=lambda: [0.6, 1.0])  # B1 to simulate
    # diffusion values to use if flag in config is set [mm²/s]
    total_num_sim: int = 4

    def get_complete_param_list(self):
        return [(t1, t2, b1) for t1 in self.t1_list
                for t2 in self.t2_list for b1 in self.b1_list]

    def get_slice_profile_bin_resolution(self):
        return self.length_z * 2 / self.acquisition_number

    def get_slice_profile_sample_resolution(self):
        return self.length_z * 2 / self.sample_number


@dataclass
class SimulationParameters(sp.Serializable):
    config: SimulationConfig = SimulationConfig()
    sequence: pypsi.parameters.EmcParameters = pypsi.parameters.EmcParameters()
    settings: SimulationSettings = SimulationSettings()

    def __post_init__(self):
        log_module.debug(
            f"spatial sampling resolution: {self.settings.get_slice_profile_sample_resolution() * 1e6:.2f} um"
            f"(per spin isochromat)"
        )
        log_module.debug(
            f"spatial readout binning resolution across slice profile:"
            f"{self.settings.get_slice_profile_bin_resolution() * 1e6:.2f} um (contributions to readout)"
        )
        # gradient area = deltaK * n = 1/FOV * num_acquisition
        grad_area = 1 / (2 * self.settings.length_z) * self.settings.acquisition_number
        # grad_area in 1/m -> / gamma for T/m
        grad_amp = grad_area / self.sequence.gamma_hz / self.sequence.duration_acquisition * 1e6  # cast to s
        self.sequence.gradient_acquisition = - grad_amp * 1e3  # cast to mT

    def set_acquisition_gradient(self):
        self.__post_init__()

    @classmethod
    def from_cli(cls, args: sp.ArgumentParser.parse_args):
        sim_params = SimulationParameters(config=args.config, settings=args.settings, sequence=args.sequence)

        non_default_config, non_default_settings, non_default_sequence = sim_params._check_non_default_vars()

        if args.config.config_file:
            sim_params = SimulationParameters.load(args.config.config_file)
            # overwrite non default input args
            for key, value in non_default_config.items():
                sim_params.config.__setattr__(key, value)
            for key, value in non_default_settings.items():
                sim_params.settings.__setattr__(key, value)
            for key, value in non_default_sequence.items():
                sim_params.sequence.__setattr__(key, value)

        # we check parsed arguments for explicit cmd line input assuming explicit input means "different from default".
        # Since everytime the cmd is parsed all values not given explicitly are parsed with respective
        # dataclass defaults.
        # Hence, if explicit input is coincidentally a default value this value will be ignored:
        # eg. to overwrite an instance (containing non-default values) loaded by a configFile
        # and explicitly trying to change entries to default via cmd input.
        # ToDo: Fix explicit cmd line input
        if args.config.emc_seq_config or sim_params.config.emc_seq_config:
            emc_seq_config = sim_params.config.emc_seq_config
            if args.config.emc_seq_config:
                emc_seq_config = args.config.emc_seq_config
            sim_params.sequence = pypsi.parameters.EmcParameters.load(emc_seq_config)
        # choose explicit emc config file over pypsi interface file
        if (args.config.pypsi_path or sim_params.config.pypsi_path) and not sim_params.config.emc_seq_config:
            pyp_path = sim_params.config.pypsi_path
            if args.config.pypsi_path:
                pyp_path = args.config.pypsi_path
            pyp_interface = pypsi.Params.load(pyp_path)
            sim_params.sequence = pyp_interface.emc

        # sanity check pyp
        if not sim_params.config.emc_seq_config:
            if sim_params.config.pypsi_path:
                o_path = plib.Path(sim_params.config.pypsi_path).absolute()
            else:
                err = f"neither direct emc config nor pypulseq interface file provided."
                log_module.error(err)
                raise FileNotFoundError(err)
        else:
            o_path = plib.Path(sim_params.config.emc_seq_config).absolute()
            # if we have no output path set, use input of pypsi
        # if we have no output path set, use input of pypsi or emc
        if not sim_params.config.save_path:
            sim_params.config.save_path = o_path.parent.as_posix().__str__()
        sim_params.set_acquisition_gradient()
        return sim_params

    def _check_non_default_vars(self) -> (dict, dict, dict):
        def_config = SimulationConfig()
        non_default_config = {}
        for key, value in vars(self.config).items():
            if self.config.__getattribute__(key) != def_config.__getattribute__(key):
                non_default_config.__setitem__(key, value)

        def_settings = SimulationSettings()
        non_default_settings = {}
        for key, value in vars(self.settings).items():
            if torch.is_tensor(value):
                continue
            if self.settings.__getattribute__(key) != def_settings.__getattribute__(key):
                non_default_settings.__setitem__(key, value)

        def_sequence = pypsi.parameters.EmcParameters()
        non_default_sequence = {}
        for key, value in vars(self.sequence).items():
            # catch post init attribute
            if key == 'gradientAcquisition':
                continue
            if self.sequence.__getattribute__(key) != def_sequence.__getattribute__(key):
                non_default_sequence.__setitem__(key, value)
        return non_default_config, non_default_settings, non_default_sequence

    def save_database(self, database: pd.DataFrame) -> None:
        base_path = plib.Path(self.config.save_path).absolute()
        # create parent folder ifn existent
        plib.Path.mkdir(base_path, exist_ok=True, parents=True)

        db_path = base_path.joinpath(self.config.database_name)
        config_path = base_path.joinpath(f"{db_path.stem}_config.json")

        log_module.info(f"writing file {db_path}")
        # mode dependent on file ending given
        save_fn = {
            ".pkl": database.to_pickle,
            ".json": lambda obj: database.to_json(obj, indent=2)
        }
        assert save_fn.get(db_path.suffix), f"Database save path{db_path}: type not recognized;" \
                                            f"Supported: {list(save_fn.keys())}"
        save_fn.get(db_path.suffix)(db_path.__str__())
        # save used config
        log_module.info(f"writing file {config_path}")
        self.save_json(config_path, indent=2, separators=(',', ':'))


@dataclass
class SimulationData:
    """ carrying data through simulation """
    t1_vals: torch.tensor
    t2_vals: torch.tensor
    b1_vals: torch.tensor

    emc_signal_mag: torch.tensor
    emc_signal_phase: torch.tensor

    sample_axis: torch.tensor
    signal_tensor: torch.tensor
    magnetization_propagation: torch.tensor

    gamma: torch.tensor

    device: torch.device

    @classmethod
    def from_sim_parameters(cls, sim_params: SimulationParameters, device: torch.device):
        # set values with some error catches
        # t1
        if isinstance(sim_params.settings.t1_list, list):
            t1_vals = torch.tensor(sim_params.settings.t1_list, device=device)
        else:
            t1_vals = torch.tensor([sim_params.settings.t1_list], dtype=torch.float32, device=device)
        # b1
        if isinstance(sim_params.settings.b1_list, list):
            b1_vals = torch.tensor(sim_params.settings.b1_list, device=device)
        else:
            b1_vals = torch.tensor([sim_params.settings.b1_list], dtype=torch.float32, device=device)
        # t2
        array = []
        if isinstance(sim_params.settings.t2_list, list):
            for item in sim_params.settings.t2_list:
                if isinstance(item, str):
                    item = [float(i) for i in item[1:-1].split(',')]
                if isinstance(item, int):
                    item = float(item)
                if isinstance(item, float):
                    array.append(item)
                else:
                    array.extend(torch.arange(*item).tolist())
        else:
            array = [sim_params.settings.t2_list]
        array = torch.tensor(array, dtype=torch.float32)
        array /= 1000.0  # cast to s
        t2_vals = array.to(device)

        sample_axis = torch.linspace(-sim_params.settings.length_z, sim_params.settings.length_z,
                                     sim_params.settings.sample_number)
        sample = torch.from_numpy(
            stats.gennorm(24).pdf(sample_axis / sim_params.settings.length_z * 1.1) + 1e-6
        )
        sample = torch.divide(sample, torch.max(sample))

        sample_axis = sample_axis.to(device)

        m_init = torch.zeros((sim_params.settings.sample_number, 4))
        m_init[:, 2] = sample
        m_init[:, 3] = sample
        m_init = m_init[None, None, None].to(device)
        # signal tensor is supposed to hold all acquisition points for all reads
        signal_tensor = torch.zeros((
            t1_vals.shape[0], t2_vals.shape[0], b1_vals.shape[0],
            sim_params.sequence.etl, sim_params.settings.acquisition_number),
            dtype=torch.complex128, device=device)

        # allocate
        # set emc data tensor -> dims: [t1s, t2s, b1s, ETL]
        # (we get this in the end by calculation, no need to allocate for it and carry it through)
        emc_signal_mag = torch.zeros(
            (t1_vals.shape[0], t2_vals.shape[0], b1_vals.shape[0],
             sim_params.sequence.etl),
            device=device
        )
        emc_signal_phase = torch.zeros(
            (t1_vals.shape[0], t2_vals.shape[0], b1_vals.shape[0],
             sim_params.sequence.etl),
            device=device)
        instance = cls(
            t1_vals=t1_vals, t2_vals=t2_vals, b1_vals=b1_vals,
            emc_signal_mag=emc_signal_mag, emc_signal_phase=emc_signal_phase,
            sample_axis=sample_axis, signal_tensor=signal_tensor, magnetization_propagation=m_init,
            gamma=torch.tensor(sim_params.sequence.gamma_hz, device=device), device=device
        )
        instance._check_args()
        return instance

    def _check_args(self):
        # sanity checks
        if torch.max(self.t2_vals) > torch.min(self.t1_vals):
            err = 'T1 T2 mismatch (T2 > T1)'
            log_module.error(err)
            raise AttributeError(err)
        if torch.max(self.t2_vals) < 1e-4:
            err = 'T2 value range exceeded, make sure to post T2 in ms'
            log_module.error(err)
            raise AttributeError(err)

    def set_device(self, device: torch.device):
        for _, value in vars(self).items():
            if torch.is_tensor(value):
                value.to(device)


def create_cli():
    """
    Build the parser for arguments
    Parse the input arguments.
    """
    parser = sp.ArgumentParser(prog='emc_torch_sim')
    parser.add_arguments(SimulationConfig, dest="config")
    parser.add_arguments(SimulationSettings, dest="settings")
    parser.add_arguments(pypsi.parameters.EmcParameters, dest="sequence")

    args = parser.parse_args()

    return parser, args


if __name__ == '__main__':
    params = SimulationParameters()
    path = plib.Path("tests/default.json").absolute()
    params.save_json(path.as_posix(), indent=2, separators=(',', ':'))
