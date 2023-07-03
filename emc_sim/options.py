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

logModule = logging.getLogger(__name__)


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
    save_path: str = sp.field(alias=["-s"], default="./data")
    # set filename of database
    database_name: str = sp.field(alias=["-db"], default="database_test.pkl")
    # set filepath to external pulse-files (pkl or json)
    path_to_rfpf: str = sp.field(alias=["-rfpf"], default="./external")

    # name of external pulse file for excitation - assumed to be rf_pulse_files compatible.
    # See rf_pulse_files to convert from .txt or .pta
    rfpf_excitation: str = sp.field(alias=["-rfpf_e"], default="")
    # name of external pulse file for refocussing - assumed to be rf_pulse_files compatible.
    # See rf_pulse_files to convert from .txt or .pta
    rfpf_refocus: str = sp.field(alias=["-rfpf_r"], default="")

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


@dataclass
class SimulationData(sp.Serializable):
    emc_signal: torch.tensor = torch.zeros(0)
    t1_s: torch.tensor = torch.tensor(1.5)
    t2_s: torch.tensor = torch.tensor(0.035)
    b1: torch.tensor = torch.tensor(1.0)
    time: float = 0.0

    def set_run_params(
            self, t1_s: typing.Union[float, torch.tensor],
            t2_s: typing.Union[float, torch.tensor],
            b1: typing.Union[float, torch.tensor]):
        self.t1_s = torch.as_tensor(t1_s)
        self.t2_s = torch.as_tensor(t2_s)
        self.b1 = torch.as_tensor(b1)

    def get_run_params(self):
        ret = {
            "T1": self.t1_s,
            "T2": self.t2_s,
            "B1": self.b1
        }
        return ret

    @classmethod
    def set_signal_array_length(cls, etl: int):
        sim_data_instance = cls()
        sim_data_instance.emc_signal = torch.zeros(etl)
        return sim_data_instance


@dataclass
class SequenceConfiguration(sp.Serializable):
    """
        Parameters related to Sequence simulation
        """
    # global parameter gamma [Hz/t]
    gammaHz: float = 42577478.518

    # echo train length
    ETL: int = 16
    # echo spacing [ms]
    ESP: float = 9.0
    # bandwidth [Hz/px]
    bw: float = 349
    # gradient mode

    # Excitation, Flip Angle [°]
    excitation_angle: float = 90.0
    # Excitation, Phase [°]
    excitation_phase: float = 90.0
    # Excitation, gradient if rectangular/trapezoid [mt/m]
    gradient_excitation: float = -18.5
    # Excitation, duration of pulse [us]
    duration_excitation: float = 2560.0

    gradient_excitation_rephase: float = -10.51  # [mT/m], rephase
    duration_excitation_rephase: float = 1080.0  # [us], rephase

    # Refocussing, Flip Angle [°]
    refocus_angle: typing.List = sp.field(default_factory=lambda: [140.0])
    # Refocussing, Phase [°]
    refocus_phase: typing.List = sp.field(default_factory=lambda: [0.0])
    # Refocussing, gradient strength if rectangular/trapezoid [mt/m]
    gradient_refocus: float = -36.2
    # Refocussing, duration of pulse [us]
    duration_refocus: float = 3584.0

    gradient_crush: float = -38.7  # [mT/m], crusher
    duration_crush: float = 1000.0  # [us], crushe
    gradient_acquisition: float = 0.0  # set automatically after settings init

    # time for acquisition (of one pixel) * 1e6 <- [(px)s] * 1e6

    def __post_init__(self):
        self.gamma_pi: float = self.gammaHz * 2 * torch.pi
        self.duration_acquisition: float = 1e6 / self.bw  # [us]
        if self.refocus_phase.__len__() != self.refocus_angle.__len__():
            err = f"provide same amount of refocusing pulse angle ({self.refocus_angle.__len__()}) " \
                  f"and phases ({self.refocus_phase.__len__()})"
            logModule.error(err)
            raise AttributeError(err)
        # check for phase values
        for l_idx in range(self.refocus_phase.__len__()):
            while abs(self.refocus_phase[l_idx]) > 180.0:
                self.refocus_phase[l_idx] = self.refocus_phase[l_idx] - torch.sign(self.refocus_phase[l_idx]) * 180.0
            while abs(self.refocus_angle[l_idx]) > 180.0:
                self.refocus_angle[l_idx] = self.refocus_angle[l_idx] - torch.sign(self.refocus_angle[l_idx]) * 180.0
        while self.refocus_angle.__len__() < self.ETL:
            # fill up list with last value
            self.refocus_angle.append(self.refocus_angle[-1])
            self.refocus_phase.append(self.refocus_phase[-1])


@dataclass
class SimulationSettings(sp.Serializable):
    """
        Optional settings for simulation eg. spatial resolution
        """
    sample_number: int = 1000  # no of sampling points along slice profile
    length_z: float = 0.005  # [m] length extension of z-axis spanned by sample -> total length 2*lengthZ (-:+)
    acquisition_number: int = 50  # number of bins across slice sample -> effectively sets spatial resolution
    # resolution = lengthZ / acquisitionNumber

    t1_list: typing.List = sp.field(default_factory=lambda: [1.5])  # T1 to simulate [s]
    t2_list: typing.List = sp.field(default_factory=lambda: [[25, 30, 0.5], [30, 35, 1]])  # T2 to simulate [ms]
    b1_list: typing.List = sp.field(default_factory=lambda: [0.6, 1.0])  # B1 to simulate
    d_list: typing.List = sp.field(default_factory=lambda: [700.0])
    # diffusion values to use if flag in config is set [mm²/s]
    total_num_sim: int = 4

    def __post_init__(self):
        array = torch.empty(0)
        for item in self.t2_list:
            if type(item) == str:
                item = [float(i) for i in item[1:-1].split(',')]
            array = torch.concatenate((array, torch.arange(*item)))

        array = [t2 / 1000.0 for t2 in array]  # cast to [s]
        self.t2_array = array
        # sanity checks
        if max(self.t2_array) > min(self.t1_list):
            logModule.error('T1 T2 mismatch (T2 > T1)')
            exit(-1)
        if max(self.t2_array) < 1e-4:
            logModule.error('T2 value range exceeded, make sure to post T2 in ms')
            exit(-1)
        else:
            self.total_num_sim = len(self.t1_list) * len(self.t2_array) * len(self.b1_list) * len(self.d_list)

    def get_complete_param_list(self):
        return [(t1, t2, b1, d) for t1 in self.t1_list
                for t2 in self.t2_array for b1 in self.b1_list for d in self.d_list]


@dataclass
class SimulationParameters(sp.Serializable):
    config: SimulationConfig = SimulationConfig()
    sequence: SequenceConfiguration = SequenceConfiguration()
    settings: SimulationSettings = SimulationSettings()

    def __post_init__(self):
        self.sequence.gradient_acquisition = - self.settings.acquisition_number * self.sequence.bw \
                                             / (self.sequence.gammaHz * 2 * self.settings.length_z) * 1000

    def set_acquisition_gradient(self):
        self.__post_init__()

    @classmethod
    def from_cmd_args(cls, args: sp.ArgumentParser.parse_args):
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
            sim_params.sequence = SequenceConfiguration.load(emc_seq_config)
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
            if self.settings.__getattribute__(key) != def_settings.__getattribute__(key):
                non_default_settings.__setitem__(key, value)

        def_sequence = SequenceConfiguration()
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

        logModule.info(f"writing file {db_path}")
        # mode dependent on file ending given
        save_fn = {
            ".pkl": database.to_pickle,
            ".json": lambda obj: database.to_json(obj, indent=2)
        }
        assert save_fn.get(db_path.suffix), f"Database save path{db_path}: type not recognized;" \
                                            f"Supported: {list(save_fn.keys())}"
        save_fn.get(db_path.suffix)(db_path.__str__())
        # save used config
        logModule.info(f"writing file {config_path}")
        self.save(config_path, indent=2, separators=(',', ':'))


@dataclass
class SimTempData:
    """
        Carrying data through simulation
        """
    sample: torch.tensor
    sample_axis: torch.tensor
    signal_tensor: torch.tensor
    magnetizationPropagation: torch.tensor
    excitation_flag: bool = True  # flag to toggle between excitation and refocus
    run: SimulationData = SimulationData()

    def __init__(self, simParams: SimulationParameters):
        self.sample_axis = torch.linspace(-simParams.settings.length_z, simParams.settings.length_z,
                                          simParams.settings.sample_number)
        sample = torch.from_numpy(stats.gennorm(24).pdf(self.sample_axis / simParams.settings.length_z * 1.1) + 1e-6)
        self.sample = torch.divide(sample, torch.max(sample))
        mInit = torch.zeros((simParams.settings.sample_number, 4))
        mInit[:, 2] = self.sample
        mInit[:, 3] = self.sample
        self.signal_tensor = torch.zeros((simParams.sequence.ETL, simParams.settings.acquisition_number), dtype=complex)
        self.magnetizationPropagation = mInit


def createCommandlineParser():
    """
    Build the parser for arguments
    Parse the input arguments.
    """
    parser = sp.ArgumentParser(prog='emc_torch_sim')
    parser.add_arguments(SimulationConfig, dest="config")
    parser.add_arguments(SimulationSettings, dest="settings")
    parser.add_arguments(SequenceConfiguration, dest="sequence")

    args = parser.parse_args()

    return parser, args

