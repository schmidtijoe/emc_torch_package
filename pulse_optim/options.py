"""
pulse optimization specific options
mainly for parameter sweeps using wandb

"""
import pathlib as plib
import dataclasses as dc
from emc_sim import options as eso
import simple_parsing as sp
import typing


@dc.dataclass
class ConfigOptimization(sp.Serializable):
    optim_config_file: typing.Union[str, plib.Path] = sp.field(
        alias=["-oc"],
        default="/data/pt_np-jschmidt/code/emc_torch/optim/config.json"
    )
    optim_save_path: typing.Union[str, plib.Path] = sp.field(alias=["-os"], default="./optim")
    run: int = sp.field(alias=["-onr"], default=0)
    lr: float = sp.field(alias=["-olr"], default=0.1)
    momentum: float = sp.field(alias=["-om"], default=0.5)
    random_seed: int = sp.field(alias=["-ors"], default=0)
    num_steps: int = sp.field(alias=["-ons"], default=100)
    base_cos_scale: float = sp.field(alias=["-bcs"], default=0.5)
    init_type: int = 1      # 0 rnd, and then cos # lobes

    def set_name(self):
        self.optim_save_path = plib.Path(self.optim_save_path).absolute()
        self.optim_save_path = self.optim_save_path.joinpath(f"run-{self.run}")
        optim_name = f"lr-{str(self.lr).replace('.', 'p')}_mom-{str(self.momentum).replace('.', 'p')}" \
                     f"_rng-seed-{self.random_seed}"
        plib.Path.mkdir(self.optim_save_path, exist_ok=True, parents=True)
        self.optim_save_path = self.optim_save_path.joinpath(optim_name)

    @classmethod
    def from_cmd_line_args(cls, args: sp.ArgumentParser.parse_args):
        config_args = args.optimization
        c_path = plib.Path(config_args.optim_config_file).absolute()
        if c_path.is_file():
            instance = cls.load(c_path.as_posix())
            instance._check_non_default_vars(config_args)
        else:
            instance = cls()
            instance.optim_save_path = config_args.optim_save_path
            instance.lr = config_args.lr
            instance.random_seed = config_args.random_seed
            instance.num_steps = config_args.num_steps
        return instance

    def _check_non_default_vars(self, config_args: sp.ArgumentParser.parse_args):
        def_config = ConfigOptimization()
        non_default_config = {}
        for key, value in vars(config_args).items():
            if config_args.__getattribute__(key) != def_config.__getattribute__(key):
                non_default_config.__setitem__(key, value)
                self.__setattr__(key, value)
        # return non_default_config


def create_cmd_line_interface() -> (sp.ArgumentParser, sp.ArgumentParser.parse_args):
    parser = sp.ArgumentParser(prog='emc_torch_pulse_optimization')
    parser.add_arguments(eso.SimulationConfig, dest="config")
    parser.add_arguments(eso.SimulationSettings, dest="settings")
    parser.add_arguments(eso.SequenceConfiguration, dest="sequence")
    parser.add_arguments(ConfigOptimization, dest="optimization")

    args = parser.parse_args()

    return parser, args


if __name__ == '__main__':
    inst = ConfigOptimization()
    inst.save_json(inst.optim_save_path.with_name("config").with_suffix(".json"), indent=2)
