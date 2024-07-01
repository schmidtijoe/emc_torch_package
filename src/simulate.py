from emc_torch import simulations, options, db_class
import logging
import pathlib as plib
log_module = logging.getLogger(__name__)
logging.getLogger('simple_parsing').setLevel(logging.WARNING)


def sim(
        emc_seq_file: str = "", pypsi_path: str = "", pulse_file: str = "",
        sim_type: str = "megesse",
        resample_pulse_to_dt_us: float = 5.0,
        use_gpu: bool = False, gpu_device: int = 0,
        visualize: bool = False, debug: bool = False,
        sample_number: int = 1000, length_z: float = 0.005,
        t2_list: list = None, b1_list: list = None) -> db_class.DB:
    """
    Function to be called when using simulation aspect of package from another python application.
    I.E. optimization or ai model training.
    basically just create the parameter options from kwargs and passing it onto core
    """
    config = options.SimulationConfig(
        pypsi_path=pypsi_path, emc_seq_file=emc_seq_file, pulse_file=pulse_file,
        save_path="", database_name="_", sim_type=sim_type, signal_fourier_sampling=False,
        visualize=visualize, debug=debug, resample_pulse_to_dt_us=resample_pulse_to_dt_us,
        use_gpu=use_gpu, gpu_device=gpu_device
    )
    settings = options.SimulationSettings(
        sample_number=sample_number, length_z=length_z,
        t1_list=[1.5], t2_list=t2_list, b1_list=b1_list
    )
    sim_params = options.SimulationParameters(config=config, settings=settings)
    db = core_sim(sim_params=sim_params)
    return db


def core_sim(sim_params: options.SimulationParameters) -> db_class.DB:
    """
    core simulation and plotting
    """
    sim_params.config.display()

    if sim_params.config.sim_type.startswith("mese"):
        sim_obj = simulations.MESE(sim_params=sim_params)
    elif sim_params.config.sim_type == "megesse":
        sim_obj = simulations.MEGESSE(sim_params=sim_params)
    elif sim_params.config.sim_type == "megessevesp":
        sim_obj = simulations.MEGESSEVESP(sim_params=sim_params)
    # if sim_params.config.sim_type == "single":
    #     sim_obj = simulations.(sim_params=sim_params, device=device)
    else:
        err = f"sequence type choice ({sim_params.config.sim_type}) not implemented for simulation"
        log_module.error(err)
        raise ValueError(err)
    # simulate sequence
    sim_obj.simulate()
    # create database
    db = db_class.DB.build_from_sim_data(sim_params=sim_params.sequence, sim_data=sim_obj.data)
    # plot stuff
    if sim_params.config.visualize:
        # plot magnetization profile snapshots
        sim_obj.plot_magnetization_profiles(animate=False)
        sim_obj.plot_emc_signal()
        if sim_params.config.signal_fourier_sampling:
            sim_obj.plot_signal_traces()
        # plot database
        db.plot(sim_obj.fig_path)
    return db


def cli_sim(sim_params: options.SimulationParameters):
    """
    Function to be called when using tool as cmd line interface, passing the cli created options.
    Just doing the core sim and saving the data
    """
    db = core_sim(sim_params=sim_params)
    # save files
    save_path = plib.Path(sim_params.config.save_path).absolute()
    if sim_params.config.config_file:
        c_name = plib.Path(sim_params.config.config_file).absolute().stem
    else:
        c_name = "emc_config"
    save_file = save_path.joinpath(c_name).with_suffix(".json")
    logging.info(f"Save Config File: {save_file.as_posix()}")
    sim_params.save_json(save_file.as_posix(), indent=2)
    # database
    save_file = save_path.joinpath(sim_params.config.database_name)
    logging.info(f"Save DB File: {save_file.as_posix()}")
    db.save(save_file)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    logging.info("__________________________________________________________")
    logging.info("__________________ EMC torch simulation __________________")
    logging.info("__________________________________________________________")

    parser, prog_args = options.create_cli()

    opts = options.SimulationParameters.from_cli(prog_args)
    # set logging level after possible config file read
    if opts.config.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        cli_sim(sim_params=opts)

    except Exception as e:
        logging.exception(e)
        parser.print_usage()
