from emc_torch import simulations, options, db_class
import logging
import torch
log_module = logging.getLogger(__name__)
logging.getLogger('simple_parsing').setLevel(logging.WARNING)


def main(sim_params: options.SimulationParameters, device: torch.device = torch.device("cpu")):

    sim_params.config.display()

    if sim_params.config.sim_type.startswith("mese"):
        sim_obj = simulations.MESE(sim_params=sim_params, device=device)
    # elif sim_params.config.sim_type == "megesse":
    #     sim_obj = simulations.MEGESSE(sim_params=sim_params, device=device)
    # elif sim_params.config.sim_type == "fid":
    #     sim_obj = simulations.FID(sim_params=sim_params, device=device)
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
        main(sim_params=opts)

    except Exception as e:
        logging.exception(e)
        parser.print_usage()
