from emc_sim import options, simulations
from emc_db import DB
import logging
logging.getLogger('simple_parsing').setLevel(logging.WARNING)


def main(sim_params: options.SimulationParameters):
    sim_params.config.display()

    # simulate
    sim_params, sim_data = simulations.simulate(sim_params=sim_params)
    db = DB.build_from_sim_data(sim_params=sim_params, sim_data=sim_data)
    # plot db curves
    db.plot()
    # plot simulation
    # plotting.plot_emc(sim_data)
    # save db


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
