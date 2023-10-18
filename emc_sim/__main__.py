from emc_sim import options, simulations
from emc_db import DB
import logging


def main():
    parser, prog_args = options.create_cli()

    sim_params = options.SimulationParameters.from_cmd_args(prog_args)
    # set logging level after possible config file read
    if sim_params.config.debug_flag:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=level)

    try:
        # simulate
        # simulations.single_pulse(sim_params=sim_params)
        sim_data, sim_params = simulations.fid(sim_params=sim_params, fft=True)
        # sim_data, sim_params = simulations.mese(sim_params=sim_params)
        db = DB.build_from_sim_data(sim_params=sim_params, sim_data=sim_data)
        # plot db curves
        db.plot()
        # plot simulation
        # plotting.plot_emc(sim_data)
        # save db

    except Exception as e:
        print(e)
        parser.print_usage()


if __name__ == '__main__':
    main()

