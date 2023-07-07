from emc_sim import options, pulse_optimization
import logging


def main():
    parser, prog_args = options.createCommandlineParser()

    sim_params = options.SimulationParameters.from_cmd_args(prog_args)
    # set logging level after possible config file read
    if sim_params.config.debug_flag:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=level)

    try:
        # sim_data, sim_params = simulations.mese(sim_params=sim_params, sim_data=sim_data)
        # plotting.plot_emc_sim_data(sim_data)
        pulse_optimization.optimize(sim_params=sim_params)
    except Exception as e:
        print(e)
        parser.print_usage()


if __name__ == '__main__':
    main()

