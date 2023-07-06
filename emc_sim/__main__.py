from emc_sim import options, simulations, plotting
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
    sim_data = options.SimulationData.set_with_etl_length(sim_params.sequence.ETL)

    try:
        # sim_data, sim_params = simulations.mese(sim_params=sim_params, sim_data=sim_data)
        # plotting.plot_emc_sim_data(sim_data)
        simulations.single_pulse(sim_params=sim_params, sim_data=sim_data)
    except Exception as e:
        print(e)
        parser.print_usage()


if __name__ == '__main__':
    main()

