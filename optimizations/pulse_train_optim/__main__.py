"""
want to optimize pulse fa (possibly phase) to minimize correlation between emc curves,
while trying to maximize snr throughout pulse train emc
"""
import sys
import pathlib as plib

p_wd = plib.Path("/data/pt_np-jschmidt/code/emc_torch").absolute()
sys.path.append(p_wd.as_posix())

from pulse_optim import options as poo
from pulse_train_optim import optimizer as pto
from emc_sim import options as eso
from emc_sim import simulations, blocks
import torch
import logging
import tqdm.auto
from collections import OrderedDict


def configure(sim_params: eso.SimulationParameters, optim_config: poo.ConfigOptimization):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # gpu device
    seed = optim_config.random_seed
    device = torch.device('cpu')
    logging.info(f"run: {optim_config.run}; torch device: {device}; rng seed: {seed}")
    # set rng
    torch.manual_seed(seed)
    # set b1_value range
    n_b1s = 5
    sim_params.settings.b1_list = torch.linspace(0.6, 1.4, n_b1s).tolist()
    # set standard t2
    sim_params.settings.t2_list = [[15, 55, 10], [60, 160, 50], [200, 500, 100]]  # in ms
    # smaller fov to emphasize profile
    sim_params.settings.length_z = 3e-3
    # set run name and folder
    optim_config.set_name()

    return sim_params, optim_config, device


def build_input_tensors(pulse_fa: torch.tensor):
    # want the ETL length optimizing params to be between 0 and 1
    # and enforce bounds here
    # fa_bounds = torch.tensor([100.0, 180.0])
    # use sigmoid to map to between 0 and 1
    # mapped_pulse_fa = torch.nn.Sigmoid()(pulse_fa)
    # add lower bound and scale with range
    # mapped_pulse_fa *= torch.gradient(fa_bounds)[0][0]
    # mapped_pulse_fa += fa_bounds[0]
    return torch.nn.Tanh()(pulse_fa)


def main(sim_params: eso.SimulationParameters, optim_config: poo.ConfigOptimization):
    sim_params, optim_config, device = configure(sim_params=sim_params, optim_config=optim_config)

    # prep pulse grad data - this holds the pulse data and timings
    gp_excitation, gps_refocus, timing, acquisition = prep.prep_gradient_pulse_mese(
        sim_params=sim_params
    )
    # set devices
    gp_excitation.set_device(device)
    timing.set_device(device)
    for gp in gps_refocus:
        gp.set_device(device)
    acquisition.set_device(device)

    # need to set initial values - take fa to be between 0 and 1. map it with a sigmoid to ensure bounds.
    # 1 is 180 degrees
    fa = torch.full((sim_params.sequence.ETL,), fill_value=0.5, device=device, requires_grad=True)

    # set optimizer
    optimizer = torch.optim.SGD([fa], lr=optim_config.lr, momentum=optim_config.momentum)
    torch.autograd.set_detect_anomaly(True)
    # set loss obj
    loss = pto.LossOptimize(lambda_snr=0.5, lambda_corr=1.0)
    sim_data = eso.SimulationData.from_sim_parameters(sim_params=sim_params, device=device)
    loss.loss_corr.set_weight_matrix(sim_data=sim_data)
    # run through optimization
    with tqdm.auto.trange(optim_config.num_steps) as t:
        t.set_description(f"progress")
        t.set_postfix(ordered_dict=OrderedDict({"loss": -1}))
        for i in t:
            # reset simulation data
            sim_data = eso.SimulationData.from_sim_parameters(sim_params=sim_params, device=device)
            # reset gradients
            optimizer.zero_grad()
            # set input
            fa_input = build_input_tensors(fa)
            # run simulation
            sim_data, _ = simulations.mese_optim(sim_params=sim_params, sim_data=sim_data,
                                                 fa_input=fa_input, gp_excitation=gp_excitation,
                                                 gp_refocusing=gps_refocus, timing=timing, acquisition=acquisition)
            # calculate loss
            loss.calculate(sim_data=sim_data)
            logging.debug("backpropagate")
            loss.value.backward()
            optimizer.step()
            t.set_postfix(ordered_dict=OrderedDict({"loss": loss.value.item()}))

    optim_fa = build_input_tensors(fa)
    logging.info(f"optimized fa train: {optim_fa * 180.0}")
    file_name = f"{optim_config.optim_save_path.stem}_optimized_fa"
    save_name = optim_config.optim_save_path.with_name(file_name).with_suffix(".pt")
    save_name.mkdir(parents=True, exist_ok=True)
    logging.info(f"saving file: {save_name}")
    torch.save(optim_fa, save_name.as_posix())


if __name__ == '__main__':
    # create parser
    parser, prog_args = poo.create_cmd_line_interface()

    sim_params = eso.SimulationParameters.from_cli(prog_args)
    optim_config = poo.ConfigOptimization.from_cmd_line_args(prog_args)
    # set logging level after possible config file read
    if sim_params.seq_params.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=level)

    try:
        main(sim_params=sim_params, optim_config=optim_config)
    except Exception as e:
        print(e)
        parser.print_usage()
