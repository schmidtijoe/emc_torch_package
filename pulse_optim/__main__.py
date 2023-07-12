""" want to use torch gradient decent to optimize pulse shape"""
import sys
import pathlib as plib

p_wd = plib.Path("/data/pt_np-jschmidt/code/emc_torch").absolute()
sys.path.append(p_wd.as_posix())

import torch
from emc_sim import functions, prep
from emc_sim import options as eso
from pulse_optim import plotting, options, losses
import logging
import tqdm.auto
from collections import OrderedDict
import wandb

logModule = logging.getLogger(__name__)


def func_to_calculate(pulse_x: torch.tensor, pulse_y: torch.tensor,
                      grad: torch.tensor, grad_rephase: torch.tensor,
                      sim_data: eso.SimulationData, dt_s: float):
    # pulse gradient propagation
    sim_data = functions.propagate_gradient_pulse_relax(
        sim_data=sim_data, pulse_x=pulse_x, pulse_y=pulse_y, grad=grad,
        dt_s=dt_s)
    # rephasing propagation
    sim_data = functions.propagate_gradient_pulse_relax(
        pulse_x=torch.zeros((pulse_x.shape[0], grad_rephase.shape[0])).to(sim_data.device),
        pulse_y=torch.zeros((pulse_x.shape[0], grad_rephase.shape[0])).to(sim_data.device), grad=grad_rephase,
        dt_s=dt_s * 10, sim_data=sim_data
    )
    return sim_data


def configure(sim_params: eso.SimulationParameters, optim_config: options.ConfigOptimization):
    # gpu device
    seed = optim_config.random_seed
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    logModule.info(f"run: {optim_config.run}; torch device: {device}; rng seed: {seed}")
    # set rng
    torch.manual_seed(seed)
    # set b1_value range
    n_b1s = 5
    sim_params.settings.b1_list = torch.linspace(0.6, 1.4, n_b1s).tolist()
    # set standard t2
    sim_params.settings.t2_list = [50]  # in ms
    # smaller fov to emphasize profile
    sim_params.settings.length_z = 3e-3
    # set run name and folder
    optim_config.set_name()

    return sim_params, optim_config, device


def set_init_tensors(sim_params: eso.SimulationParameters, device: torch.device, optim_config: options.ConfigOptimization,
                     require_grad_p_g: bool = True):
    # set pulse original
    grad_pulse = prep.GradPulse.prep_single_grad_pulse(
        params=sim_params, excitation_flag=True, grad_rephase_factor=0.0
    )
    # get initial magnetization
    sim_data = eso.SimulationData.from_sim_parameters(sim_params=sim_params, device=device)
    initial_magnetization = sim_data.magnetization_propagation
    sample_axis = sim_data.sample_axis
    # get num b1s
    n_b1s = sim_params.settings.b1_list.__len__()

    # get sampling time -> from us to s
    dt_s = torch.tensor(grad_pulse.dt_sampling_steps * 1e-6).to(device)

    # want to optimize for a pulse shape that is agnostic to b1 changes
    slice_thickness = torch.tensor(0.7).to(device)  # [mm]
    # define target shapes for magnetization values (magnitude phase and z) , after including rephasing
    # dim [mag, phase, z]
    target_shape = torch.zeros((3, n_b1s, sample_axis.shape[0])).to(device)
    # want step function in magnitude x direction with ramp
    target_shape[0, :, torch.abs(sample_axis) < 1e-3 * slice_thickness / 2] = 1.0
    # want flat phase ie nothing to do for dim 1
    # want "anti" step functions in z direction but from initial magnetization
    target_shape[2] = initial_magnetization[0, 0, :, :, -1]
    target_shape[2, :, torch.abs(sample_axis) < 1e-3 * slice_thickness / 2] = torch.zeros(1, device=device)

    # initial guess
    # take maximum amplitude from 90 degree pulse as 0.66
    p_max = torch.max(torch.abs(torch.concatenate((grad_pulse.data_pulse_x, grad_pulse.data_pulse_y)))) * 0.66
    g_max = 35.0  # mT/m  take this as maximum gradient

    # init guesses
    # pulse -> x random sampled between 0 and 0.5 with cos guide underneath (to 0.5 as well), range -1, 1
    # -> y: random sampled between 0 and 1, range (-1, 1)
    # range ensured by tanh activation
    # make sine shape below as guide
    if optim_config.init_type == 0:
        p_a = torch.rand(size=(grad_pulse.data_pulse_x.shape[1],)) - 0.5
    else:
        ax_range = torch.linspace(-torch.pi / 2 * optim_config.init_type, torch.pi / 2 * optim_config.init_type,
                                  grad_pulse.data_pulse_x.shape[1])
        p_a = torch.cos(ax_range)
    p0 = p_a * optim_config.base_cos_scale + \
         (torch.rand(size=(grad_pulse.data_pulse_x.shape[1],)) - 0.5) * (1 - optim_config.base_cos_scale)
    px = torch.tensor(p0, requires_grad=require_grad_p_g, device=device)
    # p0 = torch.rand(size=(grad_pulse.data_pulse_y.shape[1],)) - 0.5
    py = torch.zeros(size=(grad_pulse.data_pulse_y.shape[1],), requires_grad=require_grad_p_g, device=device)
    # grads -> want to enforce g to range(0,1) and gr (0, -1), by sigmoid, init with random samples from 0 to 1
    g = torch.rand(size=(int(grad_pulse.data_grad.shape[0] / 10),), requires_grad=require_grad_p_g, device=device)
    g_re = torch.rand(size=(50,), requires_grad=require_grad_p_g, device=device)

    return px, py, p_max, g, g_re, g_max, target_shape, dt_s


def optimize(sim_params: eso.SimulationParameters, optim_config: options.ConfigOptimization):
    # overwrite everything that is used by sweep should go automatically
    # configure gpu, folder names etc
    sim_params, optim_config, device = configure(sim_params=sim_params, optim_config=optim_config)

    # configure tensors, target, initial guesses etc.
    px, py, p_max, g, g_re, g_max, target_shape, dt_s = set_init_tensors(
        sim_params=sim_params, device=device, optim_config=optim_config
    )

    # set optimizer
    optimizer = torch.optim.SGD([px, py, g, g_re], lr=optim_config.lr, momentum=optim_config.momentum)
    # losses : ToDo - find right lambda values to balance
    loss = losses.LossOptimizer(lambda_shape=10, lambda_smootheness_p=1e3, lambda_smootheness_g=1e-3)
    # torch.autograd.set_detect_anomaly(True)

    with tqdm.auto.trange(optim_config.num_steps) as t:
        t.set_description(f"progress")
        t.set_postfix(ordered_dict=OrderedDict({"loss": -1, "power": torch.sum(torch.sqrt(px ** 2 + py ** 2)).item(),
                                                "g max": torch.max(torch.abs(g)).item()}))
        for i in t:
            # reset simulation data
            sim_data = eso.SimulationData.from_sim_parameters(sim_params=sim_params, device=device)
            # reset gradients
            optimizer.zero_grad()
            # set input
            px_input, py_input, g_input, gr_input = build_input_tensors(
                px, py, g, g_re, p_max, g_max, sim_data
            )

            sim_data = func_to_calculate(
                pulse_x=px_input, pulse_y=py_input, grad=g_input, grad_rephase=gr_input,
                sim_data=sim_data, dt_s=dt_s
            )

            loss.calculate_loss(
                magnetization=sim_data.magnetization_propagation, target_profile=target_shape,
                p_x=px_input, p_y=py_input, g=g_input, g_r=gr_input
            )

            wandb.log(loss.get_registered_loss_dict())

            loss.value.backward()
            optimizer.step()
            t.set_postfix(ordered_dict=OrderedDict({"loss": loss.value.item(),
                                                    "power": torch.sum(torch.sqrt(px ** 2 + py ** 2)).item(),
                                                    "g max": torch.max(torch.abs(g)).item()}))

    plotting.plot_grad_pulse_optim_run(i, px_input, py_input, g_input, gr_input, config=optim_config)
    plotting.plot_mag_prop(sim_data=sim_data,
                           target_profile=target_shape,
                           run=i, config=optim_config)

    # not necessarily needed for wandb tracking
    # plotting.plot_losses(loss_tracker=loss_tracker, config=optim_config)

    # build g and p from optimized parameters and save
    optim_px, optim_py, optim_g, optim_g_re = build_input_tensors(
        px, py, g, g_re, p_max, g_max, sim_data
    )
    optim_px *= 1 / sim_data.b1_vals[:, None]
    optim_py *= 1 / sim_data.b1_vals[:, None]
    tensors = [optim_px[0], optim_py[0], optim_g, optim_g_re]
    names = ["px", "py", "g", "g_re"]
    file_names = [f"{optim_config.optim_save_path.stem}_{names[k]}" for k in range(names.__len__())]

    for k in range(names.__len__()):
        torch.save(tensors[k], optim_config.optim_save_path.with_name(file_names[k]).with_suffix(".pt").as_posix())


def build_input_tensors(px, py, g, g_re, p_max, g_max, sim_data):
    # we want our targets range from 0 to 1,
    # but obeye input ranges, i.e. abs(gradient) < value, abs(rf_power) < value
    # hence we take the Tanh function to map the input to -1 to 1 range within our max values
    g_input = -g_max * torch.nn.Sigmoid()(g.repeat_interleave(10))
    gr_input = g_max * torch.nn.Sigmoid()(g_re)  # sampled on different raster!! (dt * 10),
    px_input = p_max * torch.nn.Tanh()(px)[None, :] * sim_data.b1_vals[:, None]
    py_input = p_max * torch.nn.Tanh()(py)[None, :] * sim_data.b1_vals[:, None]
    return px_input, py_input, g_input, gr_input


def main():
    # create parser
    parser, prog_args = options.create_cmd_line_interface()

    sim_params = eso.SimulationParameters.from_cmd_args(prog_args)
    optim_config = options.ConfigOptimization.from_cmd_line_args(prog_args)
    # set logging level after possible config file read
    if sim_params.config.debug_flag:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=level)

    # setup wandb
    wandb.init()

    try:
        optimize(sim_params=sim_params, optim_config=optim_config)

    except Exception as e:
        print(e)
        parser.print_usage()


if __name__ == '__main__':
    main()
