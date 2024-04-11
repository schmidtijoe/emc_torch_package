
# want to create script for optimizing database creation wrt. SAR minimization and FA variation to separate db entries.
import logging
import torch

import emc_torch
from emc_torch import simulations
import pathlib as plib
import wandb


def main():
    wandb.init()
    # we want to set emc params (possibly load in via CLI)
    logging.info("Setup")
    base_path = plib.Path(__file__).absolute().parent.parent
    emc_opts = emc_torch.options.SimulationParameters.load_defaults()
    # for now set jstmc sequence pypsi file to use
    pyp_path = base_path.joinpath("optimizations/config/pypsi_jsopti_mese_acc3_1a.pkl")
    run_path = plib.Path("/data/pt_np-jschmidt/data/03_sequence_dev/jstmc_optimized_et/optimization/").absolute()
    logging.info(f"load pypsi interface file {pyp_path}")
    emc_opts.set_pypsi_interface(pyp=pyp_path)
    # set some paths
    emc_opts.config.pypsi_path = pyp_path.as_posix()
    emc_opts.config.save_path = run_path.as_posix()
    # use gpu
    emc_opts.config.use_gpu = True
    # no visuals
    emc_opts.config.visualize = False

    # setup settings
    emc_opts.settings.b1_list = [[0.2, 1.6, 0.1]]
    emc_opts.settings.t2_list = [
        [1, 50, 2],
        [50, 150, 10],
        [150, 1000, 25]
    ]

    # set FA arrays
    fa_s = [
        wandb.config[f"fa_{k}"] for k in range(emc_opts.sequence.etl)
    ]
    emc_opts.sequence.refocus_angle = fa_s
    logging.info("Start Simulation")
    sim_obj = simulations.MESE(sim_params=emc_opts)
    sim_obj.simulate()
    # arrays are stored in the sim object
    # we want to compute and log the loss. can incorporate phase information here!
    # get emc data flattened, we dont care for the contributions and set all equal.

    # get curves
    emc_curves_mag = sim_obj.data.emc_signal_mag
    # flatten, we treat all curves equally
    emc_curves_mag = torch.reshape(emc_curves_mag, (-1, emc_curves_mag.shape[-1]))

    logging.info("Calculate losses")
    # get area under curve as proxy for signal aka SNR, want the sum of the signals to be as high as possible
    # since we dont want certain areas of the field to drive we take a mean
    signal_loss = torch.mean(emc_curves_mag)
    # get l2 difference between normalized curves
    emc_curves_mag_norm = torch.linalg.norm(emc_curves_mag, dim=-1, keepdim=True)
    emc_curves_mag_normalized = torch.nan_to_num(emc_curves_mag / emc_curves_mag_norm, nan=0.0, posinf=0.0)
    l2_diff = torch.linalg.norm(emc_curves_mag_normalized[:, None] - emc_curves_mag_normalized[None, :], dim=-1)
    # l2_plot = l2_diff.cpu().numpy()
    # lets plot for viualizing
    # fig = go.Figure()
    # fig.add_trace(
    #     go.Heatmap(z=l2_plot)
    # )
    # fig_name = out_path.joinpath(f"run-{r+1}_l2_diff_of_all_curves").with_suffix(".png")
    # logging.info(f"write file: {fig_name}")
    # fig.write_image(fig_name.as_posix())

    # we want this to be maximal, the rationale is that then curve mismatch is higher for different curves,
    # its a diagonal symmetric matrix
    # since we dont want certain areas of the field to drive the sum we take a mean
    mismatch_loss = torch.mean(l2_diff)
    logging.info(f"Signal loss; {signal_loss.item():.5f}, mismatch loss: {mismatch_loss.item():.5f}")
    wandb.log({"loss": -signal_loss - mismatch_loss, "signal loss": signal_loss, "mismatch loss": mismatch_loss})


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    logging.info("_________________________________________________________")
    logging.info("__________________ EMC ET optimization __________________")
    logging.info("_________________________________________________________")

    try:
        main()

    except Exception as e:
        logging.exception(e)
