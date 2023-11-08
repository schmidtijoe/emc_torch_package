import torch
import logging
from emc_sim import options, simulations
from emc_db import DB
log_module = logging.getLogger(__name__)


def simulate(sim_params: options.SimulationParameters, device: torch.device = torch.device("cpu")):
    if sim_params.config.sim_type.startswith("mese"):
        sim_obj = simulations.MESE(sim_params=sim_params, device=device)
    elif sim_params.config.sim_type == "megesse":
        sim_obj = simulations.MEGESSE(sim_params=sim_params, device=device)
    elif sim_params.config.sim_type == "fid":
        sim_obj = simulations.FID(sim_params=sim_params, device=device)
    # if sim_params.config.sim_type == "single":
    #     sim_obj = simulations.(sim_params=sim_params, device=device)
    else:
        err = f"sequence type choice ({sim_params.config.sim_type}) not implemented for simulation"
        log_module.error(err)
        raise ValueError(err)
    # simulate sequence
    sim_obj.simulate()
    # create database
    db = DB.build_from_sim_data(sim_params=sim_params.sequence, sim_data=sim_obj.data)
    # plot stuff
    if sim_params.config.visualize:
        # plot magnetization profile snapshots
        sim_obj.plot_magnetization_profiles(animate=False)
        sim_obj.plot_emc_signal()
        if sim_params.config.signal_fourier_sampling:
            sim_obj.plot_signal_traces()
        # plot database
        db.plot(sim_obj.fig_path)
    return db, sim_obj.params, sim_obj.data
