import torch
import logging
from emc_sim import options, simulations

log_module = logging.getLogger(__name__)


def simulate(sim_params: options.SimulationParameters, device: torch.device = torch.device("cpu")):
    if sim_params.config.sim_type == "mese":
        sim_obj = simulations.MESE(sim_params=sim_params, device=device)
    if sim_params.config.sim_type == "megesse":
        sim_obj = simulations.MEGESSE(sim_params=sim_params, device=device)
    if sim_params.config.sim_type == "fid":
        sim_obj = simulations.FID(sim_params=sim_params, device=device)
    # if sim_params.config.sim_type == "single":
    #     sim_obj = simulations.(sim_params=sim_params, device=device)
    else:
        err = f"sequence type choice ({sim_params}) not implemented for simulation"
        log_module.error(err)
        raise ValueError(err)
    sim_obj.simulate()
    return sim_obj.params, sim_obj.data
