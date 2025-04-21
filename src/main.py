import numpy as np
import matplotlib.pyplot as plt
from src.utils import library as lib
from src.utils.simulation import SimulationParameters, WirelessSimulation

params = SimulationParameters(N_ap=4, N_ue=8, M_t=10, TxSNR=20,
                              comm_K0_dB=10, comm_shadow_sigma_dB=10,
                              sens_K0_dB=10, sens_shadow_sigma_dB=15, sens_angle_spread_deg=5,
                              sens_spatialCorr_dim='azimuth',
                              rcs_variance=np.ones(5) / 2, seed=None)
sim = WirelessSimulation(params)

params.seed = 1
sinr_consttraint_dB = 12  # Constraint for comm SINR
splitOpt_gamma = lib.splitOpt_sinr2gamm(params.TxSNR, sinr_consttraint_dB)
min_rho = 0.1
max_rho = 0.9

sim.generate_topology(locate_target=False)
sim.set_position(obj_typy='target', obj_idx=0, coords=[250,0])
sim.plot_topology()
sim.set_PSRs(np.ones(params.N_ap) * 0.5)  # if optimizePA is False
sim.set_splitOpt_params(gamma=splitOpt_gamma, min_rho=min_rho, max_rho=max_rho)  # if optimizePA is True
sim.run_simulation(optimizePA=True, sim_log=True, print_log=False)

print('Average Comm SINR (dB):', sim.outputs.comm_avg_sinr)
print('Total Sensing SNR (dB):', sim.outputs.sens_total_snr)
print('Sensing SNR (Sensing Beam) (dB):', sim.outputs.sens_only_snr)
print('PSR vector:', sim.outputs.PSRs)
print('SplitOpt History:', sim.outputs.splitOpt_history)

lib.plot_splitOpt_history(sim.outputs.splitOpt_history)
lib.plot_PSRs(sim.outputs.PSRs)

