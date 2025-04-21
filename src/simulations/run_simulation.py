
import numpy as np
import matplotlib.pyplot as plt
from src.utils import library as lib
from src.utils.simulation import SimulationParameters, WirelessSimulation

params = SimulationParameters(N_ap=4, N_ue=8, M_t=10, TxSNR=20,
                              comm_K0_dB=10, comm_shadow_sigma_dB=10,
                              sens_K0_dB=30, sens_shadow_sigma_dB=15, sens_angle_spread_deg=5,
                              sens_spatialCorr_dim='azimuth',
                              rcs_variance=np.ones(5) / 2, seed=None)
sim = WirelessSimulation(params)

width = 30
print("-" * width)
print(">> Simulation Started <<".center(width))
print("-" * width)



# Simulation parameters
np.random.seed(1)
save_outputs = True
num_random_topologies = 100
sim_seeds = np.random.choice(10000, num_random_topologies, replace=False)
target_coords = [250, 0]  # target coordinates
splitOpt_sinr_dB = 12  # Constraint for comm SINR
splitOpt_gamma = lib.splitOpt_sinr2gamm(params.TxSNR, splitOpt_sinr_dB)
splitOpt_min_rho = 0.0
splitOpt_max_rho = 1.0
fixed_PSR = 0.2
optimizePA = True

comm_sinrs = np.zeros((len(sim_seeds), params.N_ue))
sens_snrs = np.zeros(len(sim_seeds))
sens_only_snrs = np.zeros(len(sim_seeds))

for randTopology_idx in range(len(sim_seeds)):
    if (randTopology_idx+1) % 5 == 0:
        print('>> Simulations for topology %d' % (randTopology_idx + 1))
    params.seed = sim_seeds[randTopology_idx]
    sim.generate_topology(locate_target=False)
    sim.set_position(obj_typy='target', obj_idx=0, coords=target_coords)
    # sim.plot_topology()
    if optimizePA:
        sim.set_splitOpt_params(gamma=splitOpt_gamma, min_rho=splitOpt_min_rho, max_rho=splitOpt_max_rho, xi=1.0, Psi=2, max_iters=10)
    else:
        sim.set_PSRs(np.ones(params.N_ap) * fixed_PSR)
    sim.run_simulation(optimizePA=optimizePA, sim_log=False, print_log=False)
    comm_sinrs[randTopology_idx, :] = sim.outputs.comm_user_sinr
    sens_snrs[randTopology_idx] = sim.outputs.sens_total_snr
    sens_only_snrs[randTopology_idx] = sim.outputs.sens_only_snr

outputs = {
    'TxSNR': params.TxSNR,
    'rcs_variance': params.rcs_variance,
    'num_random_topologies': num_random_topologies,
    'sim_seeds': sim_seeds,
    'target_coords': target_coords,
    'splitOpt_gamma': splitOpt_gamma,
    'splitOpt_min_rho': splitOpt_min_rho,
    'splitOpt_max_rho': splitOpt_max_rho,
    'optimizePA': optimizePA,
    'fixed_PSR': fixed_PSR,
    'comm_sinrs': comm_sinrs,
    'sens_snrs': sens_snrs,
    'sens_only_snrs': sens_only_snrs}

print("-" * width)
print(">> Simulation finished <<".center(width))
print("-" * width)

# Save outputs
if save_outputs:
    lib.save_sim_outputs(outputs, prefix='splitOpt_test', save_dir='./outputs/')


