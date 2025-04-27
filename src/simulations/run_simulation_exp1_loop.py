

import numpy as np
from src.utils import library as lib
from src.utils.simulation import SimulationParameters, WirelessSimulation
import time

start_time = time.time()
width = 40
print("=" * width)
print(">> Simulation Started <<".center(width))
print("=" * width)

# ==== Loop parameters ====
SAVE_RESULTS = True
exp_folder = 'exp1/data/'
N_ue_values = [4, 5, 6]
num_random_topologies = 1000
splitOpt_sinr_cons_dB = np.arange(5, 19, 1)

np.random.seed(2)
sim_seeds = np.random.choice(10000, num_random_topologies, replace=False)

# ==== Simulation parameters ====
M_t = 10
N_ap = 4
TxSNR = 20
# ==== Optimization parameters ====
xi = 1.0
Psi = 2
max_iters = 12
splitOpt_min_rho = 0.0
splitOpt_max_rho = 1.0
order_hint = 'UEs-Topologies-SINRconstraintValues'

# Loop over number of UEs
for num_ue_it in range(len(N_ue_values)):
    start_time_ue = time.time()
    N_ue = N_ue_values[num_ue_it]
    sinr_results = np.zeros((N_ue, num_random_topologies, len(splitOpt_sinr_cons_dB))) - 1
    snr_sens_results = np.zeros((2, num_random_topologies, len(splitOpt_sinr_cons_dB))) - 1
    # Loop over topologies
    for seed_it in range(num_random_topologies):
        if (seed_it+1) % 20 == 0:
            print('\n', '\b-----', f'N_ue: {N_ue} | Topology: {seed_it+1} / {num_random_topologies}')
        sim_seed = sim_seeds[seed_it]
        params = SimulationParameters(N_ap=N_ap, N_ue=N_ue, M_t=M_t, TxSNR=TxSNR,
                                      comm_K0_dB=10, comm_shadow_sigma_dB=10,
                                      sens_K0_dB=30, sens_shadow_sigma_dB=15, sens_angle_spread_deg=5,
                                      sens_spatialCorr_dim='azimuth', rcs_variance=0.5, seed=sim_seed)
        sim = WirelessSimulation(params)
        sim.generate_topology(locate_target=True)
        # Loop over SINR constraints
        for gamma_it in range(len(splitOpt_sinr_cons_dB)):
            splitOpt_sinr_dB = splitOpt_sinr_cons_dB[gamma_it]
            splitOpt_gamma = lib.splitOpt_sinr2gamm(TxSNR, splitOpt_sinr_dB)
            sim.set_splitOpt_params(gamma=splitOpt_gamma, min_rho=splitOpt_min_rho, max_rho=splitOpt_max_rho, xi=xi, Psi=Psi, max_iters=max_iters)
            sim.run_simulation(optimizePA=True, sim_log=False, print_log=False)
            sinr_results[:, seed_it, gamma_it] = sim.outputs.comm_user_sinr
            snr_sens_results[0, seed_it, gamma_it] = sim.outputs.sens_total_snr
            snr_sens_results[1, seed_it, gamma_it] = sim.outputs.sens_only_snr
    results_dict = {
        'TxSNR': TxSNR,
        'sim_seeds': sim_seeds,
        'SINR_constraints': splitOpt_sinr_cons_dB,
        'order_hint': order_hint,
        'sinr_results': sinr_results,
        'snr_sens_results': snr_sens_results,
    }
    # Save results
    if SAVE_RESULTS:
        prefix = f'splitOpt_Nue{N_ue}_{num_random_topologies}Topos_TxSNR{TxSNR}dB_{N_ap}APs_{M_t}ants'
        lib.save_sim_results(results_dict, prefix=prefix, save_dir='../../results/' + exp_folder)
    end_time_ue = time.time()
    print(f"==> Execution time for N_ue {N_ue}: {end_time_ue - start_time_ue:.2f} seconds <==")

end_time = time.time()
print("=" * width)
print(">> Simulation finished <<".center(width))
print(f"Execution time: {end_time - start_time:.2f} seconds")
print("=" * width)



