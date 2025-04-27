

# Generating simulation model

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
exp_folder = 'exp2/data/'
N_ue_values = [4, 5, 6]
num_random_topologies = 50
lambda_values = [0.6, 0.9]
rho_values = [0.2, 0.5, 0.8]
num_solutions = 2

np.random.seed(2)
sim_seeds = np.random.choice(10000, num_random_topologies, replace=False)

# ==== Simulation parameters ====
M_t = 3
N_ap = 11
P_max = 1.0
TxSNR = 20
# ==== Optimization parameters ====
eta_share = 1 / N_ap  # per-AP SINR share
xi_slack = 1e1 / 5
sca_iters = 7
admm_iters = 25
primal_thresh = 1*1e-3
gamma_a_init = 1e-3
v_dual_init = 0.0
order_hint = 'UEs-Topologies-Lambda-Rho-Solutions'

# Loop over number of UEs
for num_ue_it in range(len(N_ue_values)):
    start_time_ue = time.time()
    N_ue = N_ue_values[num_ue_it]
    sinr_results = np.zeros((N_ue, num_random_topologies, len(lambda_values), len(rho_values), num_solutions)) - 1
    snr_sens_results = np.zeros((2, num_random_topologies, len(lambda_values), len(rho_values), num_solutions)) - 1
    admm_hist_results = np.zeros((admm_iters, 2, num_random_topologies, len(lambda_values), len(rho_values), num_solutions)) - 1
    admm_status_results = np.zeros((num_random_topologies, len(lambda_values), len(rho_values), num_solutions)) - 1
    # Loop over topologies
    for seed_it in range(num_random_topologies):
        print('\n', '\b-----', f'N_ue: {N_ue} | Topology: {seed_it+1} / {num_random_topologies}')
        sim_seed = sim_seeds[seed_it]
        params = SimulationParameters(N_ap=N_ap, N_ue=N_ue, M_t=M_t, TxSNR=TxSNR,
                                      comm_K0_dB=10, comm_shadow_sigma_dB=10,
                                      sens_K0_dB=30, sens_shadow_sigma_dB=15, sens_angle_spread_deg=5,
                                      sens_spatialCorr_dim='azimuth', rcs_variance=0.5,
                                      seed=sim_seed)
        sim = WirelessSimulation(params)
        sim.generate_topology(locate_target=True)
        sim.set_PSRs(np.ones(N_ap) * 0.5)
        H_mat_comm = sim.generate_comm_channel()
        _, H_T_sens = sim.generate_sens_channel()
        W_hat_mat, _, _ = sim.generate_fixed_precoders(H_mat_comm)
        W_mat = sim.scale_precoders(W_hat_mat)
        # -- JointOpt with ADMM --
        # Loop over lambda_
        for lambda_it in range(len(lambda_values)):
            lambda_ = lambda_values[lambda_it]  # objective trade-off weight
            # Loop over rho
            for rho_it in range(len(rho_values)):
                rho = rho_values[rho_it]    # ADMM penalty parameter
                # Loop over number of solutions
                for sol_it in range(num_solutions):
                    print('-'*10, f'Lambda: {lambda_} | Rho: {rho} | Solution: {sol_it+1} / {num_solutions}')
                    # Set parameters
                    sim.set_jointOpt_params(lambda_=lambda_, rho=rho, eta_share=eta_share,
                                            xi_slack=xi_slack, sca_iters=sca_iters, admm_iters=admm_iters,
                                            primal_thresh=primal_thresh, gamma_a_init=gamma_a_init,
                                            v_dual_init=v_dual_init)
                    W_star = sim.jointOpt_ADMM(H_mat_comm, H_T_sens, W_mat, print_log=False)
                    H_W = sim.calc_product_H_W(H_mat_comm, W_star)
                    sirn_dB = sim.calculate_sinr(H_W)   # results for comm SINR
                    snr_dB_vec = sim.calculate_sensing_snr(H_T_sens, W_mat)     # results for sensing SNR
                    admm_hist = sim.outputs.jointOpt_history    # results for ADMM history (gamma values, primal res)
                    print('-'*15, 'ADMM Status:', sim.outputs.admm_status)
                    # Store results
                    sinr_results[:, seed_it, lambda_it, rho_it, sol_it] = sirn_dB
                    snr_sens_results[:, seed_it, lambda_it, rho_it, sol_it] = snr_dB_vec
                    admm_hist_results[:admm_hist.shape[0], :, seed_it, lambda_it, rho_it, sol_it] = admm_hist
                    admm_status_results[seed_it, lambda_it, rho_it, sol_it] = 1 if sim.outputs.admm_status == 'Converged' else 0
    results_dict = {
        'TxSNR': TxSNR,
        'sim_seeds': sim_seeds,
        'lambda_values': lambda_values,
        'rho_values': rho_values,
        'order_hint': order_hint,
        'sinr_results': sinr_results,
        'snr_sens_results': snr_sens_results,
        'admm_hist_results': admm_hist_results,
        'admm_status_results': admm_status_results
    }
    # Save results
    if SAVE_RESULTS:
        prefix = f'jointOpt_Nue{N_ue}_{num_random_topologies}Topos_TxSNR{TxSNR}dB_{N_ap}APs_{M_t}ants'
        lib.save_sim_results(results_dict, prefix=prefix, save_dir='../../results/' + exp_folder)
    end_time_ue = time.time()
    print(f"==> Execution time for N_ue {N_ue}: {end_time_ue - start_time_ue:.2f} seconds <==")


end_time = time.time()
print("=" * width)
print(">> Simulation finished <<".center(width))
print(f"Execution time: {end_time - start_time:.2f} seconds")
print("=" * width)

