
# Generating simulation model

import numpy as np
import cvxpy as cp
import src.utils.library as lib
from src.utils.simulation import SimulationParameters, WirelessSimulation
import time

start = time.time()
# ==== Simulation parameters ====
M_t = 3
N_ap = 11
N_ue = 6
P_max = 1.0
TxSNR = 20
sim_seed = 7658

params = SimulationParameters(N_ap=N_ap, N_ue=N_ue, M_t=M_t, TxSNR=TxSNR,
                              comm_K0_dB=10, comm_shadow_sigma_dB=10,
                              sens_K0_dB=30, sens_shadow_sigma_dB=15, sens_angle_spread_deg=5,
                              sens_spatialCorr_dim='azimuth', rcs_variance=0.5,
                              seed=sim_seed)
sim = WirelessSimulation(params)

sim.generate_topology()
sim.plot_topology()
sim.set_PSRs(np.ones(N_ap) * 0.5)
H_mat_comm = sim.generate_comm_channel()
_, H_T_sens = sim.generate_sens_channel()
W_hat_mat, _, _ = sim.generate_fixed_precoders(H_mat_comm)
W_mat = sim.scale_precoders(W_hat_mat)

# JointOpt with ADMM

# ==== Optimization parameters ====
lambda_ = 0.6   # objective trade-off weight
rho = 0.3   # ADMM penalty parameter
eta_share = 1 / N_ap  # per-AP SINR share
xi_slack = 1e1 / 5
sca_iters = 7
admm_iters = 25
primal_thresh = 1e-3
gamma_a_init = 1e-3
v_dual_init = 0.0


sim.set_jointOpt_params(lambda_=lambda_, rho=rho, eta_share=eta_share,
                        xi_slack=xi_slack, sca_iters=sca_iters, admm_iters=admm_iters,
                        primal_thresh=primal_thresh, gamma_a_init=gamma_a_init,
                        v_dual_init=v_dual_init)
W_star = sim.jointOpt_ADMM(H_mat_comm, H_T_sens, W_mat, print_log=True)
H_W = sim.calc_product_H_W(H_mat_comm, W_star)
sinr_dB = sim.calculate_sinr(H_W)
snr_dB_vec = sim.calculate_sensing_snr(H_T_sens, W_mat)

print('Comm SINRs:', sinr_dB)
print('Sensing SNR:', snr_dB_vec[0])

end = time.time()
print(f"Execution time: {end - start:.2f} seconds")
print('ADMM status:', sim.outputs.admm_status)

#%% Plotting results

import matplotlib.pyplot as plt

hist = sim.outputs.jointOpt_history
plt.plot(np.arange(0, len(hist)), hist[:, 0], '-o', label='Gamma values')
plt.plot(np.arange(0, len(hist)), hist[:, 1], '-s', label='Primal residual')
plt.xticks(np.arange(0, len(hist), 5))
plt.xlabel('ADMM Iteration')
plt.legend()
plt.show()

for a in range(N_ap):
    P_comm = np.linalg.norm(W_star[:, :-1, a], 'fro')**2
    P_sens = np.linalg.norm(W_star[:, -1, a])**2
    print(f'Comm power at AP {a} : {P_comm:.4f}')
    print(f'Sens power at AP {a} : {P_sens:.4f}\n')

