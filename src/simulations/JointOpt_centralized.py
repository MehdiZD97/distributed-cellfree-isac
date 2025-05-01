
# Generating simulation paramters

import numpy as np
import cvxpy as cp
import src.utils.library as lib
from src.simulations.JointOpt_ADMM import W_star
from src.utils.simulation import SimulationParameters, WirelessSimulation


M_t = 10     # same as N
N_ap = 4    # same as M
N_ue = 6    # same as K
P_max = 1.0
TxSNR = 20

params = SimulationParameters(N_ap=N_ap, N_ue=N_ue, M_t=M_t, TxSNR=TxSNR,
                              comm_K0_dB=10, comm_shadow_sigma_dB=10,
                              sens_K0_dB=20, sens_shadow_sigma_dB=15, sens_angle_spread_deg=5,
                              sens_spatialCorr_dim='azimuth', rcs_variance=0.5,
                              seed=None)
sim = WirelessSimulation(params)

sim.generate_topology()
sim.plot_topology()
sim.set_PSRs(np.ones(N_ap) * 0.5)
H_mat_comm = sim.generate_comm_channel()
_, H_T_sens = sim.generate_sens_channel()
W_hat_mat, _, _ = sim.generate_fixed_precoders(H_mat_comm)
W_mat = sim.scale_precoders(W_hat_mat)

H_W = sim.calc_product_H_W(H_mat_comm, W_mat)
sirn_dB = sim.calculate_sinr(H_W)
snr_dB_vec = sim.calculate_sensing_snr(H_T_sens, W_mat)

print('Comm SINRs:', sirn_dB)
print('Sensing SNR:', snr_dB_vec[0])


#%% JointOpt with ADMM


P_noise = sim.params.P_noise     # same as sigma2
lambda_ = 0.3   # objective trade-off weight
xi_slack = 1e2

sca_iters = 10
prob_iters = 35
conv_thresh = 1e-4

# H_comm shape: (N_ue, M_t, N_ap)
H_comm = H_mat_comm
A_target = H_T_sens

# ==== Initialization of W ====
# W shape: (M_t, N_ue+1, N_ap)
W_prev = W_mat

# ==== Initialization of interferences I ====
I_users = np.zeros(N_ue)
H_W_admm = sim.calc_product_H_W(H_mat_comm, W_prev)
for u in range(N_ue):
    I_users[u] = (np.linalg.norm(H_W_admm[u,:])**2) - (np.abs(H_W_admm[u, u])**2)

gamma_prev = 1e-4

# ==== Opt loop ====
for it in range(prob_iters):
    # ==== Inner SCA loop ====
    for sca_t in range(sca_iters):
        # CVXPY variables
        # W_var = cp.Variable((M_t, (N_ue+1), N_ap), complex=True)
        W_var = cp.Variable(((N_ue + 1), M_t*N_ap), complex=True)
        gamma_var = cp.Variable()
        s_users = cp.Variable(N_ue, nonneg=True)  # slack variable

        # Linearization parameters of objective (obj) and per-user SINR constraint (con)
        u_sens_lin = []
        for a in range(N_ap):
            g_obj_a = A_target[:, a].conj().T @ W_prev[:, :, a]
            # Linearized sensing utility
            # W_a = W_var[:, :, a]
            W_a = W_var[:, a*M_t:(a+1)*M_t].T
            u_sens_lin.append(2 * cp.real(g_obj_a @ (W_a.conj().T @ A_target[:, a])) - np.linalg.norm(g_obj_a) ** 2)
        u_sens_total = cp.sum(u_sens_lin)

        g_con_list = []
        for u in range(N_ue):
            h_u = np.reshape(H_comm[u, :, :], (M_t*N_ap, 1), order='F')
            w_u_t = np.reshape(W_prev[:, u, :], (M_t*N_ap, 1), order='F')
            # w_u_var = cp.reshape(W_var[:, u, :], (M_t*N_ap, 1), order='F')
            w_u_var = W_var[u, :]
            hw_t = h_u.T @ w_u_t
            g_con_list.append(2 * cp.real(hw_t.conj() * h_u.T @ w_u_var) - np.linalg.norm(hw_t)**2)
        g_con = cp.vstack(g_con_list)

        # Constraints
        cons = []
        for u in range(N_ue):
            I_u = I_users[u]
            cons.append(g_con[u] + s_users[u] >= (gamma_var * (I_u + P_noise)))    # modify based on Minkowski's inequality
        # Power constraint
        for a in range(N_ap):
            W_a = W_var[:, a * M_t:(a + 1) * M_t].T
            cons.append(cp.norm(W_a, 'fro')**2 <= P_max)

        # Original objective
        orig_objective = lambda_ * gamma_var + (1 - lambda_) * u_sens_total

        # Objective
        obj = cp.Maximize(orig_objective - xi_slack * cp.sum(s_users))
        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.ECOS)

        # Update W and gamma
        W_new = W_var.value
        W_new = np.reshape(W_new, (N_ue+1, M_t, N_ap), order='F')
        W_prev = np.transpose(W_new, axes=(1,0,2))

        # Update interference
        H_W_admm = sim.calc_product_H_W(H_mat_comm, W_prev)
        for u in range(N_ue):
            I_users[u] = (np.linalg.norm(H_W_admm[u, :]) ** 2) - (np.abs(H_W_admm[u, u]) ** 2)

    res = np.abs(gamma_prev - gamma_var.value)
    print('Iter:', it, 'gamma:', gamma_var.value, 'residual:', res)
    # ==== Stopping criteria ====
    if res < conv_thresh:
        print('ADMM converged')
        break
    gamma_prev = gamma_var.value

print(f'\n** Gamma opt: {gamma_prev}\n')

#%%

sim.set_jointOpt_params(lambda_=0.5, xi_slack=1e2, sca_iters=10, cent_iters=35, cent_conv_thresh=1e-4)
W_star = sim.jointOpt_centralized(H_mat_comm, H_T_sens, W_mat, print_log=True)
H_W = sim.calc_product_H_W(H_mat_comm, W_star)
sirn_dB = sim.calculate_sinr(H_W)
snr_dB_vec = sim.calculate_sensing_snr(H_T_sens, W_star)

print('Optimization Results:')
print('Comm SINRs:', sirn_dB)
print('Sensing SNR:', snr_dB_vec[0])
