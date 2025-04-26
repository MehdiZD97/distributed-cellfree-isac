
# Generating simulation paramters

import numpy as np
import cvxpy as cp
import src.utils.library as lib
from src.utils.simulation import SimulationParameters, WirelessSimulation


M_t = 10     # same as N
N_ap = 3    # same as M
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
lambda_ = 0.7   # objective trade-off weight
rho = 0.5   # ADMM penalty parameter
eta_vec = np.ones(N_ap) / N_ap  # per-AP SINR share
xi_slack = 1e1/5

sca_iters = 5
admm_iters = 35
primal_thresh = 1e-3

# H_comm shape: (N_ue, M_t, N_ap)
# H_comm = np.random.normal(0, 1/np.sqrt(2), (N_ue, M_t, N_ap)) + 1j * np.random.normal(0, 1/np.sqrt(2), (N_ue, M_t, N_ap))
H_comm = H_mat_comm
# wavelength = 3e8 / 3.5e9
# ap_target_angles = np.array([180 + i*(360/N_ap) for i in range(N_ap)]) % 360
# A_target = np.zeros((M_t, N_ap), dtype=complex)
# for a in range(N_ap):
#     A_target[:, a] = lib.uca_array_response(M_t, 0.1, wavelength, azimuth_deg=ap_target_angles[a])
A_target = H_T_sens

# ==== Initialization of W ====
# W shape: (M_t, N_ue+1, N_ap)
# W = np.zeros((M_t, N_ue+1, N_ap), dtype=complex)
# for a in range(N_ap):
#     W[:, -1, a] = A_target[:, a]
#     W[:, :-1, a] = H_comm[:, :, a].conj().T / np.linalg.norm(H_comm[:, :, a], 'fro')
W = W_mat

# ==== Initialization of interferences I ====
# I_users = np.zeros((N_ap, N_ue))
# for u in range(N_ue):
#     for a in range(N_ap):
#         I_au_inner = 0.0
#         for i in range(N_ue + 1):
#             if i == u: continue
#             I_au_inner += np.linalg.norm(H_comm[u, :, a].conj().T @ W[:, i, a])**2
#         I_users[a, u] = np.sqrt(I_au_inner)
I_users = np.zeros(N_ue)
H_W_admm = sim.calc_product_H_W(H_mat_comm, W)
for u in range(N_ue):
    I_users[u] = (np.linalg.norm(H_W_admm[u,:])**2) - (np.abs(H_W_admm[u, u])**2)


# ==== Initialization of gamma and dual variable ====
gamma_a = np.full(N_ap, 1e-3)
v_dual = np.zeros(N_ap)
gamma = np.mean(gamma_a)


# ==== ADMM loop ====
for it in range(admm_iters):
    # ==== Local updates with inner SCA loop ====
    for a in range(N_ap):
        for sca_t in range(sca_iters):
            # Linearization parameters of objective (obj) and per-user SINR constraint (con)
            g_obj_a = A_target[:, a].conj().T @ W[:, :, a]
            g_con_a = np.zeros(N_ue, dtype=complex)     # only for one AP is available
            for u in range(N_ue):
                h_au = H_comm[u, :, a]
                w_au = W[:, u, a]
                g_con_a[u] = h_au.T @ w_au

            # CVXPY variables
            W_a = cp.Variable((M_t, N_ue+1), complex=True)
            gamma_a_var = cp.Variable()
            s_users = cp.Variable(N_ue, nonneg=True)    # slack variable

            # Linearized sensing utility
            u_sens_lin = 2 * cp.real(g_obj_a @ (W_a.conj().T @ A_target[:, a])) - np.linalg.norm(g_obj_a)**2

            # Constraints
            cons = [cp.norm(W_a, 'fro')**2 <= P_max]
            for u in range(N_ue):
                sinr_lin = 2 * cp.real(g_con_a[u].conj() * H_comm[u, :, a].T @ W_a[:, u]) - np.linalg.norm(g_con_a[u])**2
                # I_u = np.sum(I_users[:, u])
                I_u = I_users[u]
                cons.append(sinr_lin + s_users[u] >= eta_vec[a] * gamma_a_var * (I_u + np.sqrt(P_noise)**2))    # modify based on Minkowski's inequality

            # Augmented Lagrangian term
            aug = lambda_ * gamma_a_var + (1 - lambda_) * u_sens_lin - (rho/2) * cp.square(gamma_a_var - gamma + v_dual[a])

            # Objective
            obj = cp.Maximize(aug - xi_slack*cp.sum(s_users))
            prob = cp.Problem(obj, cons)
            prob.solve(solver=cp.ECOS)

            # Update W and gamma
            W[:, :, a] = W_a.value
            gamma_a[a] = gamma_a_var.value

            # Update interference
            # for u in range(N_ue):
            #     I_au_inner = 0.0
            #     for i in range(N_ue + 1):
            #         if i == u: continue
            #         I_au_inner += np.linalg.norm(H_comm[u, :, a].conj().T @ W[:, i, a])**2
            #     I_users[a, u] = np.sqrt(I_au_inner)
            H_W_admm = sim.calc_product_H_W(H_mat_comm, W)
            for u in range(N_ue):
                I_users[u] = (np.linalg.norm(H_W_admm[u,:])**2) - (np.abs(H_W_admm[u, u])**2)


    # ==== Global consensus update ====
    gamma = np.mean(gamma_a + v_dual)
    primal_res = np.max(np.abs(gamma_a - gamma))
    print('ADMM iter:', it, 'gamma:', gamma, 'primal res:', primal_res)
    if gamma < 0:
        print('Gamma is negative, stopping ADMM')
        break

    # ==== Dual variable update ====
    v_dual += gamma_a - gamma

    # ==== Stopping criteria ====
    if primal_res < primal_thresh:
        print('ADMM converged')
        break

print(f'\n** Gamma opt: {gamma}\n')


#
W_star = W

H_W = sim.calc_product_H_W(H_mat_comm, W_star)
sirn_dB = sim.calculate_sinr(H_W)
snr_dB_vec = sim.calculate_sensing_snr(H_T_sens, W_star)

print('Comm SINRs:', sirn_dB)
print('Sensing SNR:', snr_dB_vec[0])

#%%

sinr_mod_dB = []
for u in range(N_ue):
    interf = (np.linalg.norm(H_W[u,:])**2) - (np.abs(H_W[u, u])**2)
    P_sig = 0.0
    for a in range(N_ap):
        h_au = H_mat_comm[u, :, a]
        w_au = W[:, u, a]
        P_sig += np.abs(np.dot(h_au, w_au))**2
    sinr_u = P_sig / interf
    sinr_mod_dB.append(10 * np.log10(sinr_u))
sinr_mod_dB = np.array(sinr_mod_dB)
print('Modified SINRs:', sinr_mod_dB)
