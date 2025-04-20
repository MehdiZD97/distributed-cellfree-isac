# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from src.utils import library as lib
from src.utils.simulation import SimulationParameters

params = SimulationParameters(N_ap=4, N_ue=8, M_t=10,
                              comm_K0_dB=10, comm_shadow_sigma_dB=10,
                              sens_K0_dB=30, sens_shadow_sigma_dB=15, sens_angle_spread_deg=5, sens_spatialCorr_dim='azimuth',
                              rcs_variance=np.ones(5)/2, seed=10)


#%% Positions

# Positions
ap_positions = lib.generate_AP_positions(params.N_ap, params.ap_radius)
ue_positions = lib.generate_UE_positions(params.N_ue, params.env_radius, random_state=params.seed)
target_position = lib.generate_target_positions(params.env_radius, N_target=1, random_state=params.seed)
# target_position = np.array([250,0])
sensingRxAP_index = lib.find_sensingRxAP(ap_positions, target_position)
# Plotting the topology
lib.plot_topology(ap_positions, ue_positions, params.env_radius, target_position=target_position)

#%% Powers

P_max = 1
PSR_vec = np.ones(params.N_ap) / 2
P_comm_linear = PSR_vec * P_max
P_sensing_linear = (1 - PSR_vec) * P_max

awgn_snr_dB = 20
awgn_snr = 10 ** (awgn_snr_dB / 10)
awgn_sigma_sq = P_max / awgn_snr

# %% Channel

# Generating comm channel
H_mat_comm = np.zeros((params.N_ue, params.M_t, params.N_ap), dtype=complex)
for a in range(params.N_ap):
    if a == sensingRxAP_index:
        continue
    angle_spreads_a = np.random.normal(params.comm_angle_spread_deg, 0.1, params.N_ue)
    angle_spreads_a = np.clip(angle_spreads_a, 0, 15)
    pl_exponents_a = np.random.normal(params.comm_pathloss_exponent, 0.1, params.N_ue)
    pl_exponents_a = np.clip(pl_exponents_a, 2, 3.5)
    shadow_sigmas_a = np.random.normal(params.comm_shadow_sigma_dB, 1, params.N_ue)
    shadow_sigmas_a = np.clip(shadow_sigmas_a, 3, 7)
    ap_a = ap_positions[:, a]
    H_a = lib.generate_channels_singleAP(params.M_t, params.N_ue, ap_position=ap_a, ue_positions=ue_positions, h_ap=params.h_ap, h_ue=params.h_ue,
                                         uca_radius=params.uca_radius, angle_spread_deg=angle_spreads_a,
                                         fc=params.fc, pathloss_exponent=pl_exponents_a, shadow_sigma_dB=shadow_sigmas_a,
                                         K0_dB=params.comm_K0_dB, spatialCorr_nsamps=1000, spatialCorr_dim=params.comm_spatialCorr_dim, mode=params.comm_channel_mode)
    H_a_norm = H_a / np.linalg.norm(H_a, axis=1).reshape(-1, 1)
    H_mat_comm[:, :, a] = H_a_norm

print('Comm Channels Generated with the Shape:')
print('(N_ue, M_t, N_ap)')
print(H_mat_comm.shape)


# Generating sensing channel
rx_ap_position = ap_positions[:, sensingRxAP_index]
target_position = target_position
rx_target_distance = np.linalg.norm(rx_ap_position - target_position, axis=0)
rx_target_distance_3D = np.sqrt(rx_target_distance ** 2 + (params.h_ap - params.h_target) ** 2)
rx_target_elevation_deg = np.rad2deg(np.arctan2(rx_target_distance, params.h_ap - params.h_target))

h_R = lib.generate_channels_singleAP(params.M_t, 1, ap_position=rx_ap_position, ue_positions=target_position.reshape(2, 1),
                                     h_ap=params.h_ap, h_ue=params.h_target, uca_radius=params.uca_radius, angle_spread_deg=params.sens_angle_spread_deg, fc=params.fc,
                                     shadow_sigma_dB=params.sens_shadow_sigma_dB, K0_dB=params.sens_K0_dB, spatialCorr_dim=params.sens_spatialCorr_dim, mode=params.sens_channel_mode)
h_R = h_R.T / np.linalg.norm(h_R)


H_mat_sens = np.zeros((params.M_t, params.M_t, params.N_ap), dtype=complex)
H_T_sens = np.zeros((params.M_t, params.N_ap), dtype=complex)
for a in range(params.N_ap):
    if a == sensingRxAP_index:
        continue
    tx_ap_position = ap_positions[:, a]
    target_tx_distance = np.linalg.norm(target_position - tx_ap_position, axis=0)
    target_tx_distance_3D = np.sqrt(target_tx_distance ** 2 + (params.h_ap - params.h_target) ** 2)
    target_tx_elevation_deg = np.rad2deg(np.arctan2(target_tx_distance, params.h_ap - params.h_target))

    h_T = lib.generate_channels_singleAP(params.M_t, 1, ap_position=tx_ap_position, ue_positions=target_position.reshape(2, 1),
                                         h_ap=params.h_ap, h_ue=params.h_target, uca_radius=params.uca_radius, angle_spread_deg=params.sens_angle_spread_deg, fc=params.fc,
                                         shadow_sigma_dB=params.sens_shadow_sigma_dB, K0_dB=params.sens_K0_dB, spatialCorr_dim=params.sens_spatialCorr_dim, mode=params.sens_channel_mode)
    h_T = h_T.T / np.linalg.norm(h_T)
    H_T_sens[:, a] = np.squeeze(h_T)

    H_a_target = np.random.normal(0, params.rcs_variance[a]) * (h_R @ h_T.T)
    H_mat_sens[:, :, a] = H_a_target

print('Sensing Channels Generated with the Shape:')
print('(M_r, M_t, N_ap)')
print(H_mat_sens.shape)


# Calculating Precoders

# Mode 'conj': conjugate
# Mode 'ns_svd': null-space based on SVD
# Mode 'ns_zf': null-space based on ZF beam
sensing_bf_mode = 'ns_zf'
print(f'Sensing Beamforming Mode: {sensing_bf_mode}')


C = np.zeros((params.N_ue, params.N_ue + 1), dtype=complex)
C[:, :-1] = np.identity(params.N_ue, dtype=complex)

weights = lib.decompose_C(C, H_mat_comm, mode='condition_number')
print('weights based on condition number: [{}]'.format(', '.join('{:.4f}'.format(w) for w in weights)))

H_W = np.zeros((params.N_ue, params.N_ue + 1), dtype=complex)
W_mat = np.zeros((params.M_t, params.N_ue + 1, params.N_ap), dtype=complex)
for a in range(params.N_ap):
    if a == sensingRxAP_index:
        continue
    ap_a = ap_positions[:, a]
    target_distance = np.linalg.norm(target_position - ap_a, axis=0)
    target_angle_deg = np.rad2deg(np.arctan2(target_position[1] - ap_a[1], target_position[0] - ap_a[0]))
    target_angle_deg = (target_angle_deg + 360) % 360
    target_elevation_deg = np.rad2deg(np.arctan2(target_distance, params.h_ap - params.h_target))

    target_dir = lib.uca_array_response(params.M_t, params.uca_radius, params.wavelength, target_angle_deg, target_elevation_deg)
    # target_dir = target_dir / np.linalg.norm(target_dir)
    d_a = target_dir.reshape(-1, 1)

    H_a = H_mat_comm[:, :, a]
    H_H = H_a.conj().T @ H_a
    H_a_plus = np.linalg.pinv(H_H) @ H_a.conj().T

    W_a = np.zeros((params.M_t, params.N_ue + 1), dtype=complex)

    if sensing_bf_mode == 'conj':
        W_a[:, -1] = np.squeeze(d_a)  # conjugate beam directly toward target
    elif sensing_bf_mode == 'ns_svd':
        numNullVecs = int(params.M_t / 4)
        v_nulls = lib.return_null_vectors(H_mat_comm, numVecs=numNullVecs)
        if numNullVecs == 1:
            v_nulls = v_nulls.reshape(v_nulls.shape[0], v_nulls.shape[1], 1)
        v_nulls_mat = v_nulls[a, :, :]
        P_null_a = v_nulls_mat @ v_nulls_mat.conj().T
        d_prime_a = P_null_a @ d_a
        d_hat_a = d_prime_a / np.linalg.norm(d_prime_a)
        W_a[:, -1] = np.squeeze(d_hat_a) * np.sqrt(P_sensing_linear[a])  # projecting on the null space using SVD
    elif sensing_bf_mode == 'ns_zf':
        H_a_null = H_a.conj().T @ np.linalg.pinv(H_a @ H_a.conj().T) @ H_a
        ns_beam = (np.eye(params.M_t) - H_a_null) @ d_a
        ns_beam = ns_beam / np.linalg.norm(ns_beam)
        W_a[:, -1] = np.squeeze(ns_beam) * np.sqrt(P_sensing_linear[a])  # null-space beam using comm ZF beam
    else:
        raise ValueError("Invalid sensing beamforming mode. Must be one of 'conj', 'ns_svd', or 'ns_zf'.")

    alignment = np.abs(np.dot(W_a[:, -1].conj().T, target_dir))
    print(f'\nAlignment for AP {a}: {alignment:.4f}')

    W_a_comm = H_a_plus @ (weights[a] * C[:, :-1])
    W_a[:, :-1] = W_a_comm / np.linalg.norm(W_a_comm, 'fro') * np.sqrt(P_comm_linear[a])
    print(f'Comm Power: {np.sum(np.linalg.norm(W_a[:, :-1], axis=0) ** 2):.4f}')
    H_W += H_a @ W_a
    W_mat[:, :, a] = W_a

plt.imshow(np.abs(H_W), cmap='gray')
plt.colorbar()
plt.title('$\\sum_{a} H_a W_a$ Matrix')
plt.tight_layout()
plt.show()
print('Precoders W_mat Calculated with the size:')
print('(M_t, N_ue+1, N_ap)')
print(W_mat.shape)


# SINR Based on H_W Matrix

sinr_dB = []
for k in range(params.N_ue):
    P_desired = np.abs(H_W[k, k]) ** 2
    P_interference = np.linalg.norm(H_W[k, :]) ** 2 - P_desired
    sinr = P_desired / (P_interference + awgn_sigma_sq)
    sinr_dB.append(10 * np.log10(sinr))
sinr_dB = np.array(sinr_dB)
print('SINRs for Users: [{}]'.format(', '.join('{:.2f}'.format(s) for s in sinr_dB)))

plt.plot(sinr_dB, marker='o')
y_min = min(np.min(sinr_dB), 0)
y_max = max(np.max(sinr_dB)+0.5, 22.5)
plt.ylim([y_min, y_max])
plt.grid(axis='y')
plt.xticks(np.arange(params.N_ue))
plt.xlabel('UE Index')
plt.ylabel('SINR (dB)')
plt.tight_layout()
plt.show()


# Sensing SNR

P_sds = 0
P_sds_sens = 0
for a in range(params.N_ap):
    if a == sensingRxAP_index:
        continue
    h_T = H_T_sens[:,a].reshape(-1, 1)
    W_a = W_mat[:, :, a]
    P_sds += params.rcs_variance[a] * np.linalg.norm(h_T.conj().T @ W_a) ** 2
    P_sds_sens += params.rcs_variance[a] * np.linalg.norm(h_T.conj().T @ W_a[:, -1]) ** 2

snr = P_sds / awgn_sigma_sq
snr_dB = 10 * np.log10(snr)
snr_sens = P_sds_sens / awgn_sigma_sq
snr_sens_dB = 10 * np.log10(snr_sens)
print('Sensing SNR: {:.2f} dB'.format(snr_dB))
print('Sensing SNR (Sensing Beam): {:.2f} dB'.format(snr_sens_dB))

#%% Sensing SNR based on Sensing Channel (H_mat_sens which has random rcs values-not rcs_variance)

rx_ap_position = ap_positions[:, sensingRxAP_index]
target_position = target_position
rx_target_distance = np.linalg.norm(rx_ap_position - target_position, axis=0)
rx_target_distance_3D = np.sqrt(rx_target_distance ** 2 + (params.h_ap - params.h_target) ** 2)
rx_target_angle_deg = np.rad2deg(
    np.arctan2(target_position[1] - rx_ap_position[1], target_position[0] - rx_ap_position[0]))
rx_target_angle_deg = (rx_target_angle_deg + 360) % 360
rx_target_elevation_deg = np.rad2deg(np.arctan2(rx_target_distance, params.h_ap - params.h_target))

rx_target_dir = lib.uca_array_response(params.M_t, params.uca_radius, params.wavelength, rx_target_angle_deg, rx_target_elevation_deg)
rx_target_dir = rx_target_dir / np.linalg.norm(rx_target_dir)
a_R = rx_target_dir.reshape(-1, 1)

rx_gain_jcs = np.zeros((params.N_ue + 1), dtype=complex)
for a in range(params.N_ap):
    if a == sensingRxAP_index:
        continue
    H_a_target = H_mat_sens[:, :, a]
    W_a = W_mat[:, :, a]
    rx_gain_jcs += np.squeeze(a_R.conj().T @ (H_a_target @ W_a))

sensing_snr = np.linalg.norm(rx_gain_jcs[-1]) ** 2 / (np.linalg.norm(a_R) ** 2 * awgn_sigma_sq)
sensing_snr_dB = 10 * np.log10(sensing_snr)
print(f'Sensing-Only SNR: {sensing_snr_dB} dB')

sensing_snr_total = np.linalg.norm(rx_gain_jcs) ** 2 / (np.linalg.norm(a_R) ** 2 * awgn_sigma_sq)
sensing_snr_total_dB = 10 * np.log10(sensing_snr_total)
print(f'Total Sensing SNR: {sensing_snr_total_dB} dB')


