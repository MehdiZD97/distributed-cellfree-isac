# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from src.utils import library as lib

# %% Example Run

# Parameters
N_ap = 4  # Number of APs
M_t = 16  # Number of antennas at BS (UCA)
N_ue = 8  # Number of users
h_ap = 10  # Height of AP in meters
h_ue = 1.5  # Height of user in meters
h_target = 0.5  # Height of target in meters
env_radius = 1000  # Environment radius
ap_radius = env_radius * 0.65
fc = 3.5e9
c = 3e8
wavelength = c / fc
uca_radius = M_t * wavelength / (4*np.pi)  # UCA radius (m)

# Positions
ap_positions = lib.generate_AP_positions(N_ap, ap_radius)
ue_positions = lib.generate_UE_positions(N_ue, env_radius, random_state=None)
# ue_positions = np.array([[0], [1000]])
# target_position = lib.generate_target_positions(env_radius, N_target=1, random_state=None)
target_position = np.array([250,0])
sensingRxAP_index = lib.find_sensingRxAP(ap_positions, target_position)
N_tx_ap = N_ap - 1

# Plotting the topology
lib.plot_topology(ap_positions, ue_positions, env_radius, target_position=target_position)

# %% Generating Communication Channels

comm_K0_dB = 10
comm_angle_spread_deg = 5
comm_pl_exponent = 2.5
comm_shadow_sigma_dB = 4
comm_spatialCorr_dim = 'none'

H_mat = np.zeros((N_ue, M_t, N_ap), dtype=complex)
for m in range(N_ap):
    if m == sensingRxAP_index:
        continue
    angle_spreads_m = np.random.normal(comm_angle_spread_deg, 0.1, N_ue)
    angle_spreads_m = np.clip(angle_spreads_m, 0, 15)
    pl_exponents_m = np.random.normal(comm_pl_exponent, 0.1, N_ue)
    pl_exponents_m = np.clip(pl_exponents_m, 2, 3.5)
    shadow_sigmas_m = np.random.normal(comm_shadow_sigma_dB, 1, N_ue)
    shadow_sigmas_m = np.clip(shadow_sigmas_m, 3, 7)
    ap_m = ap_positions[:, m]
    H_m = lib.generate_channels_singleAP(M_t, N_ue, ap_position=ap_m, ue_positions=ue_positions, h_ap=h_ap, h_ue=h_ue,
                                         uca_radius=uca_radius, angle_spread_deg=angle_spreads_m,
                                         fc=fc, pathloss_exponent=pl_exponents_m, shadow_sigma_dB=shadow_sigmas_m,
                                         K0_dB=comm_K0_dB, spatialCorr_nsamps=1000, spatialCorr_dim=comm_spatialCorr_dim, mode='rician')
    H_m_norm = H_m / np.linalg.norm(H_m, axis=1).reshape(-1, 1)
    H_mat[:, :, m] = H_m_norm

print('Channels Generated with the size:')
print('(N_ue, M_t, N_ap)')
print(H_mat.shape)

# %% Generating Sensing Channel

# OFDM Parameters
BW = 20e6  # bandwidth
N_sc = 64  # num of subcarriers
f_sc = BW / N_sc
T_sc = 1 / f_sc
T_cp = (1 / 4) * T_sc
T_sym = T_sc + T_cp
n_subcarrIdx = 20
k_symIdx = 2
target_shadow_sigma_dB = 2
target_K0_dB = 40
target_angle_spread_deg = 10
target_spatialCorr_dim = 'azimuth'
beta_rcs = 1

rx_ap_position = ap_positions[:, sensingRxAP_index]
target_position = target_position
rx_target_distance = np.linalg.norm(rx_ap_position - target_position, axis=0)
rx_target_distance_3D = np.sqrt(rx_target_distance ** 2 + (h_ap - h_target) ** 2)
rx_target_angle_deg = np.rad2deg(
    np.arctan2(target_position[1] - rx_ap_position[1], target_position[0] - rx_ap_position[0]))
rx_target_angle_deg = (rx_target_angle_deg + 360) % 360
rx_target_elevation_deg = np.rad2deg(np.arctan2(rx_target_distance, h_ap - h_target))

rx_target_dir = lib.uca_array_response(M_t, uca_radius, wavelength, rx_target_angle_deg, rx_target_elevation_deg)
rx_target_dir_normalized = rx_target_dir / np.linalg.norm(rx_target_dir)
a_R = rx_target_dir_normalized.reshape(-1, 1)

h_R = lib.generate_channels_singleAP(M_t, 1, ap_position=rx_ap_position, ue_positions=target_position.reshape(2, 1),
                                     h_ap=h_ap, h_ue=h_target, uca_radius=uca_radius, angle_spread_deg=target_angle_spread_deg, fc=fc,
                                     shadow_sigma_dB=target_shadow_sigma_dB, K0_dB=target_K0_dB, spatialCorr_dim=target_spatialCorr_dim, mode='rician')
h_R = h_R.T / np.linalg.norm(h_R)

H_mat_target = np.zeros((M_t, M_t, N_ap), dtype=complex)
for m in range(N_ap):
    if m == sensingRxAP_index:
        continue
    tx_ap_position = ap_positions[:, m]
    target_tx_distance = np.linalg.norm(target_position - tx_ap_position, axis=0)
    target_tx_distance_3D = np.sqrt(target_tx_distance ** 2 + (h_ap - h_target) ** 2)
    target_tx_angle_deg = np.rad2deg(
        np.arctan2(target_position[1] - tx_ap_position[1], target_position[0] - tx_ap_position[0]))
    target_tx_angle_deg = (target_tx_angle_deg + 360) % 360
    target_tx_elevation_deg = np.rad2deg(np.arctan2(target_tx_distance, h_ap - h_target))

    target_tx_dir = lib.uca_array_response(M_t, uca_radius, wavelength, target_tx_angle_deg, target_tx_elevation_deg)
    target_tx_dir = target_tx_dir / np.linalg.norm(target_tx_dir)
    a_T = target_tx_dir.reshape(-1, 1)

    h_T = lib.generate_channels_singleAP(M_t, 1, ap_position=tx_ap_position, ue_positions=target_position.reshape(2, 1),
                                         h_ap=h_ap, h_ue=h_target, uca_radius=uca_radius, angle_spread_deg=target_angle_spread_deg, fc=fc,
                                         shadow_sigma_dB=target_shadow_sigma_dB, K0_dB=target_K0_dB, spatialCorr_dim=target_spatialCorr_dim, mode='rician')
    h_T = h_T.T / np.linalg.norm(h_T)

    target_range = rx_target_distance_3D + target_tx_distance_3D
    tau_delay = 0  # target_range / c

    H_m_target = beta_rcs * np.exp(-1 * 2 * np.pi * n_subcarrIdx * f_sc * tau_delay) * (h_R @ h_T.T)
    H_mat_target[:, :, m] = H_m_target

print('H_mat_target Generated with the size:')
print('(M_r, M_t, N_ap)')
print(H_mat_target.shape)

# %% Calculating Precoders

# Mode 'conj': conjugate
# Mode 'ns_svd': null-space based on SVD
# Mode 'ns_zf': null-space based on ZF beam
sensing_bf_mode = 'ns_zf'
print(f'Sensing Beamforming Mode: {sensing_bf_mode}')

P_sensing_linear = np.array([1, 1, 1, 1, 1])
P_comm_linear = 2 - P_sensing_linear

C = np.zeros((N_ue, N_ue + 1), dtype=complex)
C[:, 1:] = np.identity(N_ue, dtype=complex)

if sensing_bf_mode == 'ns_svd':
    numNullVecs = int(M_t / 4)
    v_nulls = lib.return_null_vectors(H_mat, numVecs=numNullVecs)
    if numNullVecs == 1:
        v_nulls = v_nulls.reshape(v_nulls.shape[0], v_nulls.shape[1], 1)

weights = lib.decompose_C(C, H_mat, mode='condition_number')
print('weights based on condition number: [{}]'.format(', '.join('{:.4f}'.format(w) for w in weights)))

H_W = np.zeros((N_ue, N_ue + 1), dtype=complex)
W_mat = np.zeros((M_t, N_ue + 1, N_ap), dtype=complex)
for m in range(N_ap):
    if m == sensingRxAP_index:
        continue
    ap_m = ap_positions[:, m]
    target_distance = np.linalg.norm(target_position - ap_m, axis=0)
    target_angle_deg = np.rad2deg(np.arctan2(target_position[1] - ap_m[1], target_position[0] - ap_m[0]))
    target_angle_deg = (target_angle_deg + 360) % 360
    target_elevation_deg = np.rad2deg(np.arctan2(target_distance, h_ap - h_target))

    target_dir = lib.uca_array_response(M_t, uca_radius, wavelength, target_angle_deg, target_elevation_deg)
    target_dir = target_dir / np.linalg.norm(target_dir)
    d_m = target_dir.reshape(-1, 1)

    H_m = H_mat[:, :, m]
    H_H = H_m.conj().T @ H_m
    H_m_plus = np.linalg.pinv(H_H) @ H_m.conj().T

    W_m = np.zeros((M_t, N_ue + 1), dtype=complex)

    if sensing_bf_mode == 'conj':
        W_m[:, 0] = np.squeeze(d_m)  # conjugate beam directly toward target
    elif sensing_bf_mode == 'ns_svd':
        v_nulls_mat = v_nulls[m, :, :]
        P_null_m = v_nulls_mat @ v_nulls_mat.conj().T
        d_prime_m = P_null_m @ d_m
        d_hat_m = d_prime_m / np.linalg.norm(d_prime_m)
        W_m[:, 0] = np.squeeze(d_hat_m) * np.sqrt(P_sensing_linear[m])  # projecting on the null space using SVD
    elif sensing_bf_mode == 'ns_zf':
        H_m_null = H_m.conj().T @ np.linalg.pinv(H_m @ H_m.conj().T) @ H_m
        ns_beam = (np.eye(M_t) - H_m_null) @ d_m
        ns_beam = ns_beam / np.linalg.norm(ns_beam)
        W_m[:, 0] = np.squeeze(ns_beam) * np.sqrt(P_sensing_linear[m])  # null-space beam using comm ZF beam
    else:
        raise ValueError("Invalid sensing beamforming mode. Must be one of 'conj', 'ns_svd', or 'ns_zf'.")

    alignment = np.abs(np.dot(W_m[:, 0].conj().T, target_dir))
    print(f'\nAlignment for AP {m}: {alignment:.4f}')

    W_m_comm = H_m_plus @ (weights[m] * C[:, 1:])
    W_m[:, 1:] = W_m_comm / np.linalg.norm(W_m_comm, 'fro') * np.sqrt(P_comm_linear[m])
    print(f'Comm Power: {np.sum(np.linalg.norm(W_m[:, 1:], axis=0) ** 2):.4f}')
    H_W += H_m @ W_m
    W_mat[:, :, m] = W_m

plt.imshow(np.abs(H_W), cmap='gray')
plt.colorbar()
plt.title('$\\sum_{m} H_m W_m$ Matrix')
plt.show()
print('Precoders W_mat Calculated with the size:')
print('(M_t, N_ue+1, N_ap)')
print(W_mat.shape)

# %% SINR Based on H_W Matrix

awgn_snr_dB = 25
awgn_snr = 10 ** (awgn_snr_dB / 10)
P_comm = 1
P_sensing = 1
awgn_sigma_sq = (P_comm+P_sensing) / awgn_snr

sinr_dB = []
for k in range(N_ue):
    P_desired = np.abs(H_W[k, k + 1]) ** 2
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
plt.xticks(np.arange(N_ue))
plt.xlabel('UE Index')
plt.ylabel('SINR (dB)')
plt.show()

# %% Sensing SNR

rx_gain_jcs = np.zeros((N_ue + 1), dtype=complex)
for m in range(N_ap):
    if m == sensingRxAP_index:
        continue
    H_m_target = H_mat_target[:, :, m]
    W_m = W_mat[:, :, m]
    rx_gain_jcs += np.squeeze(a_R.conj().T @ H_m_target @ W_m)

sensing_snr = np.linalg.norm(rx_gain_jcs[0]) ** 2 / (np.linalg.norm(a_R) ** 2 * awgn_sigma_sq)
sensing_snr_dB = 10 * np.log10(sensing_snr)
print(f'Sensing-Only SNR: {sensing_snr_dB} dB')

sensing_snr_total = np.linalg.norm(rx_gain_jcs) ** 2 / (np.linalg.norm(a_R) ** 2 * awgn_sigma_sq)
sensing_snr_total_dB = 10 * np.log10(sensing_snr_total)
print(f'Total Sensing SNR: {sensing_snr_total_dB} dB')
