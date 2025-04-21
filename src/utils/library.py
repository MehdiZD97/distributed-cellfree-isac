# Library for utility functions

# Functions and imports

import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime


# ----------
# Functions for Decomposition
# ----------

# Decomposes the desired matrix C based on a criterion
def decompose_C(C, H_mat, mode='condition_number'):
    N_ap = H_mat.shape[2]
    if mode == 'max_singular_value':
        sing_val_max = []
        for m in range(N_ap):
            U, S, V_H = np.linalg.svd(H_mat[:, :, m])
            sing_val_max.append(S[0])
        weights = [s / sum(sing_val_max) for s in sing_val_max]
    elif mode == 'condition_number':
        condition_numbers = []
        for m in range(N_ap):
            U, S, V_H = np.linalg.svd(H_mat[:, :, m])
            condition_number = S[0] / S[-1] if S[-1] > 1e-9 else np.inf  # Avoid divide by zero
            condition_numbers.append(
                1 / condition_number if condition_number != np.inf else 0)  # Reciprocal of condition number
        weights = [cn / sum(condition_numbers) if sum(condition_numbers) != 0 else 1 / N_ap for cn in condition_numbers]
    elif mode == 'frobenius_norm':
        frobenius_norms_inv_power = []
        for m in range(N_ap):
            H_m = H_mat[:, :, m]
            H_m_plus = np.linalg.pinv(H_m)
            fro_norm = np.linalg.norm(H_m_plus, 'fro')
            frobenius_norms_inv_power.append(fro_norm ** (-1) if fro_norm != 0 else 0)  # Reciprocal of Frobenius norm
        weights = [fn / sum(frobenius_norms_inv_power) if sum(frobenius_norms_inv_power) != 0 else 1 / N_ap for fn in
                   frobenius_norms_inv_power]
    else:
        raise ValueError("Invalid mode. Must be one of 'max_singular_value', 'condition_number', or 'frobenius_norm'.")
    return np.array(weights)


# Returns the last 'numVecs' singular vectors of channels for each AP
def return_null_vectors(H_mat, numVecs=1):
    N_ap = H_mat.shape[2]
    v_nulls = []
    for m in range(N_ap):
        U, S, V_H = np.linalg.svd(H_mat[:, :, m])
        V = V_H.conj().T
        columnIdx = int(-1 * numVecs)
        v_nulls.append(V[:, columnIdx:])
    return np.squeeze(np.array(v_nulls))


# ----------
# Functions for Placement and Topology
# ----------

# Generates fixed AP positions on a circle
def generate_AP_positions(N_ap, radius, angle_shift_deg=0):
    ap_origin_distance = radius
    ap_origin_angles_deg = [i * 360 / N_ap + angle_shift_deg for i in range(N_ap)]
    ap_origin_angles_rad = np.deg2rad(ap_origin_angles_deg)
    ap_positions = ap_origin_distance * np.array([np.cos(ap_origin_angles_rad), np.sin(ap_origin_angles_rad)])
    return ap_positions


# Generates random UE positions in a circle
def generate_UE_positions(N_ue, env_radius, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    ue_origin_distances = env_radius * np.sqrt(np.random.uniform(0, 1, N_ue))
    ue_origin_angles_deg = np.random.uniform(0, 360, N_ue)
    ue_origin_angles_rad = np.deg2rad(ue_origin_angles_deg)
    ue_positions = ue_origin_distances * np.array([np.cos(ue_origin_angles_rad), np.sin(ue_origin_angles_rad)])
    return ue_positions


# Generates random target positions in a circle
def generate_target_positions(env_radius, N_target=1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    target_distances = env_radius * np.sqrt(np.random.uniform(0, 1, N_target))
    target_angles_deg = np.random.uniform(0, 360)
    target_angles_rad = np.deg2rad(target_angles_deg)
    target_positions = target_distances * np.array([np.cos(target_angles_rad), np.sin(target_angles_rad)])
    return target_positions


# Finds the closest AP to the target position
def find_sensingRxAP(ap_positions, target_position):
    distances = np.linalg.norm(ap_positions - np.tile(target_position.reshape(2, 1), ap_positions.shape[1]), axis=0)
    min_distance_index = np.argmin(distances)
    return min_distance_index


# Plots the network topology (APs, UEs, Target)
def plot_topology(ap_positions, ue_positions, env_radius, target_position=None, fig_margin_ratio=0.1,
                  SHOW_ENV_CIRCLE=True, SHOW_AP_CIRCLE=False, TITLE=True):
    ap_radius = np.linalg.norm(ap_positions[:, 0])
    N_ap = ap_positions.shape[1]
    if target_position is not None:
        sensingRxAP_index = find_sensingRxAP(ap_positions, target_position)
    else:
        sensingRxAP_index = -1
    # Plotting
    circle = plt.Circle((0, 0), env_radius, color='black', fill=False, linestyle='--', linewidth=1.5)
    circle1 = plt.Circle((0, 0), ap_radius, color='black', fill=False, linestyle='--', linewidth=1)
    fig, ax = plt.subplots(figsize=(6, 6))
    if SHOW_ENV_CIRCLE: ax.add_artist(circle)
    if SHOW_AP_CIRCLE: ax.add_artist(circle1)
    LABELED = False
    for i in range(N_ap):
        if i == sensingRxAP_index:
            ax.scatter(ap_positions[0, i], ap_positions[1, i], marker='^', color='green', s=150, label='Sensing Rx AP')
        else:
            if LABELED:
                ax.scatter(ap_positions[0, i], ap_positions[1, i], marker='^', color='red', s=150)
            else:
                ax.scatter(ap_positions[0, i], ap_positions[1, i], marker='^', color='red', s=150, label='Tx APs')
                LABELED = True
    ax.scatter(ue_positions[0, :], ue_positions[1, :], marker='o', color='blue', s=50, label='Users')
    for i, (x, y) in enumerate(zip(ue_positions[0, :], ue_positions[1, :])):
        plt.text(x + 20, y + 20, f'UE{i}', color='blue', fontsize=9)
    if sensingRxAP_index != -1:
        ax.scatter(target_position[0], target_position[1], marker='x', color='black', s=100, label='Target')
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    if TITLE: ax.set_title('Topology of the Network in a Circular Area')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.4)
    fig_margin = fig_margin_ratio * env_radius
    ax.set_xlim(-env_radius - fig_margin, env_radius + fig_margin)
    ax.set_ylim(-env_radius - fig_margin, env_radius + fig_margin)
    ax.set_aspect('equal', adjustable='box')
    ax.ticklabel_format(axis='both', style='sci', scilimits=(1, 2))
    plt.tight_layout()
    plt.show()


# ----------
# Functions for Channel Generation
# ----------


# Pathloss (Free Space Path Loss)
def free_space_pathloss(d, fc, c=3e8):
    return (4 * np.pi * d * fc / c) ** 2


# Log-distance path-loss with shadowing
def log_distance_shadowing_pathloss(d, fc, pathloss_exponent=2.5, shadow_sigma_dB=4, d0=1):
    c = 3e8  # speed of light
    PL_d0 = 20 * np.log10(4 * np.pi * d0 * fc / c)  # Reference pathloss at d0
    PL_d = PL_d0 + 10 * pathloss_exponent * np.log10(d / d0)
    shadowing = np.random.normal(0, shadow_sigma_dB)
    PL_total = PL_d + shadowing
    PL_total_linear = 10 ** (PL_total / 10)
    return PL_total_linear


# UCA array response function
def uca_array_response(M_t, uca_radius, wavelength, azimuth_deg, elevation_deg=90):
    k = 2 * np.pi / wavelength
    azimuth_rad = np.deg2rad(azimuth_deg)
    elevation_rad = np.deg2rad(elevation_deg)
    phi_n = 2 * np.pi * np.arange(M_t) / M_t
    response = np.exp(1j * k * uca_radius * np.sin(elevation_rad) * np.cos(azimuth_rad - phi_n))
    return response


# Spatial correlation matrix using UCA
def spatial_correlation_UCA(M_t, uca_radius, wavelength, azimuth_deg, angle_spread_deg, elevation_deg=90, samples=100,
                            epsilon=1e-6, spread_dimension='azimuth'):
    R = np.zeros((M_t, M_t), dtype=complex)
    if spread_dimension == 'none':
        R = np.eye(M_t) * samples
    elif spread_dimension == 'azimuth':
        angles = azimuth_deg + np.random.uniform(-angle_spread_deg / 2, angle_spread_deg / 2, samples)
        for ang in angles:
            a = uca_array_response(M_t, uca_radius, wavelength, ang, elevation_deg)
            R += np.outer(a, a.conj())
    elif spread_dimension == 'elevation':
        angles = elevation_deg + np.random.uniform(-angle_spread_deg / 2, angle_spread_deg / 2, samples)
        for ang in angles:
            a = uca_array_response(M_t, uca_radius, wavelength, azimuth_deg, ang)
            R += np.outer(a, a.conj())
    elif spread_dimension == 'both':
        azimuths = azimuth_deg + np.random.uniform(-angle_spread_deg / 2, angle_spread_deg / 2, samples)
        elevations = elevation_deg + np.random.uniform(-angle_spread_deg / 2, angle_spread_deg / 2, samples)
        for az, el in zip(azimuths, elevations):
            a = uca_array_response(M_t, uca_radius, wavelength, az, el)
            R += np.outer(a, a.conj())
    else:
        raise ValueError("Invalid spread dimension. Must be one of 'azimuth', 'elevation', 'none, or 'both'.")
    R /= samples
    R += epsilon * np.eye(M_t)  # regularize for numerical stability
    return R


# Rician factor based on distance
def distance_based_rician_factor(d, K0_dB=10, d0=1, n=1):
    K_dB = K0_dB - 10 * n * np.log10(d / d0)
    K_linear = 10 ** (K_dB / 10)
    return max(K_linear, 0)


# Printing parameters
def print_parameters(params, length=30, line='='):
    print(line * length)
    print(f'{"Simulation Parameters":^{length}}')
    print('-' * length)
    for key, value in params.items():
        if isinstance(value, float):
            print(f'{key:<{int(length / 2)}}: {value:.2e}')
        elif isinstance(value, np.ndarray):
            array_str = np.array2string(value, formatter={'float_kind': lambda x: f'{x:.2e}'})
            print(f'{key:<{int(length / 2)}}: {array_str}')
        else:
            print(f'{key:<{int(length / 2)}}: {value}')
    print(line * length)


# FUNCTION FOR GENERATING CHANNELS FOR EACH AP
def generate_channels_singleAP(M_t, N_ue, ap_position, ue_positions, h_ap, h_ue, uca_radius=0.5,
                               angle_spread_deg=5, fc=3.5e9, pathloss_exponent=2.5, shadow_sigma_dB=4, K0_dB=10, c=3e8,
                               spatialCorr_nsamps=100, spatialCorr_dim='azimuth', mode='rician'):
    # Constants
    wavelength = c / fc
    # Value/Vector
    angle_spread_deg = np.atleast_1d(angle_spread_deg)
    angle_spread_deg_vector = np.resize(angle_spread_deg, N_ue)
    pathloss_exponent = np.atleast_1d(pathloss_exponent)
    pathloss_exponent_vector = np.resize(pathloss_exponent, N_ue)
    shadow_sigma_dB = np.atleast_1d(shadow_sigma_dB)
    shadow_sigma_dB_vector = np.resize(shadow_sigma_dB, N_ue)
    # Relative Positions
    ue_ap_distances = np.linalg.norm(ue_positions - np.tile(ap_position.reshape(2, 1), ue_positions.shape[1]), axis=0)
    ue_ap_angles_deg = np.rad2deg(np.arctan2(ue_positions[1] - ap_position[1], ue_positions[0] - ap_position[0]))
    ue_ap_angles_deg = (ue_ap_angles_deg + 360) % 360
    ue_elevation_angles_deg = np.rad2deg(np.arctan2(ue_ap_distances, h_ap - h_ue))
    ue_ap_distances_3D = np.sqrt(ue_ap_distances ** 2 + (h_ap - h_ue) ** 2)

    channels = np.zeros((N_ue, M_t), dtype=complex)
    for k in range(N_ue):
        dist_k = ue_ap_distances_3D[k]
        azimuth_k = ue_ap_angles_deg[k]
        elevation_k = ue_elevation_angles_deg[k]
        angle_spread_k = angle_spread_deg_vector[k]
        pl_exponent_k = pathloss_exponent_vector[k]
        sigma_k = shadow_sigma_dB_vector[k]

        R_k = spatial_correlation_UCA(M_t, uca_radius, wavelength, azimuth_deg=azimuth_k, angle_spread_deg=angle_spread_k,
                                      elevation_deg=elevation_k, samples=spatialCorr_nsamps, spread_dimension=spatialCorr_dim)
        L_chol = np.linalg.cholesky(R_k)

        # Compute large-scale fading
        path_loss_linear = log_distance_shadowing_pathloss(d=dist_k, fc=fc, pathloss_exponent=pl_exponent_k,
                                                           shadow_sigma_dB=sigma_k)
        large_scale_fading = 1 / path_loss_linear

        # Rician factor
        if mode == 'rayleigh':
            K_factor = 0
        elif mode == 'rician':
            K_factor = distance_based_rician_factor(dist_k, K0_dB=K0_dB, d0=1, n=1)
        else:
            raise ValueError("Invalid channel mode. Must be one of 'rician' or 'rayleigh'.")

        if K_factor > 0:
            # LoS component
            los_part = uca_array_response(M_t, uca_radius, wavelength, azimuth_k, elevation_k)
            nlos_part = L_chol @ (np.random.randn(M_t) + 1j * np.random.randn(M_t)) / np.sqrt(2)
            g_k = np.sqrt(K_factor / (K_factor + 1)) * los_part + np.sqrt(1 / (K_factor + 1)) * nlos_part
        else:
            # Rayleigh fading
            g_k = (np.random.randn(M_t) + 1j * np.random.randn(M_t)) / np.sqrt(2)
        channels[k, :] = np.sqrt(large_scale_fading) * g_k

    return channels


def simulation_print_statement(start_time=None, end_time=None, width=40, start=False, end=False):
    if start:
        # Print simulation start
        print("=" * width)
        print(">> Simulation Started <<".center(width))
        print("-" * width)
        print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * width)
    if end:
        # Print simulation end
        duration = end_time - start_time
        print("=" * width)
        print(">> Simulation Completed <<".center(width))
        print("-" * width)
        print(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Duration: {duration:.2f} seconds")
        print("=" * width)

def splitOpt_sinr2gamm(TxSNR, sinr_const_dB):
    """
    Converts splitOpt SINR constraint to gamma value.

    Parameters:
        TxSNR (float): Transmit SNR in dB (P_max / P_noise).
        sinr_const_dB (float): SINR constraint in dB.

    Returns:
        float: Gamma value.
    """
    gamma = 10 ** ((sinr_const_dB - TxSNR) / 10)
    return gamma

def plot_splitOpt_history(history, title='SplitOpt History', xlabel='Iteration', ylabel='Objective Value'):
    """
    Plots the history of the SplitOpt optimization process.

    Parameters:
        history (ndarray): An array of tuples containing (mu, slack, objective_value).
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    xi_values = [entry[0] for entry in history]
    slack_values = [entry[1] for entry in history]
    objective_values = [entry[2] for entry in history]

    plt.figure(figsize=(6, 4))
    plt.plot(xi_values, label='Xi', marker='o')
    plt.plot(slack_values, label='Slack', marker='x')
    plt.plot(objective_values, label='Objective Value', marker='s')
    plt.xticks(np.arange(len(xi_values)))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_PSRs(PSR_vec, title='Power Splitting Ratios (PSRs)', xlabel='AP Index', ylabel='PSR Value'):
    """
    Plots the PSR values for each AP.

    Parameters:
        PSR_vec (ndarray): Array of PSR values for each AP.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(len(PSR_vec)), PSR_vec)
    plt.xticks(np.arange(len(PSR_vec)))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()


def save_sim_results(results_dict, prefix='simulation_results', save_dir='../../results/'):
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
    filename = f"{save_dir}{timestamp}_{prefix}.npz"
    np.savez(filename, **results_dict)
    print(f"Saved to {filename}")
