# ----------
# Simulation Parameters
# ----------

import src.utils.library as lib
from dataclasses import dataclass, field
import numpy as np
import time
import cvxpy as cp
import warnings



@dataclass
class SimulationParameters:
    # Network Parameters
    P_max: float = 1  # Maximum transmit power (W)
    P_noise: float = None  # Noise power (W)
    TxSNR: float = 30  # Transmit SNR (dB) = P_max / P_noise
    N_ap: int = 4  # Number of APs
    N_ue: int = 4  # Number of UEs
    M_t: int = 8  # Number of antennas at each AP
    N_target: int = 1  # Number of targets
    h_ap: float = 10  # Height of APs (m)
    h_ue: float = 1.5  # Height of UEs (m)
    h_target: float = 0.5  # Height of Target (m)
    env_radius: float = 1000  # Radius of the circular environment (m)
    ap_radius: float = 650  # Radius of the circle of APs (m)
    c: float = 3e8  # Speed of light (m/s)
    fc: float = 3.5e9  # Carrier frequency (Hz)
    wavelength: float = c / fc  # Wavelength (m)
    uca_radius: float = M_t * wavelength / (4 * np.pi)  # Radius of the UCA (m)
    # Communication Parameters
    comm_K0_dB: float = 10  # Rician factor at reference distance (dB)
    comm_angle_spread_deg: float = 5  # Angle spread of the UCA (deg)
    comm_pathloss_exponent: float = 2.5  # Pathloss exponent
    comm_shadow_sigma_dB: float = 4  # Shadowing standard deviation (dB)
    comm_spatialCorr_dim: str = 'none'  # Spatial correlation dimension ('azimuth', 'elevation', 'none', or 'both')
    comm_channel_mode: str = 'rician'  # Channel mode ('rician' or 'rayleigh')
    # Sensing Parameters
    sens_K0_dB: float = 40  # Rician factor at reference distance (dB)
    sens_angle_spread_deg: float = 5  # Angle spread of the UCA (deg)
    sens_pathloss_exponent: float = 2.5  # Pathloss exponent
    sens_shadow_sigma_dB: float = 2  # Shadowing standard deviation (dB)
    sens_spatialCorr_dim: str = 'azimuth'  # Spatial correlation dimension ('azimuth', 'elevation', 'none', or 'both')
    sens_channel_mode: str = 'rician'  # Channel mode ('rician' or 'rayleigh')
    sensing_bf_mode = 'ns_zf'  # Beamforming mode for sensing ('conj': conjugate, 'ns_svd': null-space based on SVD, 'ns_zf': null-space based on ZF beam)
    rcs_variance: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    seed: int = None  # Random seed for reproducibility


@dataclass
class SimulationOutput:
    comm_user_sinr: np.ndarray = field(default_factory=lambda: np.array([]))
    comm_avg_sinr: float = None
    sens_total_snr: np.ndarray = field(default_factory=lambda: np.array([]))
    sens_only_snr: np.ndarray = field(default_factory=lambda: np.array([]))
    beamforming_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    PSRs: np.ndarray = field(default_factory=lambda: np.array([]))
    splitOpt_history: np.ndarray = field(default_factory=lambda: np.array([]))
    jointOpt_history: np.ndarray = field(default_factory=lambda: np.array([]))
    admm_status: str = None


@dataclass
class SplitOptimizationParams:
    gamma: float = 0.1  # Minimum SINR requirement
    min_rho: float = 0.0  # Minimum PSR
    max_rho: float = 1.0  # Maximum PSR
    xi: float = 1.0  # penalty parameter
    Psi: float = 5.0  # multiplier for increasing xi
    slack_tol: float = 1e-4  # acceptable slack tolerance
    max_iters: int = 15


@dataclass
class JointOptimizationParams:
    lambda_: float = 0.7  # objective trade-off weight
    rho: float = 0.5  # ADMM penalty parameter
    eta_share: np.ndarray = field(default_factory=lambda: np.array([]))  # per-AP SINR share
    xi_slack: float = 1e1 / 2   # slack penalty parameter
    sca_iters: int = 5  # number of iterations for SCA
    admm_iters: int = 35  # number of iterations for ADMM
    primal_thresh: float = 1e-3  # primal convergence threshold
    gamma_a_init: np.ndarray = field(default_factory=lambda: np.array([1e-3]))  # initial per-AP gamma values
    v_dual_init: np.ndarray = field(default_factory=lambda: np.array([0.0]))  # initial dual variables
    cent_iters: int = 25    # number of iterations for centralized solution
    cent_conv_thresh: float = 1e-4  # convergence threshold for centralized solution



# ----------
# Simulation Class
# ----------

class WirelessSimulation:
    def __init__(self, params: SimulationParameters):
        self.params = params
        if self.params.P_noise is None:
            TxSNR_linear = 10 ** (self.params.TxSNR / 10)
            self.params.P_noise = self.params.P_max / TxSNR_linear
        self.params.rcs_variance = np.resize(self.params.rcs_variance, self.params.N_ap)
        self.ap_positions = None
        self.ue_positions = None
        self.target_position = None
        self.sensingRxAP_index = None
        self.PSR_vec = np.ones(self.params.N_ap)/2
        self.outputs = SimulationOutput()
        self.outputs.PSRs = self.PSR_vec
        self.splitOpt_params = SplitOptimizationParams()
        self.jointOpt_params = JointOptimizationParams()
        if np.size(self.jointOpt_params.eta_share) == 0:
            self.jointOpt_params.eta_share = np.ones(self.params.N_ap) / self.params.N_ap
        else:
            self.jointOpt_params.eta_share = np.resize(self.jointOpt_params.eta_share, self.params.N_ap)
        self.jointOpt_params.gamma_a_init = np.resize(self.jointOpt_params.gamma_a_init, self.params.N_ap)
        self.jointOpt_params.v_dual_init = np.resize(self.jointOpt_params.v_dual_init, self.params.N_ap)

    def generate_topology(self, locate_target=True):
        if self.ap_positions is None:
            self.ap_positions = lib.generate_AP_positions(self.params.N_ap, self.params.ap_radius)
        self.ue_positions = lib.generate_UE_positions(self.params.N_ue, self.params.env_radius, random_state=self.params.seed)
        if locate_target or self.target_position is None:
            self.target_position = lib.generate_target_positions(self.params.env_radius, N_target=self.params.N_target, random_state=self.params.seed)
        self.sensingRxAP_index = lib.find_sensingRxAP(self.ap_positions, self.target_position)

    def set_position(self, obj_typy=None, obj_idx=None, coords=None):
        if self.ap_positions is None or self.ue_positions is None:
            raise ValueError("Topology not generated. Call generate_topology() first.")
        if obj_typy == 'AP':
            if obj_idx >= self.params.N_ap:
                raise ValueError("AP index out of range.")
            self.ap_positions[obj_idx][0] = coords[0]
            self.ap_positions[obj_idx][1] = coords[1]
        elif obj_typy == 'UE':
            if obj_idx >= self.params.N_ue:
                raise ValueError("UE index out of range.")
            self.ue_positions[0][obj_idx] = coords[0]
            self.ue_positions[1][obj_idx] = coords[1]
        elif obj_typy == 'target':
            if self.params.N_target > 1:
                if obj_idx >= self.params.N_target:
                    raise ValueError("Target index out of range.")
                self.target_position[obj_idx][0] = coords[0]
                self.target_position[obj_idx][1] = coords[1]
            else:
                self.target_position[0] = coords[0]
                self.target_position[1] = coords[1]
        else:
            raise ValueError("Invalid object type. Must be 'AP', 'UE', or 'target'.")

    def plot_topology(self):
        if self.ap_positions is None or self.ue_positions is None:
            raise ValueError("Topology not generated. Call generate_topology() first.")
        lib.plot_topology(self.ap_positions, self.ue_positions, self.params.env_radius,
                          target_position=self.target_position)

    def generate_comm_channel(self, print_log=False):
        # Generating comm channel
        H_mat_comm = np.zeros((self.params.N_ue, self.params.M_t, self.params.N_ap), dtype=complex)
        for a in range(self.params.N_ap):
            if a == self.sensingRxAP_index:
                continue
            angle_spreads_a = np.random.normal(self.params.comm_angle_spread_deg, 0.1, self.params.N_ue)
            angle_spreads_a = np.clip(angle_spreads_a, 0, 15)
            pl_exponents_a = np.random.normal(self.params.comm_pathloss_exponent, 0.1, self.params.N_ue)
            pl_exponents_a = np.clip(pl_exponents_a, 2.5, 3.5)
            shadow_sigmas_a = np.random.normal(self.params.comm_shadow_sigma_dB, 1, self.params.N_ue)
            shadow_sigmas_a = np.clip(shadow_sigmas_a, 3, 7)
            ap_a = self.ap_positions[:, a]
            H_a = lib.generate_channels_singleAP(self.params.M_t, self.params.N_ue, ap_position=ap_a,
                                                 ue_positions=self.ue_positions,
                                                 h_ap=self.params.h_ap, h_ue=self.params.h_ue,
                                                 uca_radius=self.params.uca_radius, angle_spread_deg=angle_spreads_a,
                                                 fc=self.params.fc, pathloss_exponent=pl_exponents_a,
                                                 shadow_sigma_dB=shadow_sigmas_a,
                                                 K0_dB=self.params.comm_K0_dB, spatialCorr_nsamps=1000,
                                                 spatialCorr_dim=self.params.comm_spatialCorr_dim,
                                                 mode=self.params.comm_channel_mode)
            H_a_norm = H_a / np.linalg.norm(H_a, axis=1).reshape(-1, 1)
            H_mat_comm[:, :, a] = H_a_norm
        if print_log:
            print('Comm Channels Generated with the Shape:')
            print('(N_ue, M_t, N_ap)')
            print(H_mat_comm.shape)
        return H_mat_comm

    def generate_sens_channel(self, print_log=False):
        # Generating sensing channel
        rx_ap_position = self.ap_positions[:, self.sensingRxAP_index]
        target_position = self.target_position
        h_R = lib.generate_channels_singleAP(self.params.M_t, self.params.N_target, ap_position=rx_ap_position,
                                             ue_positions=target_position.reshape(2, 1),
                                             h_ap=self.params.h_ap, h_ue=self.params.h_target,
                                             uca_radius=self.params.uca_radius,
                                             angle_spread_deg=self.params.sens_angle_spread_deg, fc=self.params.fc,
                                             shadow_sigma_dB=self.params.sens_shadow_sigma_dB,
                                             K0_dB=self.params.sens_K0_dB,
                                             spatialCorr_dim=self.params.sens_spatialCorr_dim,
                                             mode=self.params.sens_channel_mode)
        h_R = h_R.T / np.linalg.norm(h_R)
        H_mat_sens = np.zeros((self.params.M_t, self.params.M_t, self.params.N_ap), dtype=complex)
        H_T_sens = np.zeros((self.params.M_t, self.params.N_ap), dtype=complex)
        for a in range(self.params.N_ap):
            if a == self.sensingRxAP_index:
                continue
            tx_ap_position = self.ap_positions[:, a]
            h_T = lib.generate_channels_singleAP(self.params.M_t, self.params.N_target, ap_position=tx_ap_position,
                                                 ue_positions=target_position.reshape(2, 1),
                                                 h_ap=self.params.h_ap, h_ue=self.params.h_target,
                                                 uca_radius=self.params.uca_radius,
                                                 angle_spread_deg=self.params.sens_angle_spread_deg, fc=self.params.fc,
                                                 shadow_sigma_dB=self.params.sens_shadow_sigma_dB,
                                                 K0_dB=self.params.sens_K0_dB,
                                                 spatialCorr_dim=self.params.sens_spatialCorr_dim,
                                                 mode=self.params.sens_channel_mode)
            h_T = h_T.T / np.linalg.norm(h_T)
            H_T_sens[:, a] = np.squeeze(h_T)
            H_a_target = np.random.normal(0, self.params.rcs_variance[a]) * (h_R @ h_T.T)
            H_mat_sens[:, :, a] = H_a_target
        if print_log:
            print('Sensing Channels Generated with the Shape:')
            print('(M_r, M_t, N_ap)')
            print(H_mat_sens.shape)
        return H_mat_sens, H_T_sens

    def generate_fixed_precoders(self, H_mat_comm, print_log=False):
        C = np.zeros((self.params.N_ue, self.params.N_ue + 1), dtype=complex)
        C[:, :-1] = np.identity(self.params.N_ue, dtype=complex)

        weights = lib.decompose_C(C, H_mat_comm, mode='condition_number')
        alpha_prime_vec = np.zeros(self.params.N_ap)
        Delta_prime_vec = np.zeros(self.params.N_ap)
        if print_log:
            print('decomposition weights: [{}]'.format(', '.join('{:.4f}'.format(w) for w in weights)))

        W_hat_mat = np.zeros((self.params.M_t, self.params.N_ue + 1, self.params.N_ap), dtype=complex)
        for a in range(self.params.N_ap):
            if a == self.sensingRxAP_index:
                continue
            ap_a = self.ap_positions[:, a]
            target_distance = np.linalg.norm(self.target_position - ap_a, axis=0)
            target_angle_deg = np.rad2deg(
                np.arctan2(self.target_position[1] - ap_a[1], self.target_position[0] - ap_a[0]))
            target_angle_deg = (target_angle_deg + 360) % 360
            target_elevation_deg = np.rad2deg(np.arctan2(target_distance, self.params.h_ap - self.params.h_target))

            target_dir = lib.uca_array_response(self.params.M_t, self.params.uca_radius, self.params.wavelength,
                                                target_angle_deg,
                                                target_elevation_deg)
            # target_dir = target_dir / np.linalg.norm(target_dir)
            d_a = target_dir.reshape(-1, 1)

            H_a = H_mat_comm[:, :, a]
            H_H = H_a.conj().T @ H_a
            H_a_plus = np.linalg.pinv(H_H) @ H_a.conj().T
            W_a = np.zeros((self.params.M_t, self.params.N_ue + 1), dtype=complex)
            if self.params.sensing_bf_mode == 'conj':
                W_a[:, -1] = np.squeeze(d_a)  # conjugate beam directly toward target
            elif self.params.sensing_bf_mode == 'ns_svd':
                numNullVecs = int(self.params.M_t / 4)
                v_nulls = lib.return_null_vectors(H_mat_comm, numVecs=numNullVecs)
                if numNullVecs == 1:
                    v_nulls = v_nulls.reshape(v_nulls.shape[0], v_nulls.shape[1], 1)
                v_nulls_mat = v_nulls[a, :, :]
                P_null_a = v_nulls_mat @ v_nulls_mat.conj().T
                d_prime_a = P_null_a @ d_a
                d_hat_a = d_prime_a / np.linalg.norm(d_prime_a)
                W_a[:, -1] = np.squeeze(d_hat_a)  # projecting on the null space using SVD
            elif self.params.sensing_bf_mode == 'ns_zf':
                H_a_null = H_a.conj().T @ np.linalg.pinv(H_a @ H_a.conj().T) @ H_a
                ns_beam = (np.eye(self.params.M_t) - H_a_null) @ d_a
                ns_beam = ns_beam / np.linalg.norm(ns_beam)
                W_a[:, -1] = np.squeeze(ns_beam)  # null-space beam using comm ZF beam
            else:
                raise ValueError("Invalid sensing beamforming mode. Must be one of 'conj', 'ns_svd', or 'ns_zf'.")
            if print_log:
                alignment = np.abs(np.dot(W_a[:, -1].conj().T, target_dir/np.linalg.norm(target_dir)))
                print(f'Alignment for AP {a}: {alignment:.4f}')
            delta_s = np.linalg.norm(np.dot(d_a.conj().T, W_a[:, -1]))**2
            W_a_comm = H_a_plus @ (weights[a] * C[:, :-1])
            W_a[:, :-1] = W_a_comm / np.linalg.norm(W_a_comm, 'fro')
            alpha_prime_vec[a] = weights[a] / np.linalg.norm(W_a_comm, 'fro')
            delta_c = np.linalg.norm(d_a.conj().T @ (W_a_comm / np.linalg.norm(W_a_comm, 'fro')))**2
            Delta_a = delta_c - delta_s
            Delta_prime_vec[a] = Delta_a * self.params.rcs_variance[a]
            if print_log:
                print(f'Comm Power: {np.sum(np.linalg.norm(W_a[:, :-1], axis=0) ** 2):.4f}')
            W_hat_mat[:, :, a] = W_a
        return W_hat_mat, alpha_prime_vec, Delta_prime_vec

    def set_splitOpt_params(self, gamma=None, min_rho=None, max_rho=None, xi=None, Psi=None, slack_tol=None, max_iters=None):
        if gamma is not None:       self.splitOpt_params.gamma = gamma
        if min_rho is not None:     self.splitOpt_params.min_rho = min_rho
        if max_rho is not None:     self.splitOpt_params.max_rho = max_rho
        if xi is not None:          self.splitOpt_params.xi = xi
        if Psi is not None:         self.splitOpt_params.Psi = Psi
        if slack_tol is not None:   self.splitOpt_params.slack_tol = slack_tol
        if max_iters is not None:   self.splitOpt_params.max_iters = max_iters

    def find_optimal_PSRs(self, alpha_prime_vec, Delta_prime_vec, print_log=False):
        alpha_prime_vec_t = np.delete(alpha_prime_vec, self.sensingRxAP_index)
        Delta_prime_vec_t = np.delete(Delta_prime_vec, self.sensingRxAP_index)

        history = []
        rho = None
        problem = None
        for t in range(self.splitOpt_params.max_iters):
            # Opt Variables (PSR and slack)
            rho = cp.Variable(self.params.N_ap - 1)
            s = cp.Variable(nonneg=True)
            # Constraint: (sum alpha_i * sqrt(rho_i)) + s >= sqrt(gamma)
            comm_constraint = cp.sum(cp.multiply(alpha_prime_vec_t, cp.sqrt(rho))) + s >= np.sqrt(
                self.splitOpt_params.gamma)
            constraints = [comm_constraint, rho >= self.splitOpt_params.min_rho, rho <= self.splitOpt_params.max_rho]
            # Objective: max sensing utility - xi * slack
            objective = cp.Maximize(Delta_prime_vec_t @ rho - self.splitOpt_params.xi * s)
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.ECOS)
            # Log
            history.append((self.splitOpt_params.xi, s.value, problem.value))
            # Check if slack is negligible
            if s.value <= self.splitOpt_params.slack_tol:
                break
            self.splitOpt_params.xi += self.splitOpt_params.Psi * (t+1)
        # Output
        if print_log:
            print("Optimal rho:", rho.value)
            print("Objective value:", problem.value)
        if problem.status != cp.OPTIMAL:
            warnings.warn("Optimization failed. Using default PSR values.")
        else:
            for a in range(self.params.N_ap):
                if a == self.sensingRxAP_index:
                    continue
                self.PSR_vec[a] = rho.value[a - (a > self.sensingRxAP_index)]
        self.PSR_vec = np.clip(self.PSR_vec, self.splitOpt_params.min_rho, self.splitOpt_params.max_rho)
        self.outputs.PSRs = self.PSR_vec
        self.outputs.splitOpt_history = np.array(history)

    def set_PSRs(self, PSR_vec):
        if len(PSR_vec) != self.params.N_ap:
            raise ValueError("PSR vector length must match the number of APs.")
        self.PSR_vec = np.clip(PSR_vec, self.splitOpt_params.min_rho, self.splitOpt_params.max_rho)
        self.outputs.PSRs = self.PSR_vec

    def scale_precoders(self, W_hat_mat):
        P_comm_linear = self.PSR_vec * self.params.P_max
        P_sensing_linear = (1 - self.PSR_vec) * self.params.P_max
        W_mat = np.zeros((self.params.M_t, self.params.N_ue + 1, self.params.N_ap), dtype=complex)
        for a in range(self.params.N_ap):
            if a == self.sensingRxAP_index:
                continue
            W_a = W_hat_mat[:, :, a]
            W_a[:, :-1] = W_a[:, :-1] * np.sqrt(P_comm_linear[a])
            W_a[:, -1] = W_a[:, -1] * np.sqrt(P_sensing_linear[a])
            W_mat[:, :, a] = W_a
        return W_mat

    def set_jointOpt_params(self, lambda_=None, rho=None, eta_share=None, xi_slack=None, sca_iters=None, admm_iters=None,
                            primal_thresh=None, gamma_a_init=None, v_dual_init=None, cent_iters=None, cent_conv_thresh=None):
        if lambda_ is not None:             self.jointOpt_params.lambda_ = lambda_
        if rho is not None:                 self.jointOpt_params.rho = rho
        if eta_share is not None:           self.jointOpt_params.eta_share = np.resize(eta_share, self.params.N_ap)
        if xi_slack is not None:            self.jointOpt_params.xi_slack = xi_slack
        if sca_iters is not None:           self.jointOpt_params.sca_iters = sca_iters
        if admm_iters is not None:          self.jointOpt_params.admm_iters = admm_iters
        if primal_thresh is not None:       self.jointOpt_params.primal_thresh = primal_thresh
        if gamma_a_init is not None:        self.jointOpt_params.gamma_a_init = np.resize(gamma_a_init, self.params.N_ap)
        if v_dual_init is not None:         self.jointOpt_params.v_dual_init = np.resize(v_dual_init, self.params.N_ap)
        if cent_iters is not None:          self.jointOpt_params.cent_iters = cent_iters
        if cent_conv_thresh is not None:    self.jointOpt_params.cent_conv_thresh = cent_conv_thresh

    def jointOpt_ADMM(self, H_mat_comm, H_T_sens, W_mat, print_log=False):
        # Optimization parameters
        P_noise = self.params.P_noise
        lambda_ = self.jointOpt_params.lambda_
        rho = self.jointOpt_params.rho
        eta_share = self.jointOpt_params.eta_share
        xi_slack = self.jointOpt_params.xi_slack
        sca_iters = self.jointOpt_params.sca_iters
        admm_iters = self.jointOpt_params.admm_iters
        primal_thresh = self.jointOpt_params.primal_thresh

        # ==== Initializing H, W, and steering vectors ====
        H_comm = H_mat_comm
        A_target = H_T_sens
        W = W_mat
        # ==== Initializing interferences I ====
        I_users = np.zeros(self.params.N_ue)
        H_W_admm = self.calc_product_H_W(H_comm, W)
        for u in range(self.params.N_ue):
            I_users[u] = (np.linalg.norm(H_W_admm[u, :]) ** 2) - (np.abs(H_W_admm[u, u]) ** 2)
        # ==== Initializing gamma and dual variable ====
        gamma_a = self.jointOpt_params.gamma_a_init
        v_dual = self.jointOpt_params.v_dual_init
        gamma = np.mean(gamma_a)

        admm_history = []
        self.outputs.admm_status = 'MaxIterations'
        # ==== ADMM loop ====
        for it in range(admm_iters):
            # ==== Local updates with inner SCA loop ====
            for a in range(self.params.N_ap):
                for sca_t in range(sca_iters):
                    # Linearization parameters of objective (obj) and per-user SINR constraint (con)
                    g_obj_a = A_target[:, a].conj().T @ W[:, :, a]
                    g_con_a = np.zeros(self.params.N_ue, dtype=complex)  # only for one AP is available
                    for u in range(self.params.N_ue):
                        h_au = H_comm[u, :, a]
                        w_au = W[:, u, a]
                        g_con_a[u] = h_au.T @ w_au

                    # CVXPY variables
                    W_a = cp.Variable((self.params.M_t, self.params.N_ue + 1), complex=True)
                    gamma_a_var = cp.Variable()
                    s_users = cp.Variable(self.params.N_ue, nonneg=True)  # slack variable

                    # Linearized sensing utility
                    u_sens_lin = 2 * cp.real(g_obj_a @ (W_a.conj().T @ A_target[:, a])) - np.linalg.norm(g_obj_a) ** 2

                    # Constraints
                    cons = [cp.norm(W_a, 'fro') ** 2 <= self.params.P_max]
                    for u in range(self.params.N_ue):
                        sinr_lin = 2 * cp.real(g_con_a[u].conj() * H_comm[u, :, a].T @ W_a[:, u]) - np.linalg.norm(
                            g_con_a[u]) ** 2
                        I_u = I_users[u]
                        cons.append(sinr_lin + s_users[u] >= eta_share[a] * gamma_a_var * (
                                    I_u + np.sqrt(P_noise) ** 2))  # modify based on Minkowski's inequality

                    # Augmented Lagrangian term
                    aug = lambda_ * gamma_a_var + (1 - lambda_) * u_sens_lin - (rho / 2) * cp.square(
                        gamma_a_var - gamma + v_dual[a])

                    # Objective
                    obj = cp.Maximize(aug - xi_slack * cp.sum(s_users))
                    prob = cp.Problem(obj, cons)
                    prob.solve(solver=cp.ECOS)

                    # Update W and gamma
                    W[:, :, a] = W_a.value
                    gamma_a[a] = gamma_a_var.value

                    # Update interference
                    H_W_admm = self.calc_product_H_W(H_comm, W)
                    for u in range(self.params.N_ue):
                        I_users[u] = (np.linalg.norm(H_W_admm[u, :]) ** 2) - (np.abs(H_W_admm[u, u]) ** 2)

            # ==== Global consensus update ====
            gamma = np.mean(gamma_a + v_dual)
            primal_res = np.max(np.abs(gamma_a - gamma))
            if print_log: print(f'ADMM iter: {it} gamma: {gamma:.4f} primal res: {primal_res:.4f}')
            if gamma < 0:
                warnings.warn("--- Gamma is negative, stopping ADMM! ---")
                break

            # ==== Dual variable update ====
            v_dual += gamma_a - gamma

            admm_history.append((gamma, primal_res))
            # ==== Stopping criteria ====
            if primal_res < primal_thresh:
                if print_log: print('==**== ADMM converged ==**==')
                self.outputs.admm_status = 'Converged'
                break
        # ==== Final output ====
        W_star = W
        self.outputs.jointOpt_history = np.array(admm_history)
        return W_star

    def jointOpt_centralized(self, H_mat_comm, H_T_sens, W_mat, print_log=False):
        # Optimization parameters
        P_noise = self.params.P_noise
        lambda_ = self.jointOpt_params.lambda_
        xi_slack = self.jointOpt_params.xi_slack
        sca_iters = self.jointOpt_params.sca_iters
        cent_iters = self.jointOpt_params.cent_iters
        conv_thresh = self.jointOpt_params.cent_conv_thresh

        # ==== Initializing H, W, and steering vectors ====
        # H_comm shape: (N_ue, M_t, N_ap)
        H_comm = H_mat_comm
        A_target = H_T_sens
        # W shape: (M_t, N_ue+1, N_ap)
        W = W_mat

        # ==== Initializing interferences I ====
        I_users = np.zeros(self.params.N_ue)
        H_W_admm = self.calc_product_H_W(H_comm, W)
        for u in range(self.params.N_ue):
            I_users[u] = (np.linalg.norm(H_W_admm[u, :]) ** 2) - (np.abs(H_W_admm[u, u]) ** 2)

        gamma_prev = 1e-3
        gamma_new = 1e3

        # ==== Opt loop ====
        for it in range(cent_iters):
            # ==== Inner SCA loop ====
            for sca_t in range(sca_iters):
                # CVXPY variables
                W_var = cp.Variable(((self.params.N_ue + 1), self.params.M_t * self.params.N_ap), complex=True)
                gamma_var = cp.Variable()
                s_users = cp.Variable(self.params.N_ue, nonneg=True)  # slack variable

                # Linearization of sensing utility
                u_sens_lin = []
                for a in range(self.params.N_ap):
                    g_obj_a = A_target[:, a].conj().T @ W[:, :, a]
                    # Linearized sensing utility
                    W_a = W_var[:, a * self.params.M_t:(a + 1) * self.params.M_t].T
                    u_sens_lin.append(
                        2 * cp.real(g_obj_a @ (W_a.conj().T @ A_target[:, a])) - np.linalg.norm(g_obj_a) ** 2)
                u_sens_total = cp.sum(u_sens_lin)

                # Linearization of SINR numerator
                g_con_list = []
                for u in range(self.params.N_ue):
                    h_u = np.reshape(H_comm[u, :, :], (self.params.M_t * self.params.N_ap, 1), order='F')
                    w_u_t = np.reshape(W[:, u, :], (self.params.M_t * self.params.N_ap, 1), order='F')
                    w_u_var = W_var[u, :]
                    hw_t = h_u.T @ w_u_t
                    g_con_list.append(2 * cp.real(hw_t.conj() * h_u.T @ w_u_var) - np.linalg.norm(hw_t) ** 2)
                g_con = cp.vstack(g_con_list)

                # Constraints
                cons = []
                for u in range(self.params.N_ue):
                    I_u = I_users[u]
                    cons.append(g_con[u] + s_users[u] >= (
                                gamma_var * (I_u + P_noise)))  # modify based on Minkowski's inequality
                # Power constraint
                for a in range(self.params.N_ap):
                    W_a = W_var[:, a * self.params.M_t:(a + 1) * self.params.M_t].T
                    cons.append(cp.norm(W_a, 'fro') ** 2 <= self.params.P_max)

                # Original objective
                orig_objective = lambda_ * gamma_var + (1 - lambda_) * u_sens_total

                # Objective
                obj = cp.Maximize(orig_objective - xi_slack * cp.sum(s_users))
                prob = cp.Problem(obj, cons)
                prob.solve(solver=cp.ECOS)

                # Update W and gamma
                W_new = W_var.value
                W_new = np.reshape(W_new, (self.params.N_ue + 1, self.params.M_t, self.params.N_ap), order='F')
                W = np.transpose(W_new, axes=(1, 0, 2))
                gamma_new = gamma_var.value

                # Update interference
                H_W_admm = self.calc_product_H_W(H_mat_comm, W)
                for u in range(self.params.N_ue):
                    I_users[u] = (np.linalg.norm(H_W_admm[u, :]) ** 2) - (np.abs(H_W_admm[u, u]) ** 2)
            res = np.abs(gamma_prev - gamma_new)
            if print_log: print('Iter:', it, 'gamma:', gamma_new, 'residual:', res)
            # ==== Stopping criteria ====
            if res < conv_thresh:
                if print_log: print('==**== ADMM converged ==**==')
                break
            gamma_prev = gamma_new
        # ==== Final output ====
        W_star = W
        return W_star


    def calc_product_H_W(self, H_mat_comm, W_mat):
        H_W = np.zeros((self.params.N_ue, self.params.N_ue + 1), dtype=complex)
        for a in range(self.params.N_ap):
            if a == self.sensingRxAP_index:
                continue
            H_a = H_mat_comm[:, :, a]
            W_a = W_mat[:, :, a]
            H_W += H_a @ W_a
        return H_W

    def calculate_sinr(self, H_W, print_log=False):
        sinr_dB = []
        for k in range(self.params.N_ue):
            P_desired = np.abs(H_W[k, k]) ** 2
            P_interference = np.linalg.norm(H_W[k, :]) ** 2 - P_desired
            sinr = P_desired / (P_interference + self.params.P_noise)
            sinr_dB.append(10 * np.log10(sinr))
        sinr_dB = np.array(sinr_dB)
        if print_log:
            print('SINRs for Users: [{}]'.format(', '.join('{:.2f}'.format(s) for s in sinr_dB)))
        return sinr_dB

    def calculate_sensing_snr(self, H_T_sens, W_mat, print_log=False):
        # Sensing SNR
        P_sds = 0
        P_sds_sens = 0
        for a in range(self.params.N_ap):
            if a == self.sensingRxAP_index:
                continue
            h_T = H_T_sens[:, a].reshape(-1, 1)
            W_a = W_mat[:, :, a]
            P_sds += self.params.rcs_variance[a] * np.linalg.norm(h_T.conj().T @ W_a) ** 2
            P_sds_sens += self.params.rcs_variance[a] * np.linalg.norm(h_T.conj().T @ W_a[:, -1]) ** 2

        snr = (P_sds / self.params.P_noise) + 1e-10
        snr_dB = 10 * np.log10(snr)
        snr_sens = (P_sds_sens / self.params.P_noise) + 1e-10
        snr_sens_dB = 10 * np.log10(snr_sens)
        snr_dB_vec = np.array([snr_dB, snr_sens_dB])
        if print_log:
            print('Sensing SNR: {:.2f} dB'.format(snr_dB))
            print('Sensing SNR (Sensing Beam): {:.2f} dB'.format(snr_sens_dB))
        return snr_dB_vec

    def run_simulation(self, optimizePA=True, sim_log=False, print_log=False):
        if self.ap_positions is None or self.ue_positions is None:
            raise ValueError("Topology not generated. Call generate_topology() first.")
        start_time = time.time()
        if print_log or sim_log:
            lib.simulation_print_statement(start_time=start_time, start=True)
            print('Random Seed:', self.params.seed)
            print('Optimizing PAs:', optimizePA)
            print('Running Simulation...')
        H_mat_comm = self.generate_comm_channel(print_log=print_log)
        _, H_T_sens = self.generate_sens_channel(print_log=print_log)
        W_hat_mat, alpha_prime_vec, Delta_prime_vec = self.generate_fixed_precoders(H_mat_comm, print_log=print_log)
        if optimizePA:
            self.find_optimal_PSRs(alpha_prime_vec, Delta_prime_vec, print_log=print_log)
        W_mat = self.scale_precoders(W_hat_mat)
        self.outputs.beamforming_matrix = W_mat
        H_W = self.calc_product_H_W(H_mat_comm, W_mat)
        sinr_dB = self.calculate_sinr(H_W, print_log=print_log)
        self.outputs.comm_user_sinr = sinr_dB
        self.outputs.comm_avg_sinr = np.mean(sinr_dB)
        snr_dB_vec = self.calculate_sensing_snr(H_T_sens, W_mat, print_log=print_log)
        self.outputs.sens_total_snr = snr_dB_vec[0]
        self.outputs.sens_only_snr = snr_dB_vec[1]
        end_time = time.time()
        if print_log or sim_log:
            lib.simulation_print_statement(start_time=start_time, end_time=end_time, end=True)

    def run_simulation_jointOpt(self, sim_log=False, print_log=False):
        if self.ap_positions is None or self.ue_positions is None:
            raise ValueError("Topology not generated. Call generate_topology() first.")
        start_time = time.time()
        if print_log or sim_log:
            lib.simulation_print_statement(start_time=start_time, start=True)
            print('Random Seed:', self.params.seed)
            print('Running Simulation...')
        H_mat_comm = self.generate_comm_channel(print_log=print_log)
        _, H_T_sens = self.generate_sens_channel(print_log=print_log)
        W_hat_mat, _, _ = self.generate_fixed_precoders(H_mat_comm, print_log=print_log)
        W_mat = self.scale_precoders(W_hat_mat)
        W_star = self.jointOpt_ADMM(H_mat_comm, H_T_sens, W_mat, print_log=print_log)
        H_W = self.calc_product_H_W(H_mat_comm, W_star)
        sinr_dB = self.calculate_sinr(H_W, print_log=print_log)
        self.outputs.comm_user_sinr = sinr_dB
        self.outputs.comm_avg_sinr = np.mean(sinr_dB)
        snr_dB_vec = self.calculate_sensing_snr(H_T_sens, W_mat, print_log=print_log)
        self.outputs.sens_total_snr = snr_dB_vec[0]
        self.outputs.sens_only_snr = snr_dB_vec[1]
        end_time = time.time()
        if print_log or sim_log:
            lib.simulation_print_statement(start_time=start_time, end_time=end_time, end=True)


