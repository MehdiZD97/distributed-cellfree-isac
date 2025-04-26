
import numpy as np
from src.utils import library as lib
from src.utils.simulation import SimulationParameters, WirelessSimulation

params = SimulationParameters(N_ap=4, N_ue=8, M_t=10, TxSNR=30,
                              comm_K0_dB=10, comm_shadow_sigma_dB=10,
                              sens_K0_dB=20, sens_shadow_sigma_dB=15, sens_angle_spread_deg=5,
                              sens_spatialCorr_dim='azimuth',
                              rcs_variance=np.ones(4) / 2, seed=None)
sim = WirelessSimulation(params)

width = 30
print("-" * width)
print(">> Simulation Started <<".center(width))
print("-" * width)



# Simulation parameters
np.random.seed(2)
SAVE_RESULTS = False
exp_folder = 'exp1/data/'
num_random_topologies = 1000
sim_seeds = np.random.choice(10000, num_random_topologies, replace=False)
locate_target = True    # False is target is fixed
target_coords = [250, 0]  # target coordinates for fixed target
splitOpt_sinr_dB = 24  # Constraint for comm SINR
splitOpt_gamma = lib.splitOpt_sinr2gamm(params.TxSNR, splitOpt_sinr_dB)
splitOpt_min_rho = 0.0
splitOpt_max_rho = 1.0

fixed_PSR_vec = np.array([0.2, 0.8, -1])

for psr_val_idx in range(len(fixed_PSR_vec)):
    fixed_PSR = fixed_PSR_vec[psr_val_idx]
    optimizePA = True if fixed_PSR == -1 else False
    print_psr_val = 'optPA' if optimizePA else f'value {fixed_PSR:.2f}'
    print(f'** PSR {print_psr_val} & LocTar {locate_target} **'.center(width))

    comm_sinrs = np.zeros((len(sim_seeds), params.N_ue))
    sens_snrs = np.zeros(len(sim_seeds))
    sens_only_snrs = np.zeros(len(sim_seeds))

    for randTopology_idx in range(len(sim_seeds)):
        if (randTopology_idx+1) % 100 == 0:
            print(f'>> Simulations for PSR {print_psr_val} with topology {randTopology_idx+1}')
        params.seed = sim_seeds[randTopology_idx]
        sim.generate_topology(locate_target=locate_target)
        if not locate_target:
            sim.set_position(obj_typy='target', obj_idx=0, coords=target_coords)
        # sim.plot_topology()
        if optimizePA:
            sim.set_splitOpt_params(gamma=splitOpt_gamma, min_rho=splitOpt_min_rho, max_rho=splitOpt_max_rho, xi=1.0, Psi=2, max_iters=10)
        else:
            sim.set_PSRs(np.ones(params.N_ap) * fixed_PSR)
        sim.run_simulation(optimizePA=optimizePA, sim_log=False, print_log=True)
        comm_sinrs[randTopology_idx, :] = sim.outputs.comm_user_sinr
        sens_snrs[randTopology_idx] = sim.outputs.sens_total_snr
        sens_only_snrs[randTopology_idx] = sim.outputs.sens_only_snr

    results_dict = {
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


    if optimizePA:
        extra_prefix = 'optPA_randTargetLoc' if locate_target else 'optPA_fixedTargetLoc'
    else:
        psr_val = int(fixed_PSR * 10)
        extra_prefix = f'fixedPSR0p{psr_val}_randTargetLoc' if locate_target else f'fixedPSR0p{psr_val}_fixedTargetLoc'
    # Save results
    if SAVE_RESULTS:
        lib.save_sim_results(results_dict, prefix=f'splitOpt_{extra_prefix}_sinrSNR_{splitOpt_sinr_dB}_{params.TxSNR}dB', save_dir='../../results/' + exp_folder)

print("-" * width)
print(">> Simulation finished <<".center(width))
print("-" * width)

