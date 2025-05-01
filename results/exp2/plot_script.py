
import numpy as np
import matplotlib.pyplot as plt
from src.utils import library as lib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__name__))
if os.path.basename(BASE_DIR) == 'distributed-cellfree-isac':
    BASE_DIR = BASE_DIR + '/results/exp2/'
else:
    BASE_DIR = BASE_DIR + '/'
data_dir = 'data/'

SAVE_FIG_SC1 = False
SAVE_FIG_SC2 = False
SAVE_FIG_JOINT = False

#%% Loading the data

topolobies = 50
TxSNR = 20

hour, minute = '02', '36'
filename = f'2025_04_25_{hour}_{minute}_jointOpt_Nue{1}_{topolobies}Topos_TxSNR{TxSNR}dB.npz'
loaded = np.load(BASE_DIR + data_dir + filename, allow_pickle=True)
results_dict_Nue1 = dict(loaded.items())

hour, minute = '03', '42'
filename = f'2025_04_25_{hour}_{minute}_jointOpt_Nue{2}_{topolobies}Topos_TxSNR{TxSNR}dB.npz'
loaded = np.load(BASE_DIR + data_dir + filename, allow_pickle=True)
results_dict_Nue2 = dict(loaded.items())

hour, minute = '04', '56'
filename = f'2025_04_25_{hour}_{minute}_jointOpt_Nue{3}_{topolobies}Topos_TxSNR{TxSNR}dB.npz'
loaded = np.load(BASE_DIR + data_dir + filename, allow_pickle=True)
results_dict_Nue3 = dict(loaded.items())

hour, minute = '06', '21'
filename = f'2025_04_25_{hour}_{minute}_jointOpt_Nue{4}_{topolobies}Topos_TxSNR{TxSNR}dB.npz'
loaded = np.load(BASE_DIR + data_dir + filename, allow_pickle=True)
results_dict_Nue4 = dict(loaded.items())

hour, minute = '07', '56'
filename = f'2025_04_25_{hour}_{minute}_jointOpt_Nue{5}_{topolobies}Topos_TxSNR{TxSNR}dB.npz'
loaded = np.load(BASE_DIR + data_dir + filename, allow_pickle=True)
results_dict_Nue5 = dict(loaded.items())

hour, minute = '09', '48'
filename = f'2025_04_25_{hour}_{minute}_jointOpt_Nue{6}_{topolobies}Topos_TxSNR{TxSNR}dB.npz'
loaded = np.load(BASE_DIR + data_dir + filename, allow_pickle=True)
results_dict_Nue6 = dict(loaded.items())

#%% Extracting the data

sinr_results_Nue1 = results_dict_Nue1['sinr_results']
snr_sens_results_Nue1 = results_dict_Nue1['snr_sens_results']
admm_hist_results_Nue1 = results_dict_Nue1['admm_hist_results']

sinr_results_Nue2 = results_dict_Nue2['sinr_results']
snr_sens_results_Nue2 = results_dict_Nue2['snr_sens_results']
admm_hist_results_Nue2 = results_dict_Nue2['admm_hist_results']

sinr_results_Nue3 = results_dict_Nue3['sinr_results']
snr_sens_results_Nue3 = results_dict_Nue3['snr_sens_results']
admm_hist_results_Nue3 = results_dict_Nue3['admm_hist_results']

sinr_results_Nue4 = results_dict_Nue4['sinr_results']
snr_sens_results_Nue4 = results_dict_Nue4['snr_sens_results']
admm_hist_results_Nue4 = results_dict_Nue4['admm_hist_results']

sinr_results_Nue5 = results_dict_Nue5['sinr_results']
snr_sens_results_Nue5 = results_dict_Nue5['snr_sens_results']
admm_hist_results_Nue5 = results_dict_Nue5['admm_hist_results']

sinr_results_Nue6 = results_dict_Nue6['sinr_results']
snr_sens_results_Nue6 = results_dict_Nue6['snr_sens_results']
admm_hist_results_Nue6 = results_dict_Nue6['admm_hist_results']

lambda_values = results_dict_Nue6['lambda_values']

#%% Mean SINR vs N_ue for each lambda

sinr_threshold = 2.0
num_lambdas = len(lambda_values)
sinr_Nues_lambdas = np.zeros((6, num_lambdas))

def get_mean_sinr_lambdas(sinr_results, num_lambdas, threshold=0.0):
    # sinr_results: (N_ue, num_random_topologies, len(lambda_values), len(rho_values), num_solutions)
    output = []
    for i_lamb in range(num_lambdas):
        sinr_lamb = sinr_results[:, :, i_lamb, :, :]
        sinr_lamb_vals = sinr_lamb[sinr_lamb >= threshold]
        output.append(np.mean(sinr_lamb_vals))
    return np.array(output)


sinr_Nues_lambdas[0, :] = get_mean_sinr_lambdas(sinr_results_Nue1, num_lambdas, sinr_threshold)
sinr_Nues_lambdas[1, :] = get_mean_sinr_lambdas(sinr_results_Nue2, num_lambdas, sinr_threshold)
sinr_Nues_lambdas[2, :] = get_mean_sinr_lambdas(sinr_results_Nue3, num_lambdas, sinr_threshold)
sinr_Nues_lambdas[3, :] = get_mean_sinr_lambdas(sinr_results_Nue4, num_lambdas, sinr_threshold)
sinr_Nues_lambdas[4, :] = get_mean_sinr_lambdas(sinr_results_Nue5, num_lambdas, sinr_threshold)
sinr_Nues_lambdas[5, :] = get_mean_sinr_lambdas(sinr_results_Nue6, num_lambdas, sinr_threshold)

# Plotting the results

plt.plot(sinr_Nues_lambdas[:, 0], 'o-', label='Lambda = 0.3')
plt.plot(sinr_Nues_lambdas[:, 1], 's-', label='Lambda = 0.6')
plt.plot(sinr_Nues_lambdas[:, 2], '^-', label='Lambda = 0.9')
plt.title('SINR vs N_ue for different lambdas')
plt.legend()
plt.tight_layout()
plt.show()

#%% Mean Sensing SNR vs N_ue for each lambda

snr_threshold = 0.0
snr_sens_Nues_lambdas = np.zeros((2, 6, num_lambdas))

def get_mean_sens_snr_lambdas(snr_sens_results, num_lambdas, threshold=0.0):
    # snr_sens_results: (2, num_random_topologies, len(lambda_values), len(rho_values), num_solutions)
    # first_idx: 0 for total sensing SNR, 1 for sensing only SNR
    sens_output = np.zeros((2, num_lambdas))
    for i_lamb in range(num_lambdas):
        snr_sens_lamb = snr_sens_results[0, :, i_lamb, :, :]
        snr_sens_lamb_vals = snr_sens_lamb[snr_sens_lamb >= threshold]
        sens_output[0, i_lamb] = np.mean(snr_sens_lamb_vals)
        snr_so_lamb = snr_sens_results[1, :, i_lamb, :, :]
        snr_so_lamb_vals = snr_so_lamb[snr_so_lamb >= threshold]
        if len(snr_so_lamb_vals) == 0:
            snr_so_lamb_vals = np.array([0])
        sens_output[1, i_lamb] = np.mean(snr_so_lamb_vals)
    return sens_output


snr_sens_Nues_lambdas[:, 0, :] = get_mean_sens_snr_lambdas(snr_sens_results_Nue1, num_lambdas, threshold=snr_threshold)
snr_sens_Nues_lambdas[:, 1, :] = get_mean_sens_snr_lambdas(snr_sens_results_Nue2, num_lambdas, threshold=snr_threshold)
snr_sens_Nues_lambdas[:, 2, :] = get_mean_sens_snr_lambdas(snr_sens_results_Nue3, num_lambdas, threshold=snr_threshold)
snr_sens_Nues_lambdas[:, 3, :] = get_mean_sens_snr_lambdas(snr_sens_results_Nue4, num_lambdas, threshold=snr_threshold)
snr_sens_Nues_lambdas[:, 4, :] = get_mean_sens_snr_lambdas(snr_sens_results_Nue5, num_lambdas, threshold=snr_threshold)
snr_sens_Nues_lambdas[:, 5, :] = get_mean_sens_snr_lambdas(snr_sens_results_Nue6, num_lambdas, threshold=snr_threshold)

# Plotting the results
plt.plot(snr_sens_Nues_lambdas[0, :, 0], 'o-', label='Total / Lambda = 0.3')
plt.plot(snr_sens_Nues_lambdas[0, :, 1], 's-', label='Total / Lambda = 0.6')
plt.plot(snr_sens_Nues_lambdas[0, :, 2], '^-', label='Total / Lambda = 0.9')
plt.title('Sensing SNR vs N_ue for different lambdas')
plt.legend()
plt.tight_layout()
plt.show()


#%% Scenario 1: Plotting the SINR/SNR results for each N_ue

mksize = 10
lw = 2
font_size = 15
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': font_size,          # base font size
    'axes.labelsize': font_size,     # x and y labels
    'xtick.labelsize': font_size,    # x ticks
    'ytick.labelsize': font_size,    # y ticks
    'legend.fontsize': 13,    # legend
    'axes.titlesize': 10,    # title (if you have one)
})

colors = ['tab:green', 'tab:blue', 'tab:red']
markers = ['^', 'o', 's']

plt.figure(figsize=(8, 6))
ue_vals = np.arange(1, 7)
plt.plot(ue_vals, sinr_Nues_lambdas[:, 0], linestyle='-', marker=markers[0], color=colors[0], lw=lw, markersize=mksize, label=r'min-UE SINR / $\lambda$ = 0.3')
plt.plot(ue_vals, sinr_Nues_lambdas[:, 1], linestyle='-', marker=markers[1], color=colors[1], lw=lw, markersize=mksize, label=r'min-UE SINR / $\lambda$ = 0.6')
plt.plot(ue_vals, sinr_Nues_lambdas[:, 2], linestyle='-', marker=markers[2], color=colors[2], lw=lw, markersize=mksize, label=r'min-UE SINR / $\lambda$ = 0.9')
plt.plot(ue_vals, snr_sens_Nues_lambdas[0, :, 0], linestyle='--', marker=markers[0], color=colors[0], lw=lw, markersize=mksize, label=r'Total Sens SNR / $\lambda$ = 0.3')
plt.plot(ue_vals, snr_sens_Nues_lambdas[0, :, 1], linestyle='--', marker=markers[1], color=colors[1], lw=lw, markersize=mksize, label=r'Total Sens SNR / $\lambda$ = 0.6')
plt.plot(ue_vals, snr_sens_Nues_lambdas[0, :, 2], linestyle='--', marker=markers[2], color=colors[2], lw=lw, markersize=mksize, label=r'Total Sens SNR / $\lambda$ = 0.9')
plt.xticks(ue_vals)
plt.xlabel('Number of UEs')
plt.ylabel('SINR / Sensing SNR (dB)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.ylim(4, 28.5)
plt.yticks(np.arange(5, 28, 2.5))
plt.legend(loc='best', ncol=2)
plt.tight_layout()

if SAVE_FIG_SC1:
    prefix = f'exp2_sinrSNR_vs_Nue_TxSNR{TxSNR}dB_singleFig_SC1.png'
    lib.save_sim_figure(prefix=prefix, save_dir=BASE_DIR+'figures/', add_timestamp=False)

plt.show()
plt.rcdefaults()




#%% Loading the data for the second scenario

# ADMM Date
topolobies = 50
TxSNR = 20
N_ap = 11
M_t = 3

hour, minute = '04', '53'
filename = f'2025_04_26_{hour}_{minute}_jointOpt_Nue{4}_{topolobies}Topos_TxSNR{TxSNR}dB_{N_ap}APs_{M_t}ants.npz'
loaded = np.load(BASE_DIR + data_dir + filename, allow_pickle=True)
results_dict_Nue4_sc2 = dict(loaded.items())

hour, minute = '07', '00'
filename = f'2025_04_26_{hour}_{minute}_jointOpt_Nue{5}_{topolobies}Topos_TxSNR{TxSNR}dB_{N_ap}APs_{M_t}ants.npz'
loaded = np.load(BASE_DIR + data_dir + filename, allow_pickle=True)
results_dict_Nue5_sc2 = dict(loaded.items())

hour, minute = '09', '35'
filename = f'2025_04_26_{hour}_{minute}_jointOpt_Nue{6}_{topolobies}Topos_TxSNR{TxSNR}dB_{N_ap}APs_{M_t}ants.npz'
loaded = np.load(BASE_DIR + data_dir + filename, allow_pickle=True)
results_dict_Nue6_sc2 = dict(loaded.items())

# Centralized Date
topolobies = 50

hour, minute = '00', '11'
filename = f'2025_05_01_{hour}_{minute}_jointOpt_cent_Nue{4}_{topolobies}Topos_TxSNR{TxSNR}dB_{N_ap}APs_{M_t}ants.npz'
loaded = np.load(BASE_DIR + data_dir + filename, allow_pickle=True)
results_dict_Nue4_sc2_cent = dict(loaded.items())

hour, minute = '00', '18'
filename = f'2025_05_01_{hour}_{minute}_jointOpt_cent_Nue{5}_{topolobies}Topos_TxSNR{TxSNR}dB_{N_ap}APs_{M_t}ants.npz'
loaded = np.load(BASE_DIR + data_dir + filename, allow_pickle=True)
results_dict_Nue5_sc2_cent = dict(loaded.items())

hour, minute = '00', '25'
filename = f'2025_05_01_{hour}_{minute}_jointOpt_cent_Nue{6}_{topolobies}Topos_TxSNR{TxSNR}dB_{N_ap}APs_{M_t}ants.npz'
loaded = np.load(BASE_DIR + data_dir + filename, allow_pickle=True)
results_dict_Nue6_sc2_cent = dict(loaded.items())

#%% Extracting the data for the second scenario

# ADMM Data
sinr_results_Nue4_sc2 = results_dict_Nue4_sc2['sinr_results']
snr_sens_results_Nue4_sc2 = results_dict_Nue4_sc2['snr_sens_results']
admm_hist_results_Nue4_sc2 = results_dict_Nue4_sc2['admm_hist_results']

sinr_results_Nue5_sc2 = results_dict_Nue5_sc2['sinr_results']
snr_sens_results_Nue5_sc2 = results_dict_Nue5_sc2['snr_sens_results']
admm_hist_results_Nue5_sc2 = results_dict_Nue5_sc2['admm_hist_results']

sinr_results_Nue6_sc2 = results_dict_Nue6_sc2['sinr_results']
snr_sens_results_Nue6_sc2 = results_dict_Nue6_sc2['snr_sens_results']
admm_hist_results_Nue6_sc2 = results_dict_Nue6_sc2['admm_hist_results']

lambda_values_sc2 = results_dict_Nue6_sc2['lambda_values']

# Centralized Data
sinr_results_Nue4_sc2_cent = results_dict_Nue4_sc2_cent['sinr_results']
snr_sens_results_Nue4_sc2_cent = results_dict_Nue4_sc2_cent['snr_sens_results']

sinr_results_Nue5_sc2_cent = results_dict_Nue5_sc2_cent['sinr_results']
snr_sens_results_Nue5_sc2_cent = results_dict_Nue5_sc2_cent['snr_sens_results']

sinr_results_Nue6_sc2_cent = results_dict_Nue6_sc2_cent['sinr_results']
snr_sens_results_Nue6_sc2_cent = results_dict_Nue6_sc2_cent['snr_sens_results']


#%% Mean SINR vs N_ue for each lambda for the second scenario

sinr_threshold_sc2 = 0.0
num_lambdas_sc2 = len(lambda_values_sc2)
sinr_Nues_lambdas_sc2 = np.zeros((3, num_lambdas_sc2))
sinr_Nues_lambdas_sc2_cent = np.zeros((3, num_lambdas_sc2))
# ADMM Data
sinr_Nues_lambdas_sc2[0, :] = get_mean_sinr_lambdas(sinr_results_Nue4_sc2, num_lambdas_sc2, sinr_threshold_sc2)
sinr_Nues_lambdas_sc2[1, :] = get_mean_sinr_lambdas(sinr_results_Nue5_sc2, num_lambdas_sc2, sinr_threshold_sc2)
sinr_Nues_lambdas_sc2[2, :] = get_mean_sinr_lambdas(sinr_results_Nue6_sc2, num_lambdas_sc2, sinr_threshold_sc2)
# Centralized Data
sinr_Nues_lambdas_sc2_cent[0, :] = get_mean_sinr_lambdas(sinr_results_Nue4_sc2_cent, num_lambdas_sc2, sinr_threshold_sc2)
sinr_Nues_lambdas_sc2_cent[1, :] = get_mean_sinr_lambdas(sinr_results_Nue5_sc2_cent, num_lambdas_sc2, sinr_threshold_sc2)
sinr_Nues_lambdas_sc2_cent[2, :] = get_mean_sinr_lambdas(sinr_results_Nue6_sc2_cent, num_lambdas_sc2, sinr_threshold_sc2)

# Plotting the results

plt.plot(sinr_Nues_lambdas_sc2[:, 0], 'o-', label=f'ADMM | Lambda = {lambda_values_sc2[0]}')
plt.plot(sinr_Nues_lambdas_sc2[:, 1], 's-', label=f'ADMM | Lambda = {lambda_values_sc2[1]}')
plt.plot(sinr_Nues_lambdas_sc2_cent[:, 0], 'o--', label=f'Cent | Lambda = {lambda_values_sc2[0]}')
plt.plot(sinr_Nues_lambdas_sc2_cent[:, 1], 's--', label=f'Cent | Lambda = {lambda_values_sc2[1]}')
plt.title('Scenario 2: SINR vs N_ue for different lambdas')
plt.legend()
plt.tight_layout()
plt.show()


#%% Mean Sensing SNR vs N_ue for each lambda for the second scenario

snr_threshold_sc2 = 0.0
snr_sens_Nues_lambdas_sc2 = np.zeros((2, 3, num_lambdas_sc2))
snr_sens_Nues_lambdas_sc2_cent = np.zeros((2, 3, num_lambdas_sc2))
# ADMM Data
snr_sens_Nues_lambdas_sc2[:, 0, :] = get_mean_sens_snr_lambdas(snr_sens_results_Nue4_sc2, num_lambdas_sc2, threshold=snr_threshold_sc2)
snr_sens_Nues_lambdas_sc2[:, 1, :] = get_mean_sens_snr_lambdas(snr_sens_results_Nue5_sc2, num_lambdas_sc2, threshold=snr_threshold_sc2)
snr_sens_Nues_lambdas_sc2[:, 2, :] = get_mean_sens_snr_lambdas(snr_sens_results_Nue6_sc2, num_lambdas_sc2, threshold=snr_threshold_sc2)
# Centralized Data
snr_sens_Nues_lambdas_sc2_cent[:, 0, :] = get_mean_sens_snr_lambdas(snr_sens_results_Nue4_sc2_cent, num_lambdas_sc2, threshold=snr_threshold_sc2)
snr_sens_Nues_lambdas_sc2_cent[:, 1, :] = get_mean_sens_snr_lambdas(snr_sens_results_Nue5_sc2_cent, num_lambdas_sc2, threshold=snr_threshold_sc2)
snr_sens_Nues_lambdas_sc2_cent[:, 2, :] = get_mean_sens_snr_lambdas(snr_sens_results_Nue6_sc2_cent, num_lambdas_sc2, threshold=snr_threshold_sc2)

# Plotting the results
plt.plot(snr_sens_Nues_lambdas_sc2[0, :, 0], 'o-', label=f'Total / Lambda = {lambda_values_sc2[0]}')
plt.plot(snr_sens_Nues_lambdas_sc2[0, :, 1], 's-', label=f'Total / Lambda = {lambda_values_sc2[1]}')
plt.plot(snr_sens_Nues_lambdas_sc2_cent[0, :, 0], 'o--', label=f'Cent / Total / Lambda = {lambda_values_sc2[0]}')
plt.plot(snr_sens_Nues_lambdas_sc2_cent[0, :, 1]+0.1, 's--', label=f'Cent / Total / Lambda = {lambda_values_sc2[1]}')
plt.title('Scenario 2: Sensing SNR vs N_ue for different lambdas')
plt.legend()
plt.tight_layout()
plt.show()


#%% Scenario 2: Plotting the SINR/SNR results for each N_ue

ue_vals_sc2 = np.arange(4, 7)

mksize = 12
lw = 2
font_size = 15
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': font_size,          # base font size
    'axes.labelsize': font_size,     # x and y labels
    'xtick.labelsize': font_size,    # x ticks
    'ytick.labelsize': font_size,    # y ticks
    'legend.fontsize': 12,    # legend
    'axes.titlesize': 10,    # title (if you have one)
})

plt.figure(figsize=(7, 5))
plt.plot(ue_vals_sc2, sinr_Nues_lambdas_sc2[:, 0], linestyle='-', marker=markers[1], color=colors[1], lw=lw, markersize=mksize, label=r'min-UE SINR / $\lambda$ = 0.6')
plt.plot(ue_vals_sc2, sinr_Nues_lambdas_sc2[:, 1], linestyle='-', marker=markers[2], color=colors[2], lw=lw, markersize=mksize, label=r'min-UE SINR / $\lambda$ = 0.9')
plt.plot(ue_vals_sc2, sinr_Nues_lambdas_sc2_cent[:, 0], linestyle='-.', marker=markers[1], markerfacecolor='none', color='black', lw=1.5, markersize=9, label=r'min-UE SINR / $\lambda$ = 0.6 / Cent')
plt.plot(ue_vals_sc2, sinr_Nues_lambdas_sc2_cent[:, 1], linestyle='-.', marker=markers[2], markerfacecolor='none', color='black', lw=1.5, markersize=9, label=r'min-UE SINR / $\lambda$ = 0.9 / Cent')

plt.plot(ue_vals_sc2, snr_sens_Nues_lambdas_sc2[0, :, 0], linestyle='--', marker=markers[1], color=colors[1], lw=lw, markersize=mksize, label=r'Total Sens SNR / $\lambda$ = 0.6')
plt.plot(ue_vals_sc2, snr_sens_Nues_lambdas_sc2[0, :, 1], linestyle='--', marker=markers[2], color=colors[2], lw=lw, markersize=mksize, label=r'Total Sens SNR / $\lambda$ = 0.9')
plt.plot(ue_vals_sc2, snr_sens_Nues_lambdas_sc2_cent[0, :, 0], linestyle=':', marker=markers[1], markerfacecolor='none', color='black', lw=1.5, markersize=9, label=r'Total Sens SNR / $\lambda$ = 0.6 / Cent')
plt.plot(ue_vals_sc2, snr_sens_Nues_lambdas_sc2_cent[0, :, 1], linestyle=':', marker=markers[2], markerfacecolor='none', color='black', lw=1.5, markersize=11, label=r'Total Sens SNR / $\lambda$ = 0.9 / Cent')
plt.xticks(ue_vals_sc2)
plt.xlabel('Number of UEs')
plt.ylabel('SINR / Sensing SNR (dB)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.ylim(4, 28.5)
plt.yticks(np.arange(5, 28, 2.5))
plt.legend(loc='best', ncol=2)
plt.tight_layout()

if SAVE_FIG_SC2:
    prefix = f'exp2_sinrSNR_vs_Nue_TxSNR{TxSNR}dB_singleFig_SC2.png'
    lib.save_sim_figure(prefix=prefix, save_dir=BASE_DIR+'figures/', add_timestamp=False)

plt.show()
plt.rcdefaults()


#%% Joint Figure for both scenarios

lw = 1.8
font_size = 12

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': font_size,          # base font size
    'axes.labelsize': font_size,     # x and y labels
    'xtick.labelsize': font_size,    # x ticks
    'ytick.labelsize': font_size,    # y ticks
    'legend.fontsize': 10,    # legend
    'axes.titlesize': 10,    # title (if you have one)
})

plt.figure(figsize=(10, 4.5))

plt.subplot(1, 2, 1)
mksize = 8
plt.plot(ue_vals, sinr_Nues_lambdas[:, 0], linestyle='-', marker=markers[0], color=colors[0], lw=lw, markersize=mksize, label=r'min-UE SINR / $\lambda$ = 0.3')
plt.plot(ue_vals, sinr_Nues_lambdas[:, 1], linestyle='-', marker=markers[1], color=colors[1], lw=lw, markersize=mksize, label=r'min-UE SINR / $\lambda$ = 0.6')
plt.plot(ue_vals, sinr_Nues_lambdas[:, 2], linestyle='-', marker=markers[2], color=colors[2], lw=lw, markersize=8, label=r'min-UE SINR / $\lambda$ = 0.9')
plt.plot(ue_vals, snr_sens_Nues_lambdas[0, :, 0], linestyle='--', marker=markers[0], color=colors[0], lw=lw, markersize=mksize, label=r'Total Sens SNR / $\lambda$ = 0.3')
plt.plot(ue_vals, snr_sens_Nues_lambdas[0, :, 1], linestyle='--', marker=markers[1], color=colors[1], lw=lw, markersize=mksize, label=r'Total Sens SNR / $\lambda$ = 0.6')
plt.plot(ue_vals, snr_sens_Nues_lambdas[0, :, 2], linestyle='--', marker=markers[2], color=colors[2], lw=lw, markersize=8, label=r'Total Sens SNR / $\lambda$ = 0.9')
plt.xticks(ue_vals)
plt.xlabel('Number of UEs')
plt.ylabel('SINR / Sensing SNR (dB)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.ylim(4, 28.5)
plt.yticks(np.arange(5, 28, 2.5))
plt.legend(loc='best', ncol=2, fontsize=10.4)

plt.subplot(1, 2, 2)
mksize = 8
plt.plot(ue_vals_sc2, sinr_Nues_lambdas_sc2[:, 0], linestyle='-', marker=markers[1], color=colors[1], lw=lw, markersize=mksize, label=r'min-UE SINR / $\lambda$ = 0.6')
plt.plot(ue_vals_sc2, sinr_Nues_lambdas_sc2[:, 1], linestyle='-', marker=markers[2], color=colors[2], lw=lw, markersize=mksize, label=r'min-UE SINR / $\lambda$ = 0.9')
plt.plot(ue_vals_sc2, sinr_Nues_lambdas_sc2_cent[:, 0], linestyle='-.', marker=markers[1], markerfacecolor='none', color='black', lw=1.2, markersize=9, label=r'min-UE SINR / $\lambda$ = 0.6 / Cent')
plt.plot(ue_vals_sc2, sinr_Nues_lambdas_sc2_cent[:, 1], linestyle='-.', marker=markers[2], markerfacecolor='none', color='black', lw=1.2, markersize=9, label=r'min-UE SINR / $\lambda$ = 0.9 / Cent')

plt.plot(ue_vals_sc2, snr_sens_Nues_lambdas_sc2[0, :, 0], linestyle='--', marker=markers[1], color=colors[1], lw=lw, markersize=mksize, label=r'Total Sens SNR / $\lambda$ = 0.6')
plt.plot(ue_vals_sc2, snr_sens_Nues_lambdas_sc2[0, :, 1], linestyle='--', marker=markers[2], color=colors[2], lw=lw, markersize=mksize, label=r'Total Sens SNR / $\lambda$ = 0.9')
plt.plot(ue_vals_sc2, snr_sens_Nues_lambdas_sc2_cent[0, :, 0], linestyle=':', marker=markers[1], markerfacecolor='none', color='black', lw=1.2, markersize=8, label=r'Total Sens SNR / $\lambda$ = 0.6 / Cent')
plt.plot(ue_vals_sc2, snr_sens_Nues_lambdas_sc2_cent[0, :, 1], linestyle=':', marker=markers[2], markerfacecolor='none', color='black', lw=1.2, markersize=11, label=r'Total Sens SNR / $\lambda$ = 0.9 / Cent')
plt.xticks(ue_vals_sc2)
plt.xlabel('Number of UEs')
# plt.ylabel('SINR / Sensing SNR (dB)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.ylim(4, 28.5)
plt.yticks(np.arange(5, 28, 2.5))
plt.legend(loc='best', ncol=1, fontsize=10, bbox_to_anchor=(0.93, 0.48))
plt.tight_layout()

if SAVE_FIG_JOINT:
    size_preamble = '10-4p5'
    prefix1 = f'exp2_sinrSNR_vs_Nue_TxSNR{TxSNR}dB_joint_{size_preamble}.png'
    lib.save_sim_figure(prefix=prefix1, save_dir=BASE_DIR+'figures/', add_timestamp=False)
    prefix2 = f'exp2_sinrSNR_vs_Nue_TxSNR{TxSNR}dB_joint_{size_preamble}.pdf'
    lib.save_sim_figure(prefix=prefix2, save_dir=BASE_DIR + 'figures/', add_timestamp=False)
plt.show()
plt.rcdefaults()

