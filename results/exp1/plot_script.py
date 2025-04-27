
import numpy as np
import matplotlib.pyplot as plt
from src.utils import library as lib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__name__))
if os.path.basename(BASE_DIR) == 'distributed-cellfree-isac':
    BASE_DIR = BASE_DIR + '/results/exp1/'
else:
    BASE_DIR = BASE_DIR + '/'
data_dir = 'data/'

SAVE_FIG_BAR = False
SAVE_FIG_LINE = False
SAVE_FIG_JOINT = True

#%% Loading the data

hour = 21
minute = 47
optPA_extra_min = 0
# 'fixed' or 'rand'
targetLoc_type = 'rand'
sinrSNR_type = '15_20'

filename = f'2025_04_26_{hour}_{minute}_splitOpt_fixedPSR0p2_{targetLoc_type}TargetLoc_sinrSNR_{sinrSNR_type}dB.npz'
loaded = np.load(BASE_DIR + data_dir + filename, allow_pickle=True)
results_dict_fixedPSR0p2 = dict(loaded.items())

filename = f'2025_04_26_{hour}_{minute}_splitOpt_fixedPSR0p8_{targetLoc_type}TargetLoc_sinrSNR_{sinrSNR_type}dB.npz'
loaded = np.load(BASE_DIR + data_dir + filename, allow_pickle=True)
results_dict_fixedPSR0p8 = dict(loaded.items())

filename = f'2025_04_26_{hour}_{minute+optPA_extra_min}_splitOpt_optPA_{targetLoc_type}TargetLoc_sinrSNR_{sinrSNR_type}dB.npz'
loaded = np.load(BASE_DIR + data_dir + filename, allow_pickle=True)
results_dict_optPA = dict(loaded.items())


#%% Extracting the data

comm_sinrs_optPA = results_dict_optPA['comm_sinrs']
sens_snrs_optPA = results_dict_optPA['sens_snrs']
sens_only_snrs_optPA = results_dict_optPA['sens_only_snrs']

comm_sinrs_fixedPSR0p2 = results_dict_fixedPSR0p2['comm_sinrs']
sens_snrs_fixedPSR0p2 = results_dict_fixedPSR0p2['sens_snrs']
sens_only_snrs_fixedPSR0p2 = results_dict_fixedPSR0p2['sens_only_snrs']

comm_sinrs_fixedPSR0p8 = results_dict_fixedPSR0p8['comm_sinrs']
sens_snrs_fixedPSR0p8 = results_dict_fixedPSR0p8['sens_snrs']
sens_only_snrs_fixedPSR0p8 = results_dict_fixedPSR0p8['sens_only_snrs']

N_ue = np.array(comm_sinrs_optPA).shape[1]

#%% Horizontal bar plot with error bars and synced legend

# Data
group_labels = ['PSR: 0.2', 'SplitOpt', 'PSR: 0.8']
bar_labels = ['Comm SINR', 'Sens SNR', 'SO SNR']  # Order in legend (top to bottom should match bar appearance)
values = np.array([
    [np.mean(comm_sinrs_fixedPSR0p2, axis=(0,1)), np.mean(sens_snrs_fixedPSR0p2), np.mean(sens_only_snrs_fixedPSR0p2)],
    [np.mean(comm_sinrs_optPA, axis=(0,1)), np.mean(sens_snrs_optPA), np.mean(sens_only_snrs_optPA)],
    [np.mean(comm_sinrs_fixedPSR0p8, axis=(0,1)), np.mean(sens_snrs_fixedPSR0p8), np.mean(sens_only_snrs_fixedPSR0p8)]
])
std_devs = np.array([
    [np.std(comm_sinrs_fixedPSR0p2, axis=(0,1)), np.std(sens_snrs_fixedPSR0p2), 0],
    [np.std(comm_sinrs_optPA, axis=(0,1)), np.std(sens_snrs_optPA), 0],
    [np.std(comm_sinrs_fixedPSR0p8, axis=(0,1)), np.std(sens_snrs_fixedPSR0p8), 0]
])

n_groups, n_bars = values.shape
bar_height = 0.25
group_indices = np.arange(n_groups)
offsets = np.linspace(-(n_bars-1)/2, (n_bars-1)/2, n_bars) * bar_height

colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:pink']
bar_positions = [group_indices + offset for offset in offsets]
group_to_highlight = 1
for i in range(n_bars):
    for g in range(n_groups):
        is_highlighted = g == group_to_highlight
        plt.barh(bar_positions[i][g], values[g, i], xerr=std_devs[g, i], height=bar_height, color=colors[i], hatch='/' if is_highlighted else '',
                 edgecolor='black', capsize=5, error_kw={'linewidth': 2}, label=bar_labels[i] if g == 0 else '')

# Add y-ticks and labels
plt.yticks(group_indices, group_labels, fontsize=12)
plt.xlabel('Value (dB)', fontsize=12)
plt.title('')
# Reverse legend to match top-to-bottom order of bars
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], title='Bar Type', loc='best', fontsize=12)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

if SAVE_FIG_BAR:
    lib.save_sim_figure(prefix=f'exp1_bar_plot_sinrSNR_{sinrSNR_type}dB_Nue{N_ue}.png', save_dir=BASE_DIR+'figures/', add_timestamp=False)

plt.show()



#%% Figure 2 - Loading the data

N_ap = 4
M_t = 10
num_topos = 1000
TxSNR = 20

hour, minute = '22', '27'
filename = f'2025_04_26_{hour}_{minute}_splitOpt_Nue{4}_{num_topos}Topos_TxSNR{TxSNR}dB_{N_ap}APs_{M_t}ants.npz'
loaded = np.load(BASE_DIR + data_dir + filename, allow_pickle=True)
results_dict_Nue4 = dict(loaded.items())

hour, minute = '22', '29'
filename = f'2025_04_26_{hour}_{minute}_splitOpt_Nue{5}_{num_topos}Topos_TxSNR{TxSNR}dB_{N_ap}APs_{M_t}ants.npz'
loaded = np.load(BASE_DIR + data_dir + filename, allow_pickle=True)
results_dict_Nue5 = dict(loaded.items())

hour, minute = '22', '32'
filename = f'2025_04_26_{hour}_{minute}_splitOpt_Nue{6}_{num_topos}Topos_TxSNR{TxSNR}dB_{N_ap}APs_{M_t}ants.npz'
loaded = np.load(BASE_DIR + data_dir + filename, allow_pickle=True)
results_dict_Nue6 = dict(loaded.items())

#%% Extracting the data
# sinr_results shape: (N_ue, num_random_topologies, len(splitOpt_sinr_cons_dB))

sinr_results_Nue4 = results_dict_Nue4['sinr_results']
snr_sens_results_Nue4 = results_dict_Nue4['snr_sens_results']

sinr_results_Nue5 = results_dict_Nue5['sinr_results']
snr_sens_results_Nue5 = results_dict_Nue5['snr_sens_results']

sinr_results_Nue6 = results_dict_Nue6['sinr_results']
snr_sens_results_Nue6 = results_dict_Nue6['snr_sens_results']

sinr_cons_values = results_dict_Nue6['SINR_constraints']

mean_sinr_Nue4 = np.mean(sinr_results_Nue4, axis=(0, 1))
mean_sinr_Nue5 = np.mean(sinr_results_Nue5, axis=(0, 1))
mean_sinr_Nue6 = np.mean(sinr_results_Nue6, axis=(0, 1))

mean_sens_snr_Nue4 = np.mean(snr_sens_results_Nue4[0, :, :], axis=0)
mean_sens_snr_Nue5 = np.mean(snr_sens_results_Nue5[0, :, :], axis=0)
mean_sens_snr_Nue6 = np.mean(snr_sens_results_Nue6[0, :, :], axis=0)

#%% Plotting results

lw = 1.5
mksize = 7
font_size = 12
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': font_size,          # base font size
    'axes.labelsize': font_size,     # x and y labels
    'xtick.labelsize': font_size,    # x ticks
    'ytick.labelsize': font_size,    # y ticks
    'legend.fontsize': 11,    # legend
    'axes.titlesize': 10,    # title (if you have one)
})

colors = ['tab:red', 'tab:green', 'tab:blue']
markers = ['^', 'o', 's']

plt.figure(figsize=(6, 4.5))
plt.plot(sinr_cons_values, mean_sinr_Nue4, linestyle='-', marker=markers[0], color=colors[0], lw=lw, markersize=mksize, label='Comm SINR / 4 UEs')
plt.plot(sinr_cons_values, mean_sinr_Nue5, linestyle='-', marker=markers[1], color=colors[1], lw=lw, markersize=mksize, label='Comm SINR / 5 UEs')
plt.plot(sinr_cons_values, mean_sinr_Nue6, linestyle='-', marker=markers[2], color=colors[2], lw=lw, markersize=mksize, label='Comm SINR / 6 UEs')

plt.plot(sinr_cons_values, mean_sens_snr_Nue4, linestyle='--', marker=markers[0], color=colors[0], lw=lw, markersize=mksize, label='Sens SNR / 4 UEs')
plt.plot(sinr_cons_values, mean_sens_snr_Nue5, linestyle='--', marker=markers[1], color=colors[1], lw=lw, markersize=mksize, label='Sens SNR / 5 UEs')
plt.plot(sinr_cons_values, mean_sens_snr_Nue6, linestyle='--', marker=markers[2], color=colors[2], lw=lw, markersize=mksize, label='Sens SNR / 6 UEs')

plt.xticks(sinr_cons_values)
plt.yticks(np.arange(5, 21, 1))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best', ncol=2)
plt.xlabel('SplitOpt SINR Constraint (dB)')
plt.ylabel('SINR / Sensing SNR (dB)')
plt.tight_layout()

if SAVE_FIG_LINE:
    prefix = f'exp1_line_plot_sinrSNR_vs_const_TxSNR{TxSNR}dB.png'
    lib.save_sim_figure(prefix=prefix, save_dir=BASE_DIR+'figures/', add_timestamp=False)

plt.show()


#%% Joint Figure for both plots


mksize = 7
lw = 1.6
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

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
# group_labels = ['PSR: 0.2', 'SplitOpt', 'PSR: 0.8']
group_labels = ['0.2', 'Opt', '0.8']
bar_labels = ['avg-UE SINR', 'Total Sens SNR', 'Sens-Only SNR']
bar_height = 0.25
group_indices = np.arange(n_groups)
offsets = np.linspace(-(n_bars-1)/2, (n_bars-1)/2, n_bars) * bar_height
colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:pink']
bar_positions = [group_indices + offset for offset in offsets]
group_to_highlight = 1
for i in range(n_bars):
    for g in range(n_groups):
        is_highlighted = g == group_to_highlight
        plt.barh(bar_positions[i][g], values[g, i], xerr=std_devs[g, i], height=bar_height, color=colors[i], hatch='/' if is_highlighted else '',
                 edgecolor='black', capsize=5, error_kw={'linewidth': 2}, label=bar_labels[i] if g == 0 else '')

# Add y-ticks and labels
plt.yticks(group_indices, group_labels)
plt.xlabel('Value (dB)')
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], title='Bar Type', loc='best', fontsize=12)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)



plt.subplot(1, 2, 2)
plt.plot(sinr_cons_values, mean_sens_snr_Nue4, linestyle='--', marker=markers[0], color=colors[0], lw=lw, markersize=mksize, label='Total Sens SNR / 4 UEs')
plt.plot(sinr_cons_values, mean_sens_snr_Nue5, linestyle='--', marker=markers[1], color=colors[1], lw=lw, markersize=mksize, label='Total Sens SNR / 5 UEs')
plt.plot(sinr_cons_values, mean_sens_snr_Nue6, linestyle='--', marker=markers[2], color=colors[2], lw=lw, markersize=mksize, label='Total Sens SNR / 6 UEs')

plt.plot(sinr_cons_values, mean_sinr_Nue4, linestyle='-', marker=markers[0], color=colors[0], lw=lw, markersize=mksize, label='avg-UE SINR / 4 UEs')
plt.plot(sinr_cons_values, mean_sinr_Nue5, linestyle='-', marker=markers[1], color=colors[1], lw=lw, markersize=mksize, label='avg-UE SINR / 5 UEs')
plt.plot(sinr_cons_values, mean_sinr_Nue6, linestyle='-', marker=markers[2], color=colors[2], lw=lw, markersize=mksize, label='avg-UE SINR / 6 UEs')

plt.xticks(sinr_cons_values)
plt.yticks(np.arange(5, 21, 1))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best', ncol=1)
plt.xlabel('SplitOpt SINR Constraint (dB)')
plt.ylabel('SINR / Sensing SNR (dB)')
plt.tight_layout()

if SAVE_FIG_JOINT:
    prefix1 = f'exp1_jointFig_barNue{N_ue}_TxSNR{TxSNR}dB.png'
    lib.save_sim_figure(prefix=prefix1, save_dir=BASE_DIR+'figures/', add_timestamp=False)
    prefix2 = f'exp1_jointFig_barNue{N_ue}_TxSNR{TxSNR}dB.pdf'
    lib.save_sim_figure(prefix=prefix2, save_dir=BASE_DIR + 'figures/', add_timestamp=False)

plt.show()
