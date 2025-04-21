
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

SAVE_FIG = False

#%% Loading the data

hour = 15
minute = 41
optPA_extra_min = 0
# 'fixed' or 'rand'
targetLoc_type = 'rand'
sinrSNR_type = '24_30'

filename = f'2025_04_21_{hour}_{minute}_splitOpt_fixedPSR0p2_{targetLoc_type}TargetLoc_sinrSNR_{sinrSNR_type}dB.npz'
loaded = np.load(BASE_DIR + data_dir + filename, allow_pickle=True)
results_dict_fixedPSR0p2 = dict(loaded.items())

filename = f'2025_04_21_{hour}_{minute}_splitOpt_fixedPSR0p8_{targetLoc_type}TargetLoc_sinrSNR_{sinrSNR_type}dB.npz'
loaded = np.load(BASE_DIR + data_dir + filename, allow_pickle=True)
results_dict_fixedPSR0p8 = dict(loaded.items())

filename = f'2025_04_21_{hour}_{minute+optPA_extra_min}_splitOpt_optPA_{targetLoc_type}TargetLoc_sinrSNR_{sinrSNR_type}dB.npz'
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
        plt.barh(
            bar_positions[i][g],
            values[g, i],
            xerr=std_devs[g, i],
            height=bar_height,
            color=colors[i],
            hatch='/' if is_highlighted else '',
            edgecolor='black',
            capsize=5,
            error_kw={'linewidth': 2},
            label=bar_labels[i] if g == 0 else ''
        )

# Add y-ticks and labels
plt.yticks(group_indices, group_labels, fontsize=12)
plt.xlabel('Value (dB)', fontsize=12)
plt.title('')
# Reverse legend to match top-to-bottom order of bars
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], title='Bar Type', loc='best', fontsize=12)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
if SAVE_FIG:
    lib.save_sim_figure(prefix=f'exp1_bar_plot_sinrSNR_{sinrSNR_type}dB.png', save_dir=BASE_DIR+'figures/', add_timestamp=False)
plt.show()
