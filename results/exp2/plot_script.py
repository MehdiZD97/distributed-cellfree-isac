
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

SAVE_FIG = False

#%% Loading the data

hour = 20
minute = 58
topolobies = 4
TxSNR = 20

filename = f'2025_04_24_{hour}_{minute}_jointOpt_{1}UEs_{topolobies}Topos_TxSNR{TxSNR}dB.npz'
loaded = np.load(BASE_DIR + data_dir + filename, allow_pickle=True)
results_dict_Nue1 = dict(loaded.items())

filename = f'2025_04_24_{hour}_{minute}_jointOpt_{2}UEs_{topolobies}Topos_TxSNR{TxSNR}dB.npz'
loaded = np.load(BASE_DIR + data_dir + filename, allow_pickle=True)
results_dict_Nue2 = dict(loaded.items())

#%% Extracting the data

sinr_results_Nue1 = results_dict_Nue1['sinr_results']
snr_sens_results_Nue1 = results_dict_Nue1['snr_sens_results']
admm_hist_results_Nue1 = results_dict_Nue1['admm_hist_results']

sinr_results_Nue2 = results_dict_Nue2['sinr_results']
snr_sens_results_Nue2 = results_dict_Nue2['snr_sens_results']
admm_hist_results_Nue2 = results_dict_Nue2['admm_hist_results']

#%% Plotting the results

print('Solution 1:', np.mean(sinr_results_Nue2, axis=1)[:, 0, 0, 0])
print('Solution 2:', np.mean(sinr_results_Nue2, axis=1)[:, 0, 0, 1])
