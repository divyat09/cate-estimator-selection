"""
File for loading the LBIDD semi-synthetic dataset.

Shimoni et al. (2018) took the real covariates from the Linked Births and Infant
Deaths Database (lbidd) (MacDorman & Atkinson, 1998) and generated
semi-synthetic data by generating the treatment assignments and outcomes via
random functions of the covariates.

Data Wiki: https://www.synapse.org/#!Synapse:syn11738767/wiki/512854
CDC Data Website: https://www.cdc.gov/nchs/nvss/linked-birth.htm

References:

    MacDorman, Marian F and Jonnae O Atkinson. Infant mortality statistics from
        the linked birth/infant death data set—1995 period data. Mon Vital Stat
        Rep, 46(suppl 2):1–22, 1998.
        https://www.cdc.gov/nchs/data/mvsr/supp/mv46_06s2.pdf

    Shimoni, Y., Yanover, C., Karavani, E., & Goldschmidt, Y. (2018).
        Benchmarking Framework for Performance-Evaluation of Causal Inference
        Analysis. ArXiv, abs/1802.05046.
"""

import os
import numpy as np
import pandas as pd
import tarfile
import pickle
import random

import sys

def acic_2016_main_loader(dataset_idx=1):

    data_dict={}
    base_dir= os.path.expanduser('~') + '/scratch/cate_eval_analysis/acic_2016/'
    total_datasets= 77
    #The sub directory indices for the different datasets start with 1
    dataset_idx+=1

    covar_df= pd.read_csv(base_dir + 'x.csv')
    covar_df= pd.get_dummies(covar_df)
    data_dict['w']=  covar_df.to_numpy()

    curr_dir= base_dir + str(dataset_idx) + '/'
    for _, _, files in os.walk(curr_dir):
        random.shuffle(files)
        curr_file= curr_dir + files[0]
    df= pd.read_csv(curr_file)

    # Shape: (N,)
    t= df['z'].to_numpy()
    y0= df['mu0'].to_numpy()
    y1= df['mu1'].to_numpy()

    y= t*y1 + (1-t)*y0
    ite= y1 - y0
    ate= np.mean(ite)

    data_dict['t']= t
    data_dict['y']= y
    data_dict['ites']= ite
    data_dict['ate']= ate

    # Shuffle data points before train/val split
    size = data_dict['ites'].shape[0]
    inds = np.arange(size)
    np.random.shuffle(inds)
    for key in data_dict.keys():
        if key != 'ate':
            data_dict[key] = data_dict[key][inds]

    # Train/Val split
    final_data = {'tr': {}, 'eval': {}}
    train_size = int(0.8 * size)

    for key in data_dict.keys():
        if key == 'ate':
            final_data['tr'][key] = data_dict[key]
        else:
            final_data['tr'][key] = data_dict[key][:train_size]
            final_data['tr'][key] = data_dict[key][:train_size]

    for key in data_dict.keys():
        if key == 'ate':
            final_data['eval'][key] = data_dict[key]
        else:
            final_data['eval'][key] = data_dict[key][train_size:]
            final_data['eval'][key] = data_dict[key][train_size:]

    return final_data


