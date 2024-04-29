"""
Link for downloading the ACIC 2016 dataset: https://jenniferhill7.wixsite.com/acic-2016/competition
Place it in the "base_dir" directory
"""

import os
import numpy as np
import pandas as pd
import tarfile
import pickle
import random

import sys

def acic_2016_main_loader(dataset_idx: int=1, dataset_name: str= 'acic_2016', root_dir: str= '/scratch/cate_eval_analysis/', seed: int=0):
    """
    Samples the dataset for training or evaluation.

    Inputs:
        dataset_idx: Index of the particular dataset
        root_dir: Root directory for the dataset
        seed: Random seed

    Returns:
        Dictionary containing the samples for the dataset along with the ATE and ITE
        w: Expected Shape (num_samples, covariate dimensions)
        t: Expected Shape (num_samples, 1)
        y: Expected Shape (num_samples, 1)
        ate: Float
        ite: Expected Shape (num_samples)         
    """

    base_dir= os.path.join(os.path.expanduser('~'), root_dir,  dataset_name, '')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    total_datasets= 77
    #The sub directory indices for the different datasets start with 1
    dataset_idx+=1

    data_dict={}
    covar_df= pd.read_csv(base_dir + 'x.csv')
    covar_df= pd.get_dummies(covar_df)
    data_dict['w']=  covar_df.to_numpy()

    curr_dir= os.path.join(base_dir, str(dataset_idx), '') 
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
    data_dict['ate']= ate
    data_dict['ite']= ite
    size = data_dict['ite'].shape[0]

    # Shuffle data points before train/val split
    save_dir= os.path.join(base_dir, 'shuffle_indices',  str(dataset_idx), 'seed_' + str(seed), '')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename= save_dir + 'shuffle_indices.p'
    if os.path.exists(filename):
        print("Loading shuffle indices from file for ACIC 2016 dataset")
        inds= pickle.load(open(filename, "rb"))
    else:
        print("Generating shuffle indices for ACIC 2016 dataset")
        inds = np.arange(size)
        np.random.shuffle(inds)
        pickle.dump(inds, open(filename, "wb") )

    for key in data_dict.keys():
        if key != 'ate':
            data_dict[key] = data_dict[key][inds]

    # Train/Val split
    final_data = {'train': {}, 'test': {}}
    train_size = int(0.8 * size)

    for key in data_dict.keys():
        if key == 'ate':
            final_data['train'][key] = data_dict[key]
        else:
            final_data['train'][key] = data_dict[key][:train_size]
            final_data['train'][key] = data_dict[key][:train_size]

    for key in data_dict.keys():
        if key == 'ate':
            final_data['test'][key] = data_dict[key]
        else:
            final_data['test'][key] = data_dict[key][train_size:]
            final_data['test'][key] = data_dict[key][train_size:]

    return final_data


