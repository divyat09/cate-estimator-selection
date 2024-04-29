import numpy as np
import pandas as pd
import random
import time
import sys
import os
from pathlib import Path
import argparse
import pickle

from data.samplers import *

pd.set_option("display.max_rows", None, "display.max_columns", None)

def acic_dataset_subset_selection(root_dir: str, seed: int=0):
    """
    Selects a subset of the ACIC 2016 dataset based on the variance of ITE

    Inputs:
        root_dir: str: root directory for the dataset
    """

    dataset_list=[]
    N=77
    for idx in range(N):
        dataset_list.append('acic_2016_' + str(idx))

    count=0
    heterogenous_list=[]
    homogenous_list= []
    for dataset_name in dataset_list:

        # Load dataset with true ITE, ATE from the generative model
        print('DATASET:', dataset_name)
        dataset_obj = load_dataset_obj(dataset= dataset_name, root_dir= root_dir, seed= seed)
        dataset_samples= sample_dataset(dataset_obj, case='eval')
        _, _, _, _, ite = dataset_samples['w'], dataset_samples['t'], dataset_samples['y'], dataset_samples['ate'], dataset_samples['ite']

        print('Mean', np.mean(ite))
        print('Var', np.var(ite))

        if np.var(ite) > 1e-2:
            heterogenous_list.append(dataset_name)
            count+=1
        else:
            homogenous_list.append(dataset_name)

    pickle.dump(heterogenous_list, open('datasets/acic_2016_heterogenous_list.p', "wb"))
    pickle.dump(homogenous_list, open('datasets/acic_2016_homogenous_list.p', "wb"))

    print(len(heterogenous_list), len(homogenous_list))
    print('Count of dataset with variance in ITE > 1e-2 : ', count, 100*count/N)

    return


def realcause_dataset_generation(root_dir: str, seed: int=0):
    """
    Returns dataframe with dataset details for the various RealCause datasets

    Inputs:
        root_dir: str: root directory for the dataset
    """

    dataset_list = ['twins', 'lalonde_psid1', 'lalonde_cps1']
    for dataset_name in dataset_list:
        # Load dataset with true ITE, ATE from the generative model
        print('DATASET:', dataset_name)
        dataset_obj = load_dataset_obj(dataset= dataset_name, root_dir= root_dir, seed= seed)

    return

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=11,
                    help='Random Seed for causal effect estimation experiments')
parser.add_argument('--root_dir', type=str, default='/scratch/cate_eval_analysis/')
parser.add_argument('--meta_dataset', type=str, default='acic', help='Get datasets details; allowed values: acic, realcause')

args = parser.parse_args()
seed= args.seed
root_dir =  os.path.expanduser('~') + args.root_dir

#Set the random seed
random.seed(seed)
np.random.seed(seed)

if args.meta_dataset == 'acic':
    acic_dataset_subset_selection(root_dir= root_dir, seed=seed)
elif args.meta_dataset == 'realcause':
    realcause_dataset_generation(root_dir, seed=seed)
