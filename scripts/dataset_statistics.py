import numpy as np
import pandas as pd
import random
import time
import sys
import os
from pathlib import Path
import argparse
import pickle

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.samplers import *

pd.set_option("display.max_rows", None, "display.max_columns", None)


# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=11,
                    help='Total seeds for causal effect estimation experiments')
parser.add_argument('--root_dir', type=str, default='/scratch/cate_eval_analysis/')
parser.add_argument('--meta_dataset', type=str, default='acic', help='Get datasets details; allowed values: acic, realcause')

args = parser.parse_args()
root_dir =  os.path.expanduser('~') + args.root_dir
seed= args.seed

#Set the random seed
random.seed(seed)
np.random.seed(seed)

res = {'dataset': [], 'ite_mean': [], 'ite_var': [],
        'treatment_class_perc_train': [], 'treatment_class_perc_eval': []}

if args.meta_dataset == 'acic':
    dataset_list = pickle.load(open('datasets/acic_2016_heterogenous_list.p', "rb"))
elif args.meta_dataset == 'realcause':
    # dataset_list = ['twins', 'lalonde_psid1', 'lalonde_cps1']
    dataset_list = ['twins', 'lalonde_psid1']

for dataset_name in dataset_list:

    # Load dataset with true ITE, ATE from the generative model
    print('DATASET:', dataset_name)
    dataset_obj = load_dataset_obj(dataset= dataset_name, root_dir= root_dir, seed= seed)

    dataset_samples= sample_dataset(dataset_obj, case='train')
    train_w, train_t, train_y = dataset_samples['w'], dataset_samples['t'], dataset_samples['y']

    dataset_samples= sample_dataset(dataset_obj, case='eval')
    eval_w, eval_t, eval_y, ate, ite = dataset_samples['w'], dataset_samples['t'], dataset_samples['y'], dataset_samples['ate'], dataset_samples['ite']

    print(train_w.shape, eval_w.shape)

    train_t= np.reshape(train_t, (train_t.shape[0]))
    eval_t= np.reshape(eval_t, (eval_t.shape[0]))

    res['dataset'].append(dataset_name)
    res['treatment_class_perc_train'].append( np.sum(train_t == 1) / train_t.shape[0] )
    res['treatment_class_perc_eval'].append( np.sum(eval_t == 1) / eval_t.shape[0] )
    res['ite_mean'].append(np.mean(ite))
    res['ite_var'].append(np.var(ite))

df= pd.DataFrame(res)
print(df.round(2).to_latex())