import numpy as np
import pandas as pd
import time
import sys
import os
from pathlib import Path
import argparse
import pickle

from utils.evaluation import calculate_metrics

from utils.helpers import *

pd.set_option("display.max_rows", None, "display.max_columns", None)


def acic_dataset_subset_selection(root_dir, case='acic_2018'):

    dataset_list=[]
    if case == 'acic_2016':
        N=77
        for idx in range(N):
            dataset_list.append('acic_2016_' + str(idx))
    elif case == 'acic_2018':
        N=432
        for idx in range(N):
            dataset_list.append('lbidd_'+str(idx))

    count=0
    heterogenous_list=[]
    homogenous_list= []
    for dataset_name in dataset_list:

        seed = 0
        random.seed(seed)
        np.random.seed(seed)

        # Load dataset with true ITE, ATE from the generative model
        print('DATASET:', dataset_name)
        dataset_name, dataset_obj = load_dataset_obj(dataset_name, root_dir)
        dataset_samples= sample_dataset(dataset_name, dataset_obj, seed=seed, case='eval')
        eval_w, eval_t, eval_y, ate, ite = dataset_samples['w'], dataset_samples['t'], dataset_samples['y'], dataset_samples['ate'], dataset_samples['ite']

        print('Mean', np.mean(ite))
        print('Var', np.var(ite))

        if np.var(ite) > 1e-2:
            heterogenous_list.append(dataset_name)
            count+=1
        else:
            homogenous_list.append(dataset_name)

    if case == 'acic_2016':
        pickle.dump(heterogenous_list, open('datasets/acic_2016_heterogenous_list.p', "wb"))
        pickle.dump(homogenous_list, open('datasets/acic_2016_homogenous_list.p', "wb"))
    elif case == 'acic_2018':
        pickle.dump( heterogenous_list, open('datasets/acic_2018_heterogenous_list.p', "wb") )
        pickle.dump( homogenous_list, open('datasets/acic_2018_homogenous_list.p', "wb") )

    print(heterogenous_list)
    print(homogenous_list)
    print('Count of dataset with variance in ITE > 1e-2 : ', count, 100*count/N)

    return


def acic_dataset_details(root_dir, case='acic_2016'):

    if case == 'acic_2016':
        dataset_list = pickle.load(open('datasets/acic_2016_heterogenous_list.p', "rb"))
    elif case == 'acic_2018':
        dataset_list = pickle.load(open('datasets/acic_2018_heterogenous_list.p', "rb"))

    res = {'dataset': [], 'train_size': [], 'eval_size': [], 'data_dim': [], 'ite_mean': [], 'ite_var': [],
           'treatment_class_perc_train': [], 'treatment_class_perc_eval': []}
    for dataset_name in dataset_list:

        seed = 1
        random.seed(seed)
        np.random.seed(seed)

        # Load dataset with true ITE, ATE from the generative model
        print('DATASET:', dataset_name)
        dataset_name, dataset_obj = load_dataset_obj(dataset_name, root_dir)

        #Train
        dataset_samples= sample_dataset(dataset_name, dataset_obj, seed=seed, case='train')
        train_w, train_t, train_y = dataset_samples['w'], dataset_samples['t'], dataset_samples['y']

        #Eval
        dataset_samples= sample_dataset(dataset_name, dataset_obj, seed=seed, case='eval')
        eval_w, eval_t, eval_y, ate, ite = dataset_samples['w'], dataset_samples['t'], dataset_samples['y'], dataset_samples['ate'], dataset_samples['ite']

        res['dataset'].append(dataset_name)
        res['train_size'].append(train_w.shape[0])
        res['eval_size'].append(eval_w.shape[0])
        res['data_dim'].append(eval_w.shape[1])
        res['treatment_class_perc_train'].append( np.sum(train_t == 1) / train_t.shape[0] )
        res['treatment_class_perc_eval'].append( np.sum(eval_t == 1) / eval_t.shape[0] )
        res['ite_mean'].append(np.mean(ite))
        res['ite_var'].append(np.var(ite))

    df = pd.DataFrame(res)
    print(df.to_latex())

    return

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=4,
                    help='Total seeds for causal effect estimation experiments')
parser.add_argument('--root_dir', type=str, default='/scratch/causal_val_datasets/')

args = parser.parse_args()
print(vars(args))


dataset_list= ['twins', 'lalonde_psid1', 'lalonde_cps1']
# dataset_list= ['twins', 'lalonde_psid1', 'lalonde_cps1', 'acic_2016_0', 'lbidd_0']
# dataset_list=[]
# for idx in range(432):
#     dataset_list.append('lbidd_'+str(idx))
# for idx in range(15):
#     dataset_list.append('acic_2016_'+str(idx))

seed_range = args.seed
root_dir =  os.path.expanduser('~') + args.root_dir

#ACIC Dataset Subset Selection
acic_dataset_subset_selection(root_dir, case='acic_2018')

#ACIC Dataset Analysis
# acic_dataset_details(root_dir, case='acic_2018')
# acic_dataset_details(root_dir, case='acic_2016')
sys.exit()

res={'dataset':[], 'train_size': [], 'eval_size': [], 'data_dim':[], 'ite_mean':[] ,'ite_var': [], 'treatment_class_perc_train': [], 'treatment_class_perc_eval': []}
# res={'dataset':[], 'seed':[], 'train_size': [], 'eval_size': [], 'data_dim': [], 'class_ratio_train':[], 'class_ratio_eval':[]}
# res={'dataset':[], 'train_size': [], 'eval_size': [], 'data_dim': []}
for seed in range(1, 2):
    for dataset_name in dataset_list:

        print('SEED: ', seed)
        random.seed(seed)
        np.random.seed(seed)

        # Load dataset with true ITE, ATE from the generative model
        print('DATASET:', dataset_name)
        dataset_name, dataset_obj = load_dataset_obj(dataset_name, root_dir)

        dataset_samples= sample_dataset(dataset_name, dataset_obj, seed=seed, case='train')
        train_w, train_t, train_y = dataset_samples['w'], dataset_samples['t'], dataset_samples['y']

        dataset_samples= sample_dataset(dataset_name, dataset_obj, seed=seed, case='eval')
        eval_w, eval_t, eval_y, ate, ite = dataset_samples['w'], dataset_samples['t'], dataset_samples['y'], dataset_samples['ate'], dataset_samples['ite']

        train_t= np.reshape(train_t, (train_t.shape[0]))
        eval_t= np.reshape(eval_t, (eval_t.shape[0]))

        res['dataset'].append(dataset_name)
        res['train_size'].append(train_w.shape[0])
        res['eval_size'].append(eval_w.shape[0])
        res['data_dim'].append(eval_w.shape[1])
        res['treatment_class_perc_train'].append( np.sum(train_t == 1) / train_t.shape[0] )
        res['treatment_class_perc_eval'].append( np.sum(eval_t == 1) / eval_t.shape[0] )
        res['ite_mean'].append(np.mean(ite))
        res['ite_var'].append(np.var(ite))

df= pd.DataFrame(res)
print(df.transpose().to_latex())
