import numpy as np
import pandas as pd
import time
import sys
import os
from pathlib import Path
import argparse
import pickle

from utils.helpers import *
from data.loading import load_from_folder
from data.lbidd import lbidd_main_loader

from sklearn.model_selection import cross_val_score
from sklearn.base import clone

import warnings

from flaml import AutoML

def get_propensity_model(prop_model, w, t, automl_settings={}):

    for key in t.keys():
        data_size= w[key].shape[0]
        t[key]= np.reshape(t[key], (data_size))
    
    #Propensity Model    
    prop_model.fit(X_train=w['tr'], y_train=t['tr'], **automl_settings)
    model = clone(prop_model.model.estimator)

    model.fit(w['te'], t['te'])
    score= model.score(w['te'], t['te'])

    return score, model


def get_outcome_model(out_model, w, t, y, case='t_0', automl_settings={}):

    for key in t.keys():
        data_size= w[key].shape[0]
        t[key]= np.reshape(t[key], (data_size))
        y[key]= np.reshape(y[key], (data_size))
        
    #Outcome Models
    if case == 't_0':
        indices= t['tr'] == 0
        indices_eval= t['te'] == 0
    elif case == 't_1':
        indices= t['tr'] == 1
        indices_eval= t['te'] == 1

    out_model.fit(X_train=w['tr'][indices, :], y_train=y['tr'][indices], **automl_settings)
    model = clone(out_model.model.estimator)

    model.fit(w['te'][indices_eval, :], y['te'][indices_eval])
    score = model.score(w['te'][indices_eval, :], y['te'][indices_eval])

    return score, model


def get_s_learner_model(out_model, w, t, y, automl_settings={}):
    
    for key in t.keys():
        data_size= w[key].shape[0]
        t[key]= np.reshape(t[key], (data_size, 1))
        y[key]= np.reshape(y[key], (data_size))
        
    #Outcome Models
    #Since these nuisance models would be used as part of the metric computation, we train them on the actual evaluation/validation set and test on the actual training test
    
    w_upd={'te':'', 'tr':''}
    w_upd['te']= np.hstack([w['te'],t['te']])
    w_upd['tr']= np.hstack([w['tr'],t['tr']])

    out_model.fit(X_train=w_upd['tr'], y_train=y['tr'], **automl_settings)
    model = clone(out_model.model.estimator)

    model.fit(w_upd['te'], y['te'])
    score = model.score(w_upd['te'], y['te'])

    return score, model


def get_r_score_model(out_model, w, y, automl_settings={}):
    
    for key in t.keys():    
        data_size= w[key].shape[0]
        y[key]= np.reshape(y[key], (data_size))
        
    #Outcome Models
    #Since these nuisance models would be used as part of the metric computation, we train them on the actual evaluation/validation set and test on the actual training test    

    out_model.fit(X_train=w['tr'], y_train=y['tr'], **automl_settings)
    model = clone(out_model.model.estimator)

    #Fitting on test set for the purposes of computing the metric
    model.fit(w['te'], y['te'])
    score = model.score(w['te'], y['te'])

    return score, model


# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='twins',
                    help='Datasets: lalonde_psid1; lalonde_cps1; twins; lbidd')
parser.add_argument('--seed', type=int, default=0,
                    help='Total seeds for causal effect estimation experiments')
parser.add_argument('--root_dir', type=str, default='/scratch/cate_eval_analysis/')
parser.add_argument('--res_dir', type=str, default='results_final')
parser.add_argument('--slurm_exp', type=int, default=0,
                   help='')

args = parser.parse_args()
print(vars(args))

NUISANCE_MODEL_CASES= ['t_learner_0', 't_learner_1', 's_learner', 'dml', 'prop']

#Experiments on Slurm
if args.slurm_exp:
    # dataset_list = ['twins', 'lalonde_psid1', 'lalonde_cps1', 'orthogonal_ml_dgp']
    # for idx in range(100):
    #     dataset_list.append('lbidd_' + str(idx))
    # for idx in range(77):
    #     dataset_list.append('acic_2016_' + str(idx))

    dataset_list = pickle.load(open('datasets/acic_2016_heterogenous_list.p', "rb"))
    # dataset_list = pickle.load(open('datasets/acic_2018_heterogenous_list.p', "rb"))

    slurm_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    args.dataset = dataset_list[slurm_idx]

dataset_name= args.dataset
seed = args.seed
root_dir= os.path.expanduser('~') + args.root_dir
res_dir= args.res_dir

#Create Logs Directory
RESULTS_DIR = root_dir + str(Path(res_dir))
# Loop over datasets, seeds, estimators with their hyperparams and nusiance models
print('SEED: ', seed)
random.seed(seed)
np.random.seed(seed)

print('DATASET:', dataset_name)
dataset_name, dataset_obj = load_dataset_obj(args.dataset, root_dir)

# grid_models= get_nusiance_models_grid(outcome_models, prop_models, approx=True, grid_size= grid_size)
dataset_samples= sample_dataset(dataset_name, dataset_obj, seed=seed, case='train')
train_w, train_t, train_y= dataset_samples['w'], dataset_samples['t'], dataset_samples['y']

dataset_samples= sample_dataset(dataset_name, dataset_obj, seed=seed, case='eval')
eval_w, eval_t, eval_y, ate, ite= dataset_samples['w'], dataset_samples['t'], dataset_samples['y'], dataset_samples['ate'], dataset_samples['ite']

w= { 'tr': train_w, 'te': eval_w, 'all': np.concatenate((train_w, eval_w), axis=0)}
t= { 'tr': train_t, 'te': eval_t, 'all': np.concatenate((train_t, eval_t), axis=0) }
y= { 'tr': train_y, 'te': eval_y, 'all': np.concatenate((train_y, eval_y), axis=0) }

# print('Shape Check')
# print(w['tr'].shape, w['te'].shape, w['all'].shape)
# print(t['tr'].shape, t['te'].shape, t['all'].shape)
# print(y['tr'].shape, y['te'].shape, y['all'].shape)
# sys.exit()

model_sel_res={}
for key in NUISANCE_MODEL_CASES:
    model_sel_res[key]= {}
    model_sel_res[key]['score']= -sys.maxsize - 1
    model_sel_res[key]['model']= -sys.maxsize - 1

for model_case in NUISANCE_MODEL_CASES:
    if model_case == 'prop':
        automl_settings = {
            "time_budget": 1800,  # in seconds
            "task": 'classification',
            "eval_method": 'cv',
            "n_splits": 3,
            "verbose": 0
        }
    else:
        automl_settings = {
            "time_budget": 1800,  # in seconds
            "task": 'regression',
            "eval_method": 'cv',
            "n_splits": 3,
            "verbose": 0
        }

    #AutoML
    automl = AutoML()
    if model_case == 'prop':
        score, best_model = get_propensity_model(automl, w, t,  automl_settings= automl_settings)
    elif model_case == 't_learner_0':
        score, best_model = get_outcome_model(automl, w, t, y, case='t_0', automl_settings= automl_settings)
    elif model_case == 't_learner_1':
        score, best_model = get_outcome_model(automl, w, t, y, case='t_1', automl_settings= automl_settings)
    elif model_case == 's_learner':
        score, best_model = get_s_learner_model(automl, w, t, y, automl_settings= automl_settings)
    elif model_case == 'dml':
        score, best_model = get_r_score_model(automl, w, y, automl_settings= automl_settings)

    model_sel_res[model_case]['score']= score
    model_sel_res[model_case]['model']= best_model
    print(model_case, score, best_model)

save_dir= RESULTS_DIR + '/' + dataset_name + '/' + 'seed_' + str(seed) + '/nuisance_models/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for model_case in NUISANCE_MODEL_CASES:
    print('Best ', model_case, model_sel_res[model_case]['score'], model_sel_res[model_case]['model'])
    pickle.dump(model_sel_res[model_case]['model'], open(save_dir + model_case + '.p', "wb"))

#Storing which nusiance model out of S/T Learner does a better fit of the data as this is used for score computation later
#TODO: Update loading S, T learner model to just sampling from model_sel_red dictionary

s_learner= pickle.load(open(save_dir + 's_learner.p', "rb") )
t_learner_0= pickle.load(open(save_dir + 't_learner_0.p', "rb") )
t_learner_1= pickle.load(open(save_dir + 't_learner_1.p', "rb") )

for key in t.keys():
    data_size= w[key].shape[0]
    t[key]= np.reshape(t[key], (data_size, 1))
    y[key]= np.reshape(y[key], (data_size))

#Computing S Learner Score
s_score = s_learner.score( np.hstack([w['te'],t['te']]), y['te'] )

for key in t.keys():
    data_size= w[key].shape[0]
    t[key]= np.reshape(t[key], (data_size))
    y[key]= np.reshape(y[key], (data_size))

#Computing T Learner Score
indices_eval= t['te'] == 0
t_score_0 = t_learner_0.score(w['te'][indices_eval, :], y['te'][indices_eval])

indices_eval= t['te'] == 1
t_score_1 = t_learner_1.score(w['te'][indices_eval, :], y['te'][indices_eval])

t_score= (t_score_0 + t_score_1)/2

if t_score > s_score:
    better_nuisance_model= 't_score'
else:
    better_nuisance_model= 's_score'

log= {}
log['s_score']= s_score
log['t_score']= t_score
log['opt_model']= better_nuisance_model

pickle.dump( log, open(save_dir + 'reg_stats' + '.p', "wb") )