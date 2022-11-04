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

def get_propensity_model(prop_model, w, t, automl= 0, automl_settings={}, selection_case= 'estimator'):

    for key in t.keys():
        data_size= w[key].shape[0]
        t[key]= np.reshape(t[key], (data_size))
    
    #Propensity Model    
    if automl:
        prop_model.fit(X_train=w['tr'], y_train=t['tr'], **automl_settings)
        prop_model = clone(prop_model.model.estimator)

    prop_model.fit(w['tr'], t['tr'])
    score= prop_model.score(w['te'], t['te'])

    if automl:
        return score, prop_model
    else:
        return score
    
def get_outcome_model(out_model, w, t, y, case='t_0', automl= 0, automl_settings={}, selection_case= 'estimator'):

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

    if automl:
        out_model.fit(X_train=w['tr'][indices, :], y_train=y['tr'][indices], **automl_settings)
        out_model = clone(out_model.model.estimator)

    out_model.fit(w['tr'][indices, :], y['tr'][indices])
    score = out_model.score(w['te'][indices_eval, :], y['te'][indices_eval])

    if automl:
        return score, out_model
    else:
        return score


def get_s_learner_model(out_model, w, t, y, automl= 0, automl_settings={}, selection_case= 'estimator'):
    
    for key in t.keys():
        data_size= w[key].shape[0]
        t[key]= np.reshape(t[key], (data_size, 1))
        y[key]= np.reshape(y[key], (data_size))
        
    #Outcome Models
    #Since these nuisance models would be used as part of the metric computation, we train them on the actual evaluation/validation set and test on the actual training test
    
    w_upd={'te':'', 'tr':''}
    w_upd['te']= np.hstack([w['te'],t['te']])
    w_upd['tr']= np.hstack([w['tr'],t['tr']])

    if automl:
        out_model.fit(X_train=w_upd['tr'], y_train=y['tr'], **automl_settings)
        out_model = clone(out_model.model.estimator)

    out_model.fit(w_upd['tr'], y['tr'])
    score = out_model.score(w_upd['te'], y['te'])

    if automl:
        return score, out_model
    else:
        return score


def get_r_score_model(out_model, w, y, automl= 0, automl_settings={}, selection_case= 'estimator'):
    
    for key in t.keys():    
        data_size= w[key].shape[0]
        y[key]= np.reshape(y[key], (data_size))
        
    #Outcome Models
    #Since these nuisance models would be used as part of the metric computation, we train them on the actual evaluation/validation set and test on the actual training test    

    if automl:
        out_model.fit(X_train=w['tr'], y_train=y['tr'], **automl_settings)
        out_model = clone(out_model.model.estimator)

    out_model.fit(w['tr'], y['tr'])
    score = out_model.score(w['te'], y['te'])

    if automl:
        return score, out_model
    else:
        return score


# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='twins',
                    help='Datasets: lalonde_psid1; lalonde_cps1; twins; lbidd')
parser.add_argument('--seed', type=int, default=0,
                    help='Total seeds for causal effect estimation experiments')
parser.add_argument('--root_dir', type=str, default='/scratch/causal_val_datasets/')
parser.add_argument('--res_dir', type=str, default='results_final')
parser.add_argument('--selection_case', type=str, default='metric', help='model selection for estimator or metric')
parser.add_argument('--slurm_exp', type=int, default=0,
                   help='')

args = parser.parse_args()
print(vars(args))

#Experiments on Slurm
if args.slurm_exp:
    dataset_list = ['twins', 'lalonde_psid1', 'lalonde_cps1', 'orthogonal_ml_dgp']
    # for idx in range(100):
    #     dataset_list.append('lbidd_' + str(idx))
    # for idx in range(77):
    #     dataset_list.append('acic_2016_' + str(idx))

    # dataset_list = pickle.load(open('datasets/acic_2018_heterogenous_list.p', "rb"))
    # dataset_list = pickle.load(open('datasets/acic_2016_heterogenous_list.p', "rb"))

    slurm_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    args.dataset = dataset_list[slurm_idx]

dataset_name= args.dataset
seed = args.seed
root_dir= os.path.expanduser('~') + args.root_dir
res_dir= args.res_dir

#Create Logs Directory
RESULTS_DIR = root_dir + str(Path(res_dir))

#Obtain list of nuisance (outcome, propensity) models
outcome_models, prop_models= get_nuisance_models_list()

# for key in outcome_models.keys():
#     print(key)
#
# for key in prop_models.keys():
#     print(key)

grid_size= 80

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
for key in ['t_learner_0', 't_learner_1', 's_learner', 'dml', 'prop']:
    model_sel_res[key]= {}
    model_sel_res[key]['score']= -sys.maxsize - 1
    model_sel_res[key]['model']= -sys.maxsize - 1

for model_case in ['t_learner_0', 't_learner_1', 's_learner', 'dml', 'prop']:
    if model_case == 'prop':
        automl_settings = {
            "time_budget": 1800,  # in seconds
            "task": 'classification',
            "eval_method": 'cv',
            "n_splits": 3,
            "verbose": 0
        }
        nuisance_list= prop_models
    else:
        automl_settings = {
            "time_budget": 1800,  # in seconds
            "task": 'regression',
            "eval_method": 'cv',
            "n_splits": 3,
            "verbose": 0
        }
        nuisance_list= outcome_models

    #AutoML
    automl = AutoML()
    if model_case == 'prop':
        score, best_model = get_propensity_model(automl, w, t, automl= 1, automl_settings= automl_settings, selection_case= args.selection_case)
    elif model_case == 't_learner_0':
        score, best_model = get_outcome_model(automl, w, t, y, case='t_0', automl= 1, automl_settings= automl_settings, selection_case= args.selection_case)
    elif model_case == 't_learner_1':
        score, best_model = get_outcome_model(automl, w, t, y, case='t_1', automl= 1, automl_settings= automl_settings, selection_case= args.selection_case)
    elif model_case == 's_learner':
        score, best_model = get_s_learner_model(automl, w, t, y, automl= 1, automl_settings= automl_settings, selection_case= args.selection_case)
    elif model_case == 'dml':
        score, best_model = get_r_score_model(automl, w, y, automl= 1, automl_settings= automl_settings, selection_case= args.selection_case)

    model_sel_res[model_case]['score']= score
    model_sel_res[model_case]['model']= best_model
    print(model_case, score, best_model)

    # print(automl.model.estimator)
    # best_est= automl.best_estimator
    # print(best_est)
    # print(automl.best_model_for_estimator(best_est))
    # sys.exit(-1)

    # #Searching over the existing grid
    # count=0
    # for key in nuisance_list.keys():
    #     for model in nuisance_list[key]:
    #         count+=1
    #
    #         curr_model = model['model_func']
    #
    #         start_time = time.time()
    #
    #         if model_case == 'prop':
    #             score = get_propensity_model(curr_model, w, t)
    #         elif model_case == 't_learner_0':
    #             score = get_outcome_model(curr_model, w, t, y, case='t_0')
    #         elif model_case == 't_learner_1':
    #             score = get_outcome_model(curr_model, w, t, y, case='t_1')
    #         elif model_case == 's_learner':
    #             score = get_s_learner_model(curr_model, w, t, y)
    #         elif model_case == 'dml':
    #             score = get_r_score_model(curr_model, w, y)
    #
    #         #             print(model['model_y']['name'], model['model_y']['hparam'])
    #         #             print(score, 'time: ', time.time() - start_time)
    #         if score > model_sel_res[model_case]['score']:
    #             model_sel_res[model_case]['score'] = score
    #             model_sel_res[model_case]['model'] = curr_model
    #         #             print('Curr Best Model: ', best_outcome_score_0)
    #
    # print(count)

save_dir= RESULTS_DIR + '/' + dataset_name + '/' + 'seed_' + str(seed) + '/' + args.selection_case + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for model_case in ['t_learner_0', 't_learner_1', 's_learner', 'dml', 'prop']:
    print('Best ', model_case)
    print(model_sel_res[model_case]['score'])
    print(model_sel_res[model_case]['model'])

    if model_case == 'prop':
        pickle.dump(model_sel_res[model_case]['model'], open(save_dir + 'prop' + '.p', "wb"))
    elif model_case == 't_learner_0':
        pickle.dump(model_sel_res[model_case]['model'], open(save_dir + 'mu_0' + '.p', "wb"))
    elif model_case == 't_learner_1':
        pickle.dump(model_sel_res[model_case]['model'], open(save_dir + 'mu_1' + '.p', "wb"))
    elif model_case == 's_learner':
        pickle.dump(model_sel_res[model_case]['model'], open(save_dir + 'mu_s' + '.p', "wb"))
    elif model_case == 'dml':
        pickle.dump(model_sel_res[model_case]['model'], open(save_dir + 'mu_r_score' + '.p', "wb"))
