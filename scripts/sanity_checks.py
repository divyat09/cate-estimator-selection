import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import sys
import os
import pickle

root_dir = os.path.expanduser('~') + '/scratch/'
RESULTS_DIR = root_dir + str(Path('cate_eval_analysis/results_final'))

DATASET_NAMES = [
                'lalonde_psid1',
                'lalonde_cps1',
                'twins',
                ]

DATASET_NAMES+= pickle.load(open('datasets/acic_2016_heterogenous_list.p', "rb"))

TOTAL_SEEDS= 20
ESTIMATOR_LIST= ['dr_learner', 'dml_learner', 's_learner_upd', 'x_learner', 's_learner', 't_learner', 'causal_forest_learner']
#ESTIMATOR_LIST= ['dr_learner', 'dml_learner', 's_learner_upd', 'x_learner']
#ESTIMATOR_LIST= ['s_learner', 't_learner', 'causal_forest_learner']
SCORES = ['baseline_score', 'mu_score', 'mu_iptw_score', 'value_score', 'value_dr_score', 'influence_score', 'tau_t_score', 'tau_s_score', 'tau_match_score', 'tau_iptw_score', 'tau_dr_score', 'rscore']

def get_dataset_statistics(target, base_dir= RESULTS_DIR, dataset_names=DATASET_NAMES, scores=SCORES, estimator_list=ESTIMATOR_LIST):
    for dataset_name in dataset_names:
        for seed in range(TOTAL_SEEDS):
            for estimator in estimator_list:
                curr_dir= base_dir + '/' + dataset_name + '/' 'seed_' + str(seed) + '/' + estimator + '/'
                count=0
                for _, _, f_list in os.walk(curr_dir):
                    for fname in f_list:
                        if '.p' in fname:
                            count+=1

                if count!= target:
                    print('Dataset: ', dataset_name, 'Seed: ', seed, 'Estimator: ', estimator, 'Total files: ', count)
                    print('\n')


def get_nuisance_model_statistics(target, base_dir= RESULTS_DIR, dataset_names=DATASET_NAMES, scores=SCORES):
    total_count= 0
    for dataset_name in dataset_names:
        for seed in range(TOTAL_SEEDS):

            #Metric Nuisance Models
            curr_dir= base_dir + '/' + dataset_name + '/' 'seed_' + str(seed) + '/nuisance_models/'
            count=0
            for _, _, f_list in os.walk(curr_dir):
                for fname in f_list:
                    if '.p' in fname:
                        count+=1

            if count!= target:
                total_count += 1
                print('Metric Case: ', 'Dataset: ', dataset_name, 'Seed: ', seed, 'Total files: ', count)
                print('\n')

    print('Total Errors: ', total_count)

def get_estimator_statistics(target, base_dir= RESULTS_DIR, dataset_names=DATASET_NAMES, scores=SCORES, estimator_list=ESTIMATOR_LIST):
    total_count= 0
    for dataset_name in dataset_names:
        for seed in range(TOTAL_SEEDS):
            for estimator in estimator_list:

                curr_dir= base_dir + '/' + dataset_name + '/' 'seed_' + str(seed) + '/' + str(estimator) +  '/'
                count=0
                for _, _, f_list in os.walk(curr_dir):
                    for fname in f_list:
                        if '.p' in fname:
                            count+=1

                if count!= target:
                    total_count += 1
                    print('Estimator Case: ', 'Dataset: ', dataset_name, 'Seed: ', seed, 'Estimator', estimator, 'Total files: ', count)
                    print('\n')

    print('Total Errors: ', total_count)


def delete_estimator_logs(base_dir= RESULTS_DIR, dataset_names=DATASET_NAMES, scores=SCORES, estimator_list=ESTIMATOR_LIST):
    total_count= 0
    for dataset_name in dataset_names:
        for seed in range(TOTAL_SEEDS):
            for estimator in estimator_list:

                curr_dir= base_dir + '/' + dataset_name + '/' 'seed_' + str(seed) + '/' + str(estimator) +  '/'
                count=0
                for _, _, f_list in os.walk(curr_dir):
                    for fname in f_list:
                        if '.p' in fname:
                            logs = curr_dir + fname
                            print(logs)
                            os.remove(logs)
    return


def get_ensemble_cate_statistics(target, base_dir= RESULTS_DIR, dataset_names=DATASET_NAMES, scores=SCORES):
    total_count= 0
    for dataset_name in dataset_names:
        for seed in range(TOTAL_SEEDS):

            #Metric Nuisance Models
            curr_dir= base_dir + '/' + dataset_name + '/' 'seed_' + str(seed) + '/'
            count=0
            for _, _, f_list in os.walk(curr_dir):
                for fname in f_list:
                    if 'ensemble' in fname:
                        count+=1

            if count!= target:
                total_count += 1
                print('Metric Case: ', 'Dataset: ', dataset_name, 'Seed: ', seed, 'Total files: ', count)
                print('\n')

    print('Total Errors: ', total_count)
    return


def delete_ensemble_logs(base_dir= RESULTS_DIR, dataset_names=DATASET_NAMES, scores=SCORES):
    total_count= 0
    for dataset_name in dataset_names:
        for seed in range(TOTAL_SEEDS):

            #Metric Nuisance Models
            curr_dir= base_dir + '/' + dataset_name + '/' 'seed_' + str(seed) + '/'
            count=0
            for _, _, f_list in os.walk(curr_dir):
                for fname in f_list:
                    if fname == 'ensemble.csv':
                        logs= curr_dir + fname
                        print(logs)
                        os.remove(logs)
    return

#Main Code
target= int(sys.argv[1])
# get_dataset_statistics(target)
get_nuisance_model_statistics(target)
# get_estimator_statistics(target)
# get_ensemble_cate_statistics(target)
# delete_ensemble_logs()
# delete_estimator_logs()
