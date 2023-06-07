import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import argparse
import sys
import pickle
import pathlib
import os

# Using sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet, RidgeClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor,\
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from scipy.stats import sem
from scipy.special import softmax

# Using cvxpy for convex optimization solvers under constraints
import cvxpy as cp

from utils.helpers import *
from utils.evaluation import *

SCORES = ['value_score', 'value_dr_score', 'value_dr_clip_prop_score',
          'influence_score', 'influence_clip_prop_score',
          'tau_t_score', 'tau_s_score', 'tau_match_score',
          'tau_iptw_score', 'tau_iptw_clip_score',
          'tau_switch_iptw_s_score', 'tau_switch_iptw_t_score', 'tau_cab_iptw_s_score', 'tau_cab_iptw_t_score',
          'tau_dr_s_score', 'tau_dr_s_clip_score', 'tau_dr_t_score', 'tau_dr_t_clip_score',
          'tau_switch_dr_s_score', 'tau_switch_dr_t_score', 'tau_cab_dr_s_score', 'tau_cab_dr_t_score',
          'tau_tmle_s_score', 'tau_tmle_t_score',
          'cal_dr_s_score', 'cal_dr_t_score', 'cal_tmle_s_score', 'cal_tmle_t_score',
          'qini_dr_s_score', 'qini_dr_t_score', 'qini_tmle_s_score', 'qini_tmle_t_score',
          'x_score', 'rscore']

ESTIMATOR_LIST= ['dr_learner', 'dml_learner', 'causal_forest_learner', 's_learner', 's_learner_upd', 't_learner', 'x_learner']
# ESTIMATOR_LIST= [ 'causal_forest_learner', 's_learner', 't_learner']

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='twins',
                    help='Datasets: lalonde_psid1; lalonde_cps1; twins; lbidd')
parser.add_argument('--seed', type=int, default=0,
                    help='Seed for causal effect estimation experiments')
parser.add_argument('--root_dir', type=str, default='/scratch/cate_eval_analysis/')
parser.add_argument('--res_dir', type=str, default='results_final')
parser.add_argument('--slurm_exp', type=int, default=0,
                    help='0: None; 1: Parallelize across datasets; 2: Parallelize across final models')

args = parser.parse_args()

# Experiments on Slurm
if args.slurm_exp == 1:
    # dataset_list = ['twins', 'lalonde_psid1', 'lalonde_cps1', 'orthogonal_ml_dgp']
    # dataset_list = pickle.load(open('datasets/acic_2016_heterogenous_list.p', "rb"))
    dataset_list = pickle.load(open('datasets/acic_2018_heterogenous_list.p', "rb"))

    slurm_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    args.dataset = dataset_list[slurm_idx]

dataset_name = args.dataset
seed= args.seed

root_dir = os.path.expanduser('~') + args.root_dir
RESULTS_DIR = root_dir + str(Path(args.res_dir))

print('SEED: ', seed)
random.seed(seed)
np.random.seed(seed)

dataset_df= []
for estimator in ESTIMATOR_LIST:
    curr_dir = RESULTS_DIR + '/' + dataset_name + '/' 'seed_' + str(seed) + '/' + estimator

    sub_df = []
    for log_file in pathlib.Path(curr_dir).glob('logs.p'):
        curr_df = pickle.load(open(log_file, 'rb'))
        sub_df.append(curr_df)

    if len(sub_df):
        sub_df = pd.concat(sub_df, axis=0)

    # To ensure consistency that lower value for the metric is better
    for metric in ['cal_dr_s_score', 'cal_dr_t_score', 'cal_tmle_s_score', 'cal_tmle_t_score', 'qini_dr_s_score',
                   'qini_dr_t_score', 'qini_tmle_s_score', 'qini_tmle_t_score']:
        sub_df[metric] = -1 * sub_df[metric]

    dataset_df.append(sub_df)
dataset_df= pd.concat(dataset_df, axis= 0)

# Loading Datasets
dataset_name, dataset_obj = load_dataset_obj(dataset_name, root_dir)
dataset_samples = sample_dataset(dataset_name, dataset_obj, seed=seed, case='eval')
eval_w, eval_t, eval_y, _, true_ite = dataset_samples['w'], dataset_samples['t'], dataset_samples['y'], \
                                      dataset_samples['ate'], dataset_samples['ite']

softmax_temp_grid = np.logspace(-3.0, 5.0, num=10)

final_df = []
for score in SCORES:
    print(dataset_name, seed, score)

    # Computing the ensemble approach
    ensemble_df = []
    for meta_estimator in ESTIMATOR_LIST:
        meta_estimator_df = dataset_df[dataset_df['meta-estimator'] == meta_estimator]
        if meta_estimator in ['dr_learner', 'dml_learner', 's_learner_upd', 'x_learner']:
            # If there are multiple estimators that are minimum corresponding to the metric, then retain all of them
            if meta_estimator == 'dr_learner':
                final_model_opt_score= 'tau_dr_t_clip_score'
            elif meta_estimator == 'dml_learner':
                final_model_opt_score= 'rscore'
            elif meta_estimator == 'x_learner':
                final_model_opt_score= 'x_score'
            elif meta_estimator == 's_learner_upd':
                final_model_opt_score= 'tau_s_score'

            #TODO: Make this as input of argparse. This is for the case of using current score to choose among the final models
            # final_model_opt_score= score
            
            meta_estimator_df = meta_estimator_df[ meta_estimator_df[final_model_opt_score] == meta_estimator_df[final_model_opt_score].min() ]
        ensemble_df.append(meta_estimator_df)
    ensemble_df = pd.concat(ensemble_df)

    # #Debugging to see whether the best estimator as per current score is contained in the ensembel set of estimators
    # print('/////')
    # print(ensemble_df['pehe'])
    # print('----------')
    # print(ensemble_df[ensemble_df[score] == ensemble_df[score].min()]['pehe'].mean())

    # ITE Estimates Array: (number of meta estimators, number of data samples in particular dataset)
    ite_estimates = []
    for item in ensemble_df['ite-estimates'].to_numpy():
        ite_estimates.append(item[0])
    #     print(true_ite.shape)
    #     print(true_ite[:5])
    #     print(item[0].shape)
    #     print(item[0][:5])
    #     print('Please Work.')
    #     print(np.sqrt(np.mean((item[0] - true_ite) ** 2)))
    # sys.exit()
    ite_estimates = np.array(ite_estimates)

    ite_estimates_train = []
    for item in ensemble_df['ite-estimates-train'].to_numpy():
        ite_estimates_train.append(item[0])
    ite_estimates_train = np.array(ite_estimates_train)

    score_arr_org = ensemble_df[score].to_numpy(dtype='float32')

    for softmax_temp in softmax_temp_grid:
        score_arr = copy.deepcopy(score_arr_org)
        score_arr = -1 * softmax_temp * score_arr
        softmax_score_arr = softmax(score_arr)

        # # To debug the type of softmax temperatures and the resulting score arrays
        # print(softmax_temp)
        # print(softmax_score_arr)
        # continue

        softmax_score_arr = np.reshape(softmax_score_arr, (softmax_score_arr.shape[0], 1))

        # Replace mean with sum
        new_ite_estimates = np.sum(softmax_score_arr * ite_estimates, axis=0)
        new_ite_estimates_train= [np.sum(softmax_score_arr * ite_estimates_train, axis=0)]

        # print(new_ite_estimates.shape)
        # print(new_ite_estimates_train.shape)

        # Computing relevant evaluation metric for ensemble
        nuisance_stats_dir = RESULTS_DIR + '/' + dataset_name + '/' + 'seed_' + str(
            seed) + '/' + 'nuisance_models' + '/'
        # Nuisance Models
        prop_prob, prop_score = get_nuisance_propensity_pred(eval_w, eval_t, save_dir=nuisance_stats_dir)
        outcome_s_pred = get_nuisance_outome_s_pred(eval_w, eval_t, save_dir=nuisance_stats_dir)
        outcome_t_pred = get_nuisance_outcome_t_pred(eval_w, eval_t, save_dir=nuisance_stats_dir)
        outcome_dml_pred = get_nuisance_outcome_dml_pred(eval_w, save_dir=nuisance_stats_dir)

        inv_prop_threshold = 10
        # Compute Evaluation Metric
        if score == 'rscore':
            eval_metric_score = calculate_r_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                                 outcome_pred=outcome_dml_pred, treatment_prob=prop_prob[:, 1])
        elif score == 'value_score':
            eval_metric_score = calculate_value_risk(new_ite_estimates, eval_w, eval_t, eval_y, dataset_name,
                                                     prop_score=prop_score)
        elif score == 'value_dr_score':
            eval_metric_score = calculate_value_dr_risk(new_ite_estimates, eval_w, eval_t, eval_y, dataset_name,
                                                        outcome_pred=outcome_t_pred,
                                                        prop_score=copy.deepcopy(prop_score))
        elif score == 'value_dr_clip_prop_score':
            eval_metric_score = calculate_value_dr_risk(new_ite_estimates, eval_w, eval_t, eval_y, dataset_name,
                                                        outcome_pred=outcome_t_pred, prop_score=prop_score,
                                                        min_propensity=0.1)
        elif score == 'tau_s_score':
            eval_metric_score = calculate_all_tau_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                                       outcome_pred=outcome_s_pred,
                                                       prop_score=copy.deepcopy(prop_score), case='s_score')
        elif score == 'tau_t_score':
            eval_metric_score = calculate_all_tau_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                                       outcome_pred=outcome_t_pred,
                                                       prop_score=copy.deepcopy(prop_score), case='t_score')
        elif score == 'tau_match_score':
            eval_metric_score = calculate_tau_match_risk(new_ite_estimates, eval_w, eval_t, eval_y)
        elif score == 'tau_iptw_score':
            eval_metric_score = calculate_all_tau_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                                       prop_score=copy.deepcopy(prop_score), case='iptw_score')
        elif score == 'tau_iptw_clip_score':
            eval_metric_score = calculate_all_tau_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                                       prop_score=copy.deepcopy(prop_score), case='iptw_clip_score',
                                                       inv_prop_threshold=inv_prop_threshold)
        elif score == 'tau_switch_iptw_s_score':
            eval_metric_score = calculate_all_tau_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                                       outcome_pred=outcome_s_pred,
                                                       prop_score=copy.deepcopy(prop_score), case='switch_iptw_s_score',
                                                       inv_prop_threshold=inv_prop_threshold)
        elif score == 'tau_switch_iptw_t_score':
            eval_metric_score = calculate_all_tau_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                                       outcome_pred=outcome_t_pred,
                                                       prop_score=copy.deepcopy(prop_score), case='switch_iptw_t_score',
                                                       inv_prop_threshold=inv_prop_threshold)
        elif score == 'tau_cab_iptw_s_score':
            eval_metric_score = calculate_all_tau_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                                       outcome_pred=outcome_s_pred,
                                                       prop_score=copy.deepcopy(prop_score), case='cab_iptw_s_score',
                                                       inv_prop_threshold=inv_prop_threshold)
        elif score == 'tau_cab_iptw_t_score':
            eval_metric_score = calculate_all_tau_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                                       outcome_pred=outcome_t_pred,
                                                       prop_score=copy.deepcopy(prop_score), case='cab_iptw_t_score',
                                                       inv_prop_threshold=inv_prop_threshold)
        elif score == 'tau_dr_s_score':
            eval_metric_score = calculate_all_tau_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                                       outcome_pred=outcome_s_pred,
                                                       prop_score=copy.deepcopy(prop_score), case='dr_s_score')
        elif score == 'tau_dr_t_score':
            eval_metric_score = calculate_all_tau_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                                       outcome_pred=outcome_t_pred,
                                                       prop_score=copy.deepcopy(prop_score), case='dr_t_score')
        elif score == 'tau_dr_s_clip_score':
            eval_metric_score = calculate_all_tau_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                                       outcome_pred=outcome_s_pred,
                                                       prop_score=copy.deepcopy(prop_score), case='dr_s_clip_score',
                                                       inv_prop_threshold=inv_prop_threshold)
        elif score == 'tau_dr_t_clip_score':
            eval_metric_score = calculate_all_tau_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                                       outcome_pred=outcome_t_pred,
                                                       prop_score=copy.deepcopy(prop_score), case='dr_t_clip_score',
                                                       inv_prop_threshold=inv_prop_threshold)
        elif score == 'tau_switch_dr_s_score':
            eval_metric_score = calculate_all_tau_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                                       outcome_pred=outcome_s_pred,
                                                       prop_score=copy.deepcopy(prop_score), case='switch_dr_s_score',
                                                       inv_prop_threshold=inv_prop_threshold)
        elif score == 'tau_switch_dr_t_score':
            eval_metric_score = calculate_all_tau_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                                       outcome_pred=outcome_t_pred,
                                                       prop_score=copy.deepcopy(prop_score), case='switch_dr_t_score',
                                                       inv_prop_threshold=inv_prop_threshold)
        elif score == 'tau_cab_dr_s_score':
            eval_metric_score = calculate_all_tau_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                                       outcome_pred=outcome_s_pred,
                                                       prop_score=copy.deepcopy(prop_score), case='cab_dr_s_score',
                                                       inv_prop_threshold=inv_prop_threshold)
        elif score == 'tau_cab_dr_t_score':
            eval_metric_score = calculate_all_tau_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                                       outcome_pred=outcome_t_pred,
                                                       prop_score=copy.deepcopy(prop_score), case='cab_dr_t_score',
                                                       inv_prop_threshold=inv_prop_threshold)
        elif score == 'tau_tmle_s_score':
            eval_metric_score = calculate_all_tau_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                                       outcome_pred=outcome_s_pred, outcome_dml_pred=outcome_dml_pred,
                                                       prop_prob=copy.deepcopy(prop_prob),
                                                       prop_score=copy.deepcopy(prop_score), case='tmle_s_score',
                                                       inv_prop_threshold=inv_prop_threshold,
                                                       save_dir=nuisance_stats_dir)
        elif score == 'tau_tmle_t_score':
            eval_metric_score = calculate_all_tau_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                                       outcome_pred=outcome_t_pred, outcome_dml_pred=outcome_dml_pred,
                                                       prop_prob=copy.deepcopy(prop_prob),
                                                       prop_score=copy.deepcopy(prop_score), case='tmle_t_score',
                                                       inv_prop_threshold=inv_prop_threshold,
                                                       save_dir=nuisance_stats_dir)
        elif score == 'influence_score':
            eval_metric_score = calculate_influence_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                                         outcome_pred=outcome_t_pred, prop_prob=prop_prob,
                                                         min_propensity=0.01)
        elif score == 'influence_clip_prop_score':
            eval_metric_score = calculate_influence_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                                         outcome_pred=outcome_t_pred, prop_prob=prop_prob,
                                                         min_propensity=0.01)
        elif score == 'x_score':
            eval_metric_score = calculate_x_risk(new_ite_estimates, eval_w, eval_t, eval_y, outcome_pred=outcome_t_pred)
        elif score == 'cal_dr_s_score':
            eval_metric_score = calculate_calibration_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                           ite_estimates_train=new_ite_estimates_train,
                                           outcome_pred=outcome_s_pred,
                                           prop_score=copy.deepcopy(prop_score), case='dr_s_score')
        elif score == 'cal_dr_t_score':
            eval_metric_score = calculate_calibration_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                           ite_estimates_train=new_ite_estimates_train,
                                           outcome_pred=outcome_t_pred,
                                           prop_score=copy.deepcopy(prop_score), case='dr_t_score')
        elif score == 'cal_tmle_s_score':
            eval_metric_score = calculate_calibration_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                               ite_estimates_train=new_ite_estimates_train,
                                               outcome_pred=outcome_s_pred, outcome_dml_pred=outcome_dml_pred,
                                               prop_prob=copy.deepcopy(prop_prob),
                                               prop_score=copy.deepcopy(prop_score), case='tmle_s_score',
                                               save_dir=nuisance_stats_dir)
        elif score == 'cal_tmle_t_score':
            eval_metric_score = calculate_calibration_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                               ite_estimates_train=new_ite_estimates_train,
                                               outcome_pred=outcome_t_pred, outcome_dml_pred=outcome_dml_pred,
                                               prop_prob=copy.deepcopy(prop_prob),
                                               prop_score=copy.deepcopy(prop_score), case='tmle_t_score',
                                               save_dir=nuisance_stats_dir)
        elif score == 'qini_dr_s_score':
            eval_metric_score = calculate_qini_risk(new_ite_estimates, eval_w, eval_t, eval_y, outcome_pred=outcome_s_pred,
                                        prop_score=copy.deepcopy(prop_score), case='dr_s_score')
        elif score == 'qini_dr_t_score':
            eval_metric_score = calculate_qini_risk(new_ite_estimates, eval_w, eval_t, eval_y, outcome_pred=outcome_t_pred,
                                    prop_score=copy.deepcopy(prop_score), case='dr_t_score')
        elif score == 'qini_tmle_s_score':
            eval_metric_score = calculate_qini_risk(new_ite_estimates, eval_w, eval_t, eval_y, outcome_pred=outcome_s_pred,
                                        outcome_dml_pred=outcome_dml_pred, prop_prob=copy.deepcopy(prop_prob),
                                        prop_score=copy.deepcopy(prop_score), case='tmle_s_score',
                                        save_dir=nuisance_stats_dir)
        elif score == 'qini_tmle_t_score':
            eval_metric_score = calculate_qini_risk(new_ite_estimates, eval_w, eval_t, eval_y, outcome_pred=outcome_t_pred,
                                        outcome_dml_pred=outcome_dml_pred, prop_prob=copy.deepcopy(prop_prob),
                                        prop_score=copy.deepcopy(prop_score), case='tmle_t_score',
                                        save_dir=nuisance_stats_dir)
        else:
            continue

        # Computing PEHE
        pehe = np.sqrt(np.mean((new_ite_estimates - true_ite) ** 2))

        # #Debugging to see whether we are able to get correct PEHE estimates from the old ite estimates
        # print('Computed PEHE')
        # print(pehe)

        metrics = {}
        metrics.update({'dataset': dataset_name})
        metrics.update({'seed': seed})
        metrics.update({'ensemble_type': score})
        metrics.update({'softmax_temp': softmax_temp})
        metrics.update({'ensemble_score': eval_metric_score[score]})
        metrics.update({'pehe': pehe})
        metrics.update({'oracle_pehe': dataset_df['pehe'].min()})

        final_df.append(pd.DataFrame([metrics]))

final_df= pd.concat(final_df, axis=0)
save_location=  RESULTS_DIR + '/' + dataset_name + '/' 'seed_' + str(seed) + '/' + 'ensemble_upd.csv'
final_df.to_csv(save_location, float_format='%.2f', index=False)
