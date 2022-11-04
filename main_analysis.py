import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
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

#pd.set_option('max_columns', None)
# pd.set_option('expand_frame_repr', False)

root_dir = os.path.expanduser('~') + '/scratch/causal_val_datasets/'
RESULTS_DIR = root_dir + str(Path('results_final'))

REAL_DATASET_NAMES = [
                'twins',
                'lalonde_psid1',
                'lalonde_cps1'
                ]

#Heterogenous ACIC 2016
SYN_DATASET_NAMES = pickle.load(open('datasets/acic_2016_heterogenous_list.p', 'rb'))
# #Heterogenous ACIC 2018
SYN_DATASET_NAMES += pickle.load(open('datasets/acic_2018_heterogenous_list.p', 'rb'))

# SYN_DATASET_NAMES= []
DATASET_NAMES = REAL_DATASET_NAMES + SYN_DATASET_NAMES

TOTAL_SEEDS= 3
ESTIMATOR_LIST= ['dr_learner', 'dml_learner', 'causal_forest_learner', 's_learner', 's_learner_upd', 't_learner', 'x_learner']
SCORES = ['value_score', 'value_dr_score', 'value_dr_clip_prop_score', 'influence_score', 'influence_clip_prop_score', 'tau_t_score', 'tau_s_score', 'tau_match_score', 'tau_iptw_score', 'tau_iptw_clip_prop_score', 'tau_dr_score', 'tau_dr_clip_prop_score', 'rscore']
CVX_TRAIN_SCORES= ['value_score', 'value_dr_score', 'value_dr_clip_prop_score', 'tau_t_score', 'tau_s_score', 'tau_match_score', 'tau_iptw_score', 'tau_iptw_clip_prop_score', 'tau_dr_score', 'tau_dr_clip_prop_score', 'rscore']

MAX_Y_SCORE = 'model_y_score'
MAX_T_SCORE = 'model_t_score'


def dataset_to_group(dataset):
    """Map from dataset name to dataset group (that are conditioned on)"""
    if dataset.startswith('acic_2016'):
        return 'acic_2016'
    # elif dataset.startswith('acic_2018'):
    #     return 'acic_2018'
    elif dataset.startswith('lbidd'):
        return 'lbidd'
    else:
        return dataset

def converter(instr):
    return np.fromstring(instr[0],sep=' ')


def delete_datasets(base_dir= RESULTS_DIR, dataset_names=DATASET_NAMES, estimator_list=ESTIMATOR_LIST):

    final_df=[]
    for dataset_name in dataset_names:
        for seed in range(TOTAL_SEEDS):
            for estimator in estimator_list:
                sub_df= []
                curr_dir= base_dir + '/' + dataset_name + '/' 'seed_' + str(seed) + '/' + estimator

                for log_file in pathlib.Path(curr_dir).glob('logs.p'):
                    print(log_file)
                    os.remove(log_file)

    return

def save_meta_dataset(base_dir= RESULTS_DIR, dataset_names=DATASET_NAMES, estimator_list=ESTIMATOR_LIST):

    final_df=[]
    for dataset_name in dataset_names:
        for seed in range(TOTAL_SEEDS):
            for estimator in estimator_list:
                sub_df= []

                curr_dir= base_dir + '/' + dataset_name + '/' 'seed_' + str(seed) + '/' + estimator
                for log_file in pathlib.Path(curr_dir).glob('logs.p'):
                    curr_df= pickle.load(open(log_file, 'rb'))
                    sub_df.append(curr_df)

                # Removing data points with negative R2 score on the training dataset
                # sub_df= sub_df[ sub_df['model_train_y_score'] >= 0 ]

                if len(sub_df):
                    sub_df= pd.concat(sub_df, axis=0)
                    final_df.append(sub_df)
                    print(curr_dir, sub_df.shape)
                else:
                    continue

                # sub_df_prop= []
                # curr_dir= base_dir + '/' + dataset_name + '/' 'seed_' + str(seed) + '/' + estimator
                # for log_file in pathlib.Path(curr_dir).glob('logs-prop.p'):
                #     curr_df= pickle.load(open(log_file, 'rb'))
                #     sub_df_prop.append(curr_df)
                #
                # if len(sub_df_prop):
                #     sub_df_prop= pd.concat(sub_df_prop, axis=0)
                #     print(curr_dir, sub_df_prop.shape)
                # else:
                #     continue
                #
                # sub_df['tau_dr_clip_prop_score']= sub_df_prop['tau_dr_clip_prop_score']
                # sub_df['tau_iptw_clip_prop_score']= sub_df_prop['tau_iptw_clip_prop_score']
                # sub_df['value_dr_clip_prop_score']= sub_df_prop['value_dr_clip_prop_score']
                # sub_df['influence_clip_prop_score']= sub_df_prop['influence_clip_prop_score']

                if len(sub_df):
                    final_df.append(sub_df)

    final_df= pd.concat(final_df, axis=0)
    save_location= 'results/logs'

    # #Saving the whole dataframe with ITE estiamtes
    # pickle.dump(final_df, open(save_location + '.p', 'wb'))

    #Drop the ite-estimates vector columns
    final_df= final_df.drop(columns=['ite-estimates'])
    final_df.to_csv(save_location + '.csv', float_format='%.2f', index=False)

    return

def save_ensemble_approach(base_dir= RESULTS_DIR, dataset_names=DATASET_NAMES, estimator_list=ESTIMATOR_LIST, scores_list= SCORES):

    #Load the dataframe
    save_location = 'results/logs'
    df= pickle.load(open(save_location + '.p', 'rb'))

    final_df=[]
    for dataset_name in dataset_names:
        print(dataset_name)
        for seed in range(TOTAL_SEEDS):

            # dataset_df= []
            # for estimator in estimator_list:
            #     curr_dir= base_dir + '/' + dataset_name + '/' 'seed_' + str(seed) + '/' + estimator
            #
            #     sub_df= []
            #     # for log_file in pathlib.Path(curr_dir).glob('*.p'):
            #     for log_file in pathlib.Path(curr_dir).glob('logs-new.p'):
            #         curr_df= pickle.load(open(log_file, 'rb'))
            #         sub_df.append(curr_df)
            #
            #     if len(sub_df):
            #         sub_df= pd.concat(sub_df, axis=0)
            #         dataset_df.append(sub_df)
            #         print(curr_dir, sub_df.shape)
            #     else:
            #         continue
            #
            #     # #Removing data points with negative R2 score on the training dataset
            #     # sub_df= sub_df[ sub_df['model_train_y_score'] >= 0 ]
            #
            #
            # #Concatenating dataframes across estimators
            # dataset_df= pd.concat(dataset_df, axis=0)

            # Sub sampling from dataframe corresponding to current seed and dataset
            dataset_df= df[df['dataset'] == dataset_name]
            dataset_df= dataset_df[dataset_df['seed'] == seed]

            #Loading Datasets
            dataset_name, dataset_obj = load_dataset_obj(dataset_name, root_dir)
            dataset_samples = sample_dataset(dataset_name, dataset_obj, seed=seed, case='eval')
            eval_w, eval_t, eval_y, _, true_ite = dataset_samples['w'], dataset_samples['t'], dataset_samples['y'], \
                                               dataset_samples['ate'], dataset_samples['ite']

            #Computing the ensemble approach
            ite_estimates= []
            for item in dataset_df['ite-estimates'].to_numpy():
                ite_estimates.append(item[0].tolist())
            ite_estimates= np.array(ite_estimates)

            for score in scores_list + ['pehe']:
                softmax_temp_grid= np.logspace(-10.0, 10.0, num=25)
                for softmax_temp in softmax_temp_grid:

                    score_arr= dataset_df[score].to_numpy(dtype='float32')
                    score_arr= -1*softmax_temp*score_arr
                    softmax_score_arr= softmax(score_arr)
                    softmax_score_arr = np.reshape(softmax_score_arr, (softmax_score_arr.shape[0], 1))

                    #Replace mean with sum
                    new_ite_estimates= np.sum(softmax_score_arr * ite_estimates, axis=0)
                    # print(new_ite_estimates.shape, true_ite.shape)

                    #Computing relevant evaluation metric for ensemble
                    nuisance_stats_dir= RESULTS_DIR + '/' + dataset_name + '/' + 'seed_' + str(seed) + '/' + 'metric' + '/'
                    # Nuisance Models
                    prop_prob, prop_score = get_nuisance_propensity_pred(eval_w, eval_t, save_dir=nuisance_stats_dir)
                    outcome_s_pred = get_nuisance_outome_s_pred(eval_w, eval_t, save_dir=nuisance_stats_dir)
                    outcome_t_pred = get_nuisance_outcome_t_pred(eval_w, eval_t, save_dir=nuisance_stats_dir)
                    outcome_r_pred = get_nuisance_outcome_r_pred(eval_w, save_dir=nuisance_stats_dir)

                    #Compute Evaluation Metric
                    if score == 'rscore':
                        eval_metric_score = calculate_r_risk(new_ite_estimates, eval_w, eval_t, eval_y, outcome_pred= outcome_r_pred, treatment_prob=prop_prob[:, 1])
                        eval_metric_score = eval_metric_score['rscore']
                    elif score == 'value_score':
                        eval_metric_score = calculate_value_risk(new_ite_estimates, eval_w, eval_t, eval_y, dataset_name, prop_score= prop_score)
                        eval_metric_score = eval_metric_score['value_score']
                    elif score == 'value_dr_score':
                        eval_metric_score = calculate_value_dr_risk(new_ite_estimates, eval_w, eval_t, eval_y, dataset_name, outcome_pred= outcome_t_pred, prop_score= prop_score, min_propensity= 0.1)
                        eval_metric_score = eval_metric_score['value_dr_score']
                    elif score == 'value_dr_clip_prop_score':
                        eval_metric_score = calculate_value_dr_risk(new_ite_estimates, eval_w, eval_t, eval_y, dataset_name, outcome_pred= outcome_t_pred, prop_score= prop_score, min_propensity= 0.1)
                        eval_metric_score = eval_metric_score['value_dr_clip_prop_score']
                    elif score == 'tau_s_score':
                        eval_metric_score = calculate_tau_s_risk(new_ite_estimates, eval_w, eval_t, eval_y, outcome_pred= outcome_s_pred)
                        eval_metric_score = eval_metric_score['tau_s_score']
                    elif score == 'tau_t_score':
                        eval_metric_score = calculate_tau_t_risk(new_ite_estimates, eval_w, eval_t, eval_y,
                                                                 outcome_pred=outcome_t_pred)
                        eval_metric_score = eval_metric_score['tau_t_score']
                    elif score == 'tau_match_score':
                        eval_metric_score = calculate_tau_risk(new_ite_estimates, eval_w, eval_t, eval_y)
                        eval_metric_score = eval_metric_score['tau_match_score']
                    elif score == 'tau_iptw_score':
                        eval_metric_score = calculate_tau_iptw_risk(new_ite_estimates,eval_w, eval_t, eval_y, prop_score= prop_score, min_propensity= 0.1)
                        eval_metric_score = eval_metric_score['tau_iptw_score']
                    elif score == 'tau_iptw_clip_prop_score':
                        eval_metric_score = calculate_tau_iptw_risk(new_ite_estimates,eval_w, eval_t, eval_y, prop_score= prop_score, min_propensity= 0.1)
                        eval_metric_score = eval_metric_score['tau_iptw_clip_prop_score']
                    elif score == 'tau_dr_score':
                        eval_metric_score = calculate_tau_dr_risk(new_ite_estimates, eval_w, eval_t, eval_y, outcome_pred= outcome_t_pred, prop_score= prop_score, min_propensity= 0.1)
                        eval_metric_score = eval_metric_score['tau_dr_score']
                    elif score == 'tau_dr_clip_prop_score':
                        eval_metric_score = calculate_tau_dr_risk(new_ite_estimates, eval_w, eval_t, eval_y, outcome_pred= outcome_t_pred, prop_score= prop_score, min_propensity= 0.1)
                        eval_metric_score = eval_metric_score['tau_dr_clip_prop_score']
                    elif score == 'influence_score':
                        eval_metric_score = calculate_influence_risk(new_ite_estimates, eval_w, eval_t, eval_y, outcome_pred= outcome_t_pred, prop_prob= prop_prob, min_propensity= 0.1)
                        eval_metric_score = eval_metric_score['influence_score']
                    elif score == 'influence_clip_prop_score':
                        eval_metric_score = calculate_influence_risk(new_ite_estimates, eval_w, eval_t, eval_y, outcome_pred= outcome_t_pred, prop_prob= prop_prob, min_propensity= 0.1)
                        eval_metric_score = eval_metric_score['influence_clip_prop_score']

                    #Computing PEHE
                    pehe= np.sqrt(np.mean((new_ite_estimates - true_ite)**2))

                    metrics={}
                    metrics.update({'dataset': dataset_name})
                    metrics.update({'seed': seed})
                    metrics.update({'ensemble_type': score})
                    metrics.update({'softmax_temp': softmax_temp})
                    metrics.update({'ensemble_score': eval_metric_score})
                    metrics.update({'pehe': pehe})

                    # print(score, eval_metric_score, pehe)
                    # print(new_ite_estimates)

                    final_df.append(pd.DataFrame([metrics]))

    final_df= pd.concat(final_df, axis=0)
    save_location= 'results/logs-ensemble-check.csv'
    final_df.to_csv(save_location, float_format='%.2f', index=False)

    return


def compute_rank_correlation(conditioning_cols=['dataset_group', 'dataset', 'seed'], scores=SCORES, save_location= 'results/'):

    if not os.path.exists(save_location+'logs.csv'):
        save_meta_dataset(dataset_names=DATASET_NAMES)

    #Read the dataframe file
    df= pd.read_csv(save_location+'logs.csv')

    # Insert dataset group column
    df.insert(0, 'dataset_group', df['dataset'].apply(dataset_to_group))

    # Conditioning on dataset group, instances, and seeds
    grouped_df = df.groupby(conditioning_cols)

    # Computing spearman rank correlation
    cols = {score: grouped_df.apply(
        lambda x: spearmanr(x['pehe'].to_numpy(dtype='float32'), x[score].to_numpy(dtype='float32'))[0]) for score
        in scores}
    correlation_df = pd.DataFrame(cols)

    # Aggregation along dataset instances and seed columns
    agg_correlation_df = correlation_df.T.agg("mean", axis="columns", level=0)
    print(agg_correlation_df.round(2).to_latex())

    return

def argopt_softmax_temp(x, score):

    indices= x['ensemble_type'] == score
    argopt_ready_df= x[indices]
    # print(argopt_ready_df)

    convert_dict = {'ensemble_score': float,
                    }
    argopt_ready_df= argopt_ready_df.astype(convert_dict)
    argopt_indices= argopt_ready_df['ensemble_score'].argmin()
    # print(argopt_indices)
    pehe= argopt_ready_df.iloc[argopt_indices, -1].mean()

    return pehe

def compute_ensemble_pehe(conditioning_cols=['dataset_group', 'dataset', 'seed'], save_location='results/'):

    #Read the dataframe file
    df= pd.read_csv(save_location+'logs-ensemble-old.csv')

    # Insert dataset group column
    df.insert(0, 'dataset_group', df['dataset'].apply(dataset_to_group))

    # Argmin over the softmax temperature column to get the best performing ensemble based on each score
    grouped_df = df.groupby(conditioning_cols)
    scores= df['ensemble_type'].unique()
    cols = {score: grouped_df.apply(lambda x: argopt_softmax_temp(x, score)) for score in scores}
    ensemble_df = pd.DataFrame(cols)

    #Aggregation over the specificed columns
    agg_ensemble_df = ensemble_df.T.agg("mean", axis="columns", level=0)
    # print(agg_ensemble_df)
    print(agg_ensemble_df.round(2).to_latex())

    return

# Only needed to save a single csv file for all the datasets
# save_meta_dataset(dataset_names= DATASET_NAMES)
# delete_datasets(dataset_names= DATASET_NAMES)
# save_ensemble_approach(dataset_names= DATASET_NAMES)
# compute_ensemble_pehe()
compute_rank_correlation()
