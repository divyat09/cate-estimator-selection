import warnings
warnings.filterwarnings('ignore')

import random
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import argparse
import sys
import pickle
import pathlib
import os

from scipy.special import softmax

from data.samplers import load_dataset_obj, sample_dataset
from utils.consts import SCORES, ESTIMATOR_LIST, SCORES_SIGN_FLIP, \
                         SOFTMAX_TEMP_GRID, FINAL_MODEL_OPT_SCORE
from causal_estimators.metrics import get_nuisance_propensity_pred, get_nuisance_outome_s_pred,\
                             get_nuisance_outcome_t_pred, get_nuisance_outcome_dml_pred,\
                             calculate_metrics

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='twins',
                    help='Datasets: lalonde_psid1; lalonde_cps1; twins; lbidd')
parser.add_argument('--seed', type=int, default=0,
                    help='Seed for causal effect estimation experiments')
parser.add_argument('--root_dir', type=str, default='/scratch/cate_eval_analysis/')
parser.add_argument('--res_dir', type=str, default='results_final')
parser.add_argument('--slurm_exp', type=int, default=0,
                    help='0: None; 1: Parallelize across datasets')

args = parser.parse_args()

# Experiments on Slurm
if args.slurm_exp:
    dataset_list = ['twins', 'lalonde_psid1', 'lalonde_cps1']
    # dataset_list = pickle.load(open('datasets/acic_2016_heterogenous_list.p', "rb"))

    slurm_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    args.dataset = dataset_list[slurm_idx]

dataset_name = args.dataset
seed= args.seed

root_dir = os.path.expanduser('~') + args.root_dir
RESULTS_DIR = root_dir + str(Path(args.res_dir))
nuisance_stats_dir = RESULTS_DIR + '/' + dataset_name + '/' + 'seed_' + str(seed) + '/' + 'nuisance_models' + '/'

def load_cate_df(dataset_name: str, seed: int) -> pd.DataFrame:
    """
        Load the DataFrame containing metrics evaluated for the CATE estimators for the particular dataset and seed
    """

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
        for metric in SCORES_SIGN_FLIP:
            sub_df[metric] = -1 * sub_df[metric]

        dataset_df.append(sub_df)
    dataset_df= pd.concat(dataset_df, axis= 0)
    return dataset_df


#Fix random seed
print('SEED: ', seed)
random.seed(seed)
np.random.seed(seed)

# Loading Datasets
dataset_obj = load_dataset_obj(dataset= dataset_name, root_dir= root_dir, seed= seed)
dataset_samples = sample_dataset(dataset_obj, case='eval')
eval_w, eval_t, eval_y, _, true_ite = dataset_samples['w'], dataset_samples['t'], dataset_samples['y'], \
                                      dataset_samples['ate'], dataset_samples['ite']

#Loading CATE Estimators Statistics Dataframe
dataset_df= load_cate_df(dataset_name= dataset_name, seed= seed)

final_df = []
for score in SCORES:
    print(dataset_name, seed, score)

    #Selecting the CATE Estimators to be considered for the ensemble by optimizing over final models of Meta Learners
    ensemble_df = []
    for meta_estimator in ESTIMATOR_LIST:
        estimator_arr= dataset_df['meta-estimator']
        meta_estimator_df = dataset_df[estimator_arr == meta_estimator]
        if meta_estimator in ['dr_learner', 'dml_learner', 's_learner_upd', 'x_learner']:
            final_model_opt_score= FINAL_MODEL_OPT_SCORE[meta_estimator]
            score_arr= meta_estimator_df[final_model_opt_score]            
            # If there are multiple estimators that are minimum corresponding to the metric, then retain all of them                        
            meta_estimator_df = meta_estimator_df[ score_arr == score_arr.min() ]
        ensemble_df.append(meta_estimator_df)
    ensemble_df = pd.concat(ensemble_df)

    # ITE Estimates Array: (number of meta estimators, number of data samples in particular dataset)
    ite_estimates = []
    for item in ensemble_df['ite-estimates'].to_numpy():
        ite_estimates.append(item)
    ite_estimates = np.array(ite_estimates)

    # ITE Estimates Train Array: (number of meta estimators, number of data samples in particular dataset)
    ite_estimates_train = []
    for item in ensemble_df['ite-estimates-train'].to_numpy():
        ite_estimates_train.append(item)
    ite_estimates_train = np.array(ite_estimates_train)

    #Obtain the original score array for CATE estimators in the ensemble; (number of meta estimators)
    score_arr_org = ensemble_df[score].to_numpy(dtype='float32')

    for softmax_temp in SOFTMAX_TEMP_GRID:
        score_arr = copy.deepcopy(score_arr_org)

        #Multiplying the score array with -1 as originally a lower value of score was desirable; but with softmax we need the opposite
        score_arr = -1 * softmax_temp * score_arr
        softmax_score_arr = softmax(score_arr)

        # Reshape the softmax_score_arr to (number of meta estimators, 1) to perform element-wise multiplication with ite_estimates
        softmax_score_arr = np.reshape(softmax_score_arr, (softmax_score_arr.shape[0], 1))

        #Get ITE estimates of the ensemble by taking the softmax average of the ITE estimates of the meta-learners
        new_ite_estimates = np.sum(softmax_score_arr * ite_estimates, axis=0)
        new_ite_estimates_train= np.sum(softmax_score_arr * ite_estimates_train, axis=0)

        #Obtain nuisance models for computing metrics
        prop_prob, prop_score= get_nuisance_propensity_pred(eval_w, eval_t, save_dir=nuisance_stats_dir)
        outcome_s_pred= get_nuisance_outome_s_pred(eval_w, eval_t, save_dir=nuisance_stats_dir)
        outcome_t_pred= get_nuisance_outcome_t_pred(eval_w, eval_t, save_dir=nuisance_stats_dir)
        outcome_dml_pred= get_nuisance_outcome_dml_pred(eval_w, save_dir=nuisance_stats_dir)

        # Compute Evaluation Metric
        eval_metrics = calculate_metrics(
                                            eval_w= eval_w,
                                            eval_t= eval_t,
                                            eval_y= eval_y,
                                            prop_prob= prop_prob,
                                            prop_score= copy.deepcopy(prop_score),
                                            outcome_s_pred= outcome_s_pred,
                                            outcome_t_pred= outcome_t_pred,
                                            outcome_dml_pred= outcome_dml_pred,
                                            ite_estimates= new_ite_estimates,
                                            ite_estimates_train= new_ite_estimates_train,
                                            score= score,
                                            dataset_name= dataset_name,
                                            nuisance_stats_dir= nuisance_stats_dir                                    
                                            )     

        # Computing PEHE
        pehe_squared= np.mean( (new_ite_estimates - true_ite)**2 )
        pehe= np.sqrt(pehe_squared)

        metrics = {}
        metrics.update({'dataset': dataset_name})
        metrics.update({'seed': seed})
        metrics.update({'ensemble_type': score})
        metrics.update({'softmax_temp': softmax_temp})
        metrics.update({'ensemble_score': eval_metrics[score]})
        metrics.update({'pehe': pehe})
        metrics.update({'oracle_pehe': dataset_df['pehe'].min()})

        final_df.append(pd.DataFrame([metrics]))

final_df= pd.concat(final_df, axis=0)
save_location=  RESULTS_DIR + '/' + dataset_name + '/' 'seed_' + str(seed) + '/' + 'ensemble.p'
pickle.dump(final_df, open(save_location, 'wb'))