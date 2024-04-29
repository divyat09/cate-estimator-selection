import sys
import os
import copy
import random
from pathlib import Path
import argparse
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import UndefinedMetricWarning

from data.samplers import load_dataset_obj, sample_dataset
from utils.consts import SCORES

from causal_estimators.nusiance_models import get_cate_nuisance_models, get_cate_final_models
from causal_estimators.econml_estimators import EconMLEstimator
from causal_estimators.metrics import get_nuisance_propensity_pred, get_nuisance_outome_s_pred,\
                             get_nuisance_outcome_t_pred, get_nuisance_outcome_dml_pred,\
                             calculate_metrics


warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)
# warnings.filterwarnings("ignore", message="UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.")


def print(*objects, sep=' ', end='\n', file=sys.stdout, flush=True):
    __builtins__.print(*objects, sep=sep, end=end, file=file, flush=flush)
        
#Debugging EconML Estimators
def run_experiments_for_estimator(
                                  dataset_name: str, 
                                  dataset_obj: dict,
                                  estimator,
                                  nuisance_stats_dir
                                  ):
    """ Train the CATE estimator and compute the several metrics on validation dataset.

    Inputs:
        dataset_name: Name of the dataset
        dataset_obj: Dictionary containing the dataset samples
        estimator: CATE estimator object
        nuisance_stats_dir: Directory containing the nuisance models based on AutoML selection
    """

    metrics= {}

    #Training Data
    dataset_samples = sample_dataset(dataset_obj, case='train')
    train_w, train_t, train_y = dataset_samples['w'], dataset_samples['t'], dataset_samples['y']

    #Train the CATE Estimator    
    estimator.fit(train_w, train_t, np.reshape(train_y, (train_y.shape[0])))

    #Evaluation Data
    dataset_samples= sample_dataset(dataset_obj, case='eval')
    eval_w, eval_t, eval_y, _, ite= dataset_samples['w'], dataset_samples['t'], dataset_samples['y'], dataset_samples['ate'], dataset_samples['ite']

    #Estimator's ITE vector on the training set
    train_t0 = train_t * 0
    train_t1 = train_t * 0 + 1
    ite_estimates_train= estimator.effect(train_w, train_t0, train_t1) 
    metrics.update({'ite-estimates-train': ite_estimates_train})

    #Estimator's ITE vector on the evaluation set
    eval_t0 = eval_t * 0
    eval_t1 = eval_t * 0 + 1
    ite_estimates = estimator.effect(eval_w, eval_t0, eval_t1)
    metrics.update({'ite-estimates': ite_estimates})

    #ITE Metrics on the evaluation set
    pehe_squared= np.mean( (ite_estimates - ite)**2 )
    pehe= np.sqrt(pehe_squared)
    true_ite_var= [np.var(ite)]        
    metrics.update({'pehe': pehe, 'pehe_squared': pehe_squared, 'true_ite_var': true_ite_var})

    #Obtain nuisance models for computing metrics
    prop_prob, prop_score= get_nuisance_propensity_pred(eval_w, eval_t, save_dir=nuisance_stats_dir)
    outcome_s_pred= get_nuisance_outome_s_pred(eval_w, eval_t, save_dir=nuisance_stats_dir)
    outcome_t_pred= get_nuisance_outcome_t_pred(eval_w, eval_t, save_dir=nuisance_stats_dir)
    outcome_dml_pred= get_nuisance_outcome_dml_pred(eval_w, save_dir=nuisance_stats_dir)

    #Compute the Evaluation metrics
    for score in SCORES:
        eval_metrics = calculate_metrics(
                                    eval_w= eval_w,
                                    eval_t= eval_t,
                                    eval_y= eval_y,
                                    prop_prob= prop_prob,
                                    prop_score= copy.deepcopy(prop_score),
                                    outcome_s_pred= outcome_s_pred,
                                    outcome_t_pred= outcome_t_pred,
                                    outcome_dml_pred= outcome_dml_pred,
                                    ite_estimates= ite_estimates,
                                    ite_estimates_train= ite_estimates_train,
                                    score= score,
                                    dataset_name= dataset_name,
                                    nuisance_stats_dir= nuisance_stats_dir                                    
                                    )
        metrics.update(eval_metrics)

    return metrics

if __name__ == "__main__":

    # Input Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='twins',
                        help='Datasets: lalonde_psid1; lalonde_cps1; twins; lbidd')
    parser.add_argument('--estimator', type=str, default="s_learner",
                       help='List of different estimators for causal inference')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for causal effect estimation experiments')
    parser.add_argument('--grid_size', type=int, default=10,
                       help='Grid Size for the final estimators nuisance models')
    parser.add_argument('--slurm_exp', type=int, default=0,
                       help='0: None; 1: Parallelize across datasets' )
    parser.add_argument('--root_dir', type=str, default='/scratch/cate_eval_analysis/')
    parser.add_argument('--res_dir', type=str, default='results_final')

    args = parser.parse_args()

    #Experiments on Slurm
    if args.slurm_exp:
        #dataset_list = ['twins', 'lalonde_psid1', 'lalonde_cps1']
        dataset_list = pickle.load(open('datasets/acic_2016_heterogenous_list.p', "rb"))

        slurm_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
        args.dataset = dataset_list[slurm_idx]

    dataset_name= args.dataset
    estimator_name= args.estimator
    seed = args.seed
    grid_size= args.grid_size
    root_dir= os.path.expanduser('~') + args.root_dir
    res_dir= args.res_dir

    #Create Logs Directory
    RESULTS_DIR = root_dir + str(Path(res_dir))
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    nuisance_stats_dir= RESULTS_DIR + '/' + dataset_name + '/' + 'seed_' + str(seed) + '/' + 'nuisance_models' + '/'
    save_dir= RESULTS_DIR + '/' + dataset_name + '/' + 'seed_' + str(seed) + '/' +  estimator_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_location = save_dir + 'logs.p'

    #Fix random seed
    print('SEED: ', seed)
    random.seed(seed)
    np.random.seed(seed)

    #Load dataset with true ITE, ATE from the generative model
    dataset_obj = load_dataset_obj(dataset= dataset_name, root_dir= root_dir, seed= seed)

    #Nuisance models based on the AutoML selection 
    nuisance_model= get_cate_nuisance_models(estimator_name, nuisance_stats_dir)

    #Final model for Meta Learners based on the randomly sampled list of outcome models
    final_models= get_cate_final_models(estimator_name, grid_size)

    df_list=[]
    for final_model in final_models:

        print('Dataset: ', dataset_name)
        print('Seed: ', seed)
        print('CATE Estimator: ', estimator_name)
        print('ESTIMATOR Hyper param: ', final_model)

        #CATE estimator object to support common interface to various EconML meta learners
        estimator = EconMLEstimator(estimator_name, nuisance_model, final_model)

        #Train the CATE estimator and compute the several metrics on validation dataset
        metrics= run_experiments_for_estimator(
            dataset_name= dataset_name, 
            dataset_obj= dataset_obj,
            estimator= estimator,
            nuisance_stats_dir= nuisance_stats_dir
        )

        #Store the results in a dataframe
        df = pd.DataFrame([metrics])
        df.insert(0, 'dataset', dataset_name)
        df.insert(1, 'meta-estimator', estimator_name)
        df.insert(2, 'seed', seed)
        df.insert(3, 'outcome-model', nuisance_model['model_y']['name'])
        df.insert(4, 'outcome-model-hparam', nuisance_model['model_y']['hparam'])
        df.insert(5, 'prop-model', nuisance_model['model_t']['name'])
        df.insert(6, 'prop-model-hparam', nuisance_model['model_t']['hparam'])
        df.insert(7, 'final-model', final_model['name'])
        df.insert(8, 'final-model-hparan', final_model['hparam'])
        df_list.append(df)

    res_df = pd.concat(df_list, axis=0)
    pickle.dump(res_df, open(save_location, 'wb'))
