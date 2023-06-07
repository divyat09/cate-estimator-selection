import numpy as np
import pandas as pd
import time
import sys
import os
from pathlib import Path
import argparse
import pickle

from sklearn.base import clone

from causal_estimators.econml_estimators import EconMLEstimator

from utils.evaluation import calculate_metrics
from utils.helpers import *

import warnings

warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)
# warnings.filterwarnings("ignore", message="UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples. Use `zero_division` parameter to control this behavior.")


def print(*objects, sep=' ', end='\n', file=sys.stdout, flush=True):
    __builtins__.print(*objects, sep=sep, end=end, file=file, flush=flush)
        
#Debugging EconML Estimators
def run_experiments_for_estimator(
                                  dataset_name, dataset_obj,
                                  estimator_name, estimator,
                                  model_f, seed,
                                  train_estimator_flag,
                                  save_dir,
                                  nuisance_stats_dir
                                  ):

    print('Dataset: ', dataset_name)
    print('Seed: ', seed)
    print('ESTIMATOR Hyper param: ', model_f)
    # print('Nuisance Model: ', nuisance_model)c
    print('Estimator Constructed: ', estimator_name)

    model_t, model_y= get_nuisance_models_names(nuisance_model)
    nuisance_model_config= model_t['name'] + '_' + model_t['hparam'] + '_' + model_y['name'] + '_' + model_y['hparam'] + '_' + model_f['name'] + '_' + model_f['hparam']

    # Training Estimators
    dataset_samples = sample_dataset(dataset_name, dataset_obj, seed=seed, case='train')
    train_w, train_t, train_y = dataset_samples['w'], dataset_samples['t'], dataset_samples['y']

    # print('Treatment Details')
    # _, treatment_counts = np.unique(train_t, return_counts=True)
    # print(treatment_counts / np.sum(treatment_counts))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if train_estimator_flag and not os.path.isfile(save_dir + nuisance_model_config + '.p'):
        print('Training Meta Estimator')
        estimator.fit(train_w, train_t, np.reshape(train_y, (train_y.shape[0])))
        # pickle.dump( estimator, open(save_dir + nuisance_model_config + '.p', "wb") )
    else:
        print('Loading Pre Trained Meta Estimator')
        estimator = pickle.load(open(save_dir + nuisance_model_config + '.p', "rb"))

    metrics = calculate_metrics(
                                dataset_name, dataset_obj,
                                estimator_name, estimator,
                                nuisance_model_config,
                                seed=seed, conf_ints=False,
                                nuisance_stats_dir= nuisance_stats_dir,
                                debug_save_dir= save_dir,
                                )

    model_df = pd.DataFrame([metrics])
    model_df.insert(0, 'dataset', dataset_name)
    model_df.insert(1, 'meta-estimator', estimator_name)
    model_df.insert(2, 'seed', seed)
    model_df.insert(3, 'outcome-model', model_y['name'])
    model_df.insert(4, 'outcome-model-hparam', model_y['hparam'])
    model_df.insert(5, 'prop-model', model_t['name'])
    model_df.insert(6, 'prop-model-hparam', model_t['hparam'])
    model_df.insert(7, 'final-model', model_f['name'])
    model_df.insert(8, 'final-model-hparan', model_f['hparam'])

    return model_df

if __name__ == "__main__":

    # Input Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='twins',
                        help='Datasets: lalonde_psid1; lalonde_cps1; twins; lbidd')
    parser.add_argument('--estimator', type=str, default="dml_learner",
                       help='List of different estimators for causal inference')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for causal effect estimation experiments')
    parser.add_argument('--grid_size', type=int, default=20,
                       help='Grid Size for the nuisance models')
    parser.add_argument('--est_grid_size', type=int, default=10,
                       help='Grid Size for the final estimators nuisance models')
    parser.add_argument('--train_estimators', type=int, default=1,
                       help='Train causal estimators from scratch or load pretrained estimators')
    parser.add_argument('--automl_selection', type=int, default=1,
                       help= '0: Whole grid of nuisance models; 1: AutoML selection of nuisance models')
    parser.add_argument('--slurm_exp', type=int, default=0,
                       help='0: None; 1: Parallelize across datasets; 2: Parallelize across final models' )
    parser.add_argument('--root_dir', type=str, default='/scratch/cate_eval_analysis/')
    parser.add_argument('--res_dir', type=str, default='results_final')

    args = parser.parse_args()

    #Experiments on Slurm
    if args.slurm_exp == 1:
        # dataset_list = ['twins', 'lalonde_psid1', 'lalonde_cps1', 'orthogonal_ml_dgp']
        # dataset_list = pickle.load(open('datasets/acic_2016_heterogenous_list.p', "rb"))
        dataset_list = pickle.load(open('datasets/acic_2018_heterogenous_list.p', "rb"))

        slurm_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
        args.dataset = dataset_list[slurm_idx]

    dataset_name= args.dataset
    grid_size= args.grid_size
    estimator_name= args.estimator
    train_estimators= args.train_estimators
    seed = args.seed
    est_grid_size= args.est_grid_size
    root_dir= os.path.expanduser('~') + args.root_dir
    res_dir= args.res_dir

    #Create Logs Directory
    RESULTS_DIR = root_dir + str(Path(res_dir))

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    save_dir= RESULTS_DIR + '/' + dataset_name + '/' + 'seed_' + str(seed) + '/' +  estimator_name + '/'
    nuisance_stats_dir= RESULTS_DIR + '/' + dataset_name + '/' + 'seed_' + str(seed) + '/' + 'nuisance_models' + '/'

    print('SEED: ', seed)
    random.seed(seed)
    np.random.seed(seed)

    #Load dataset with true ITE, ATE from the generative model
    print('DATASET:', dataset_name)
    dataset_name, dataset_obj= load_dataset_obj(args.dataset, root_dir)

    #Obtain list of EconML estimators
    hparam_list, estimator_func= get_estimators_list(estimator_name)
    print('ESTIMATOR:', estimator_name, estimator_func)

    #Obtain list of nuisance (outcome, propensity) models
    outcome_models, prop_score_models= get_nuisance_models_list()

    # Final nuisance model selection
    final_model_samples= stratified_random_sampler(outcome_models, est_grid_size)

    if args.slurm_exp == 2:
        slurm_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
        final_model_samples= [final_model_samples[slurm_idx]]
        save_location= save_dir + 'logs_' + str(slurm_idx) + '.csv'
    else:
        save_location = save_dir + 'logs.p'

    #Obtain grid of relevant nuisance model for different meta estimators
    #Set approx=True if you want random samples from each type of model instead of the whole grid
    if args.automl_selection:
        # Propensity Model
        model_t = clone(pickle.load(open(nuisance_stats_dir + 'prop.p', "rb")))

        # Outcome Model
        if estimator_name in ['dml_learner', 'causal_forest_learner']:
            model_y = clone(pickle.load(open(nuisance_stats_dir + 'dml.p', "rb")))
        elif estimator_name in ['dr_learner', 'dr_learner_tune_0.1', 'dr_learner_tune_0.01', 's_learner',
                                's_learner_upd']:
            model_y = clone(pickle.load(open(nuisance_stats_dir + 's_learner.p', "rb")))
        elif estimator_name in ['t_learner', 'x_learner']:
            model_y = clone(pickle.load(open(nuisance_stats_dir + 's_learner.p', "rb")))

        nuisance_model = {}
        nuisance_model['model_t'] = {}
        nuisance_model['model_t']['name']= str(model_t).split('(')[0]
        nuisance_model['model_t']['hparam']= str(model_t).split('(')[-1].replace(')','')
        nuisance_model['model_t']['model_func']= model_t

        nuisance_model['model_y'] = {}
        nuisance_model['model_y']['name']= str(model_y).split('(')[0]
        nuisance_model['model_y']['hparam']= str(model_y).split('(')[-1].replace(')','')
        nuisance_model['model_y']['model_func']= model_y

        nuisance_models= [nuisance_model]

    else:
        nuisance_models= get_nusiance_models_grid(outcome_models, prop_score_models, approx=True, grid_size= grid_size)

    # print('Grid Dimensions', len(final_model_samples), len(nuisance_models))

    if os.path.isfile(save_location) and 1==0:
        print('Logs already present for the specific configuration of seed, dataset, and meta-estimator')
    else:
        if len(hparam_list):
            df_list=[]
            for final_model in final_model_samples:
                for nuisance_model in nuisance_models:
                    estimator = estimator_func(nuisance_model, final_model)
                    estimator = EconMLEstimator(estimator_name, estimator)
                    df= run_experiments_for_estimator(
                        dataset_name, dataset_obj,
                        estimator_name, estimator,
                        final_model,
                        seed, train_estimators,
                        save_dir, nuisance_stats_dir
                    )
                    df_list.append(df)
        else:
            final_model= {'name':'none', 'hparam':'none'}
            df_list=[]
            for nuisance_model in nuisance_models:
                estimator = estimator_func(nuisance_model)
                estimator = EconMLEstimator(estimator_name, estimator)
                df= run_experiments_for_estimator(
                    dataset_name, dataset_obj,
                    estimator_name, estimator,
                    final_model,
                    seed, train_estimators,
                    save_dir, nuisance_stats_dir
                )
                df_list.append(df)

        res_df = pd.concat(df_list, axis=0)
        pickle.dump(res_df, open(save_location, 'wb'))
