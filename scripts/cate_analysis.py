import warnings
warnings.filterwarnings('ignore')

import sys
import copy
import pickle
import pathlib
import os
from pathlib import Path
import pickle
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

path= os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path)

from utils.consts import TOTAL_SEEDS, DATASET_LIST, DATASET_GROUP_LIST, \
                         SCORES, SCORES_SIGN_FLIP, SCORES_MAP, \
                         FINAL_MODEL_OPT_SCORE, ESTIMATOR_LIST, TOTAL_CATE_ESTIMATORS

root_dir = os.path.expanduser('~') + '/scratch/cate_eval_analysis_camera_ready/'
RESULTS_DIR = root_dir + str(Path('results_final'))

def comp_intervals(ref_itv, curr_itv) -> int:
    """
        Compare where the current interval is better than the reference interval; where being better means having smaller value
    """

    if ref_itv["upper"] < curr_itv["lower"]:
        # Reference interval is completely bounded above than the current interval; hence the current interval is worse
        return -1
    elif curr_itv["upper"] < ref_itv["lower"]:
        # Current interval is completely bounded above than the reference interval; hence the current interval is better
        return 1        
    else:
        #The intervals overlaps and we need to check whether one interval had much higher standard error in comparison to the other
        if curr_itv["se"] > 5* ref_itv["se"]:
            return -1
        elif ref_itv["se"] > 5* curr_itv["se"]:
            return 1
        return 0
        
    return

def get_improvement_stats(ref_df, df):

    ref_df = ref_df.rename(columns={"Unnamed: 0": "Metric"}).to_dict('records')
    df = df.rename(columns={"Unnamed: 0": "Metric"}).to_dict('records')

    win_count= 0
    tie_count= 0
    total_count= 0
    
    comp_df=[]
    for idx in range(len(ref_df)):
        ref_pehe= ref_df[idx]
        curr_pehe= df[idx]

        if ref_pehe['Metric'] != curr_pehe['Metric']:
            print('Error')
            print(ref_pehe['Metric'], curr_pehe['Metric'])
            sys.exit(-1)

        comp_item={}
        comp_item['Metric']= ref_pehe['Metric']

        for dataset in DATASET_GROUP_LIST:
            
            ref_mean= float( ref_pehe[dataset].split('(')[0].split('\\;')[0] ) 
            ref_sem= float( ref_pehe[dataset].split('(')[-1].split(')')[0] )
            ref_itv = {
                        "mean": ref_mean,
                        "se": ref_sem,
                        "lower": ref_mean - 1.96 * ref_sem,
                        "upper": ref_mean + 1.96 * ref_sem
                    }

            curr_mean= float( curr_pehe[dataset].split('(')[0].split('\\;')[0])
            curr_sem= float(  curr_pehe[dataset].split('(')[-1].split(')')[0] )
            curr_itv= {
                        "mean": curr_mean,
                        "se": curr_sem,
                        "lower": curr_mean - 1.96 * curr_sem,
                        "upper": curr_mean + 1.96 * curr_sem
                    } 

            comp_item[dataset]= comp_intervals(ref_itv, curr_itv)
            if comp_item[dataset] == 1:
                win_count+=1
            elif comp_item[dataset] == 0:
                tie_count+=1
            total_count+=1

        comp_df.append(comp_item)
    
    comp_df= pd.DataFrame.from_dict(comp_df)
    print(comp_df.to_latex())

    print('Win Percentage: ', 100*win_count/total_count)
    print('Tie Percentage: ', 100*tie_count/total_count)
    return 

def merge_mean_sem_df(
                        mean_df: pd.DataFrame, 
                        se_df: pd.DataFrame, 
                        save_location: str
                    ):
    
    print(mean_df.head())
    print(se_df.head())

    mean_df = mean_df.reset_index().rename(columns={"index": "Metric"}).to_dict('records')
    se_df = se_df.reset_index().rename(columns={"index": "Metric"}).to_dict('records')

    merged_df= []
    for idx in range(len(mean_df)):

        item_mean= mean_df[idx]
        item_se= se_df[idx]

        if item_mean['Metric'] != item_se['Metric']:
            print('Error')
            sys.exit(-1)

        item_merged= {}
        item_merged['Metric']=  SCORES_MAP[ item_mean['Metric'].replace('_pc', '')]

        for key in DATASET_GROUP_LIST:
            # item_merged[key]=  str(item_mean[key]) + ' ( ' +  str(item_se[key]) +  ' ) '
            item_merged[key]=  str(item_mean[key]) + ' \; ( ' +  str(item_se[key]) +  ' ) '
        merged_df.append(item_merged)

    merged_df= pd.DataFrame.from_dict(merged_df)
    merged_df.to_csv(save_location +  'model_sel_cate_pehe.csv', float_format='%.2f', index=False)

    res = {}
    for idx in range(len(mean_df)):

        item_mean = mean_df[idx]
        item_se = se_df[idx]

        if item_mean['Metric'] != item_se['Metric']:
            print('Error')
            sys.exit(-1)

        metric = SCORES_MAP[ item_mean['Metric'].replace('_pc', '')]
        for key in DATASET_GROUP_LIST:
            if key not in res.keys():
                res[key] = {}
            res[key][metric] = {"lower": 0.0, "upper": 0.0}
            res[key][metric]["mean"] = item_mean[key]
            res[key][metric]["se"] = item_se[key]
            res[key][metric]["lower"] = item_mean[key] - 1.96 * item_se[key]
            res[key][metric]["upper"] = item_mean[key] + 1.96 * item_se[key]

    datasets = res.keys()
    metrics = res['twins'].keys()

    final_res = {}
    for dataset in datasets:
        print('Dataset', dataset)
        final_list = []

        for metric in metrics:
            flag = 1
            for other_metric in metrics:
                if metric == other_metric:
                    continue                
                comp_res= comp_intervals(res[dataset][metric], res[dataset][other_metric])
                if comp_res == 1:
                    flag = 0
                    break

            if flag:
                final_list.append(metric)
        final_res[dataset] = final_list

    df = {}
    for metric in metrics:
        if metric not in df.keys():
            df[metric] = []
        for dataset in datasets:
            if metric in final_res[dataset]:
                df[metric].append(dataset)

    for metric in metrics:
        curr_df = merged_df[merged_df['Metric'] == metric]
        # for dataset in df[metric]:
        #     curr_df[dataset] = '\\textbf{' + curr_df[dataset] + '}'
        for dataset in datasets:
            if dataset in df[metric]:
                curr_df[dataset] = '$ \mathbf{' + curr_df[dataset] + '} $'
            else:
                curr_df[dataset] = '$ ' + curr_df[dataset] + ' $'
        merged_df[merged_df['Metric'] == metric] = curr_df

    print(merged_df.to_latex(escape=False, index=False))

    return

def dataset_to_group(dataset):
    """Map from dataset name to dataset group (that are conditioned on)"""
    if dataset.startswith('acic_2016'):
        return 'acic_2016'
    elif dataset.startswith('lbidd'):
        return 'lbidd'
    else:
        return dataset

def sanity_check_cate_df(
                    save_location: str,
                    analysis_case: str,
                    specific_meta_learner: str
                ):
    """Perform sanity checks on the dataframe containing the CATE estimators and their evaluated scores"""
    
    #Read the dataframe file
    df = pd.read_csv(save_location + 'logs_cate.csv')
    print('Nan values in the dataframe: ', df.isnull().sum())

    if analysis_case == 'single_level_selection':
        for dataset_name in DATASET_LIST:
            for seed in range(TOTAL_SEEDS):
                indices= df['dataset'] == dataset_name
                indices= indices & (df['seed'] == seed)
                sub_df= df[indices]
                if sub_df.shape[0] != TOTAL_CATE_ESTIMATORS:
                    print('ERROR: Some Meta Estimators are missing!')
                    print(sub_df['meta-estimator'].value_counts())
                        
    elif analysis_case == 'two_level_selection':
        for dataset_name in DATASET_LIST:
            for seed in range(TOTAL_SEEDS):
                indices= df['dataset'] == dataset_name
                indices= indices & (df['seed'] == seed)
                sub_df= df[indices]
                total_meta_estimators= len(sub_df['meta-estimator'].unique())
                if total_meta_estimators != len(ESTIMATOR_LIST):
                    print('ERROR: Some Meta Estimators are missing!')
    
    elif analysis_case == 'meta_learner_selection':
        for dataset_name in DATASET_LIST:
            for seed in range(TOTAL_SEEDS):
                indices= df['dataset'] == dataset_name
                indices= indices & (df['seed'] == seed)
                sub_df= df[indices]
                meta_estimators= sub_df['meta-estimator'].unique()
                for estimator in ESTIMATOR_LIST:
                    if estimator in meta_estimators and estimator!= specific_meta_learner:
                        print('ERROR: Extra meta learners present in the dataframe other than the specific meta learner!')
                    if estimator not in meta_estimators and estimator == specific_meta_learner:
                        print('ERROR: The specific meta learner the dataframe was conditioned on is not present')

    elif analysis_case == 'ensemble_selection':
        for dataset_name in DATASET_LIST:
            for seed in range(TOTAL_SEEDS):
                indices= df['dataset'] == dataset_name
                indices= indices & (df['seed'] == seed)
                sub_df= df[indices]
                if sub_df['ensemble_type'].value_counts().sum() % len(SCORES) !=0:
                    print('ERROR: Some Ensembles are missing!')

    return 

def get_dataset_df(dataset_name: str, seed: int) -> pd.DataFrame:
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
            sub_df= sub_df.drop(columns=['ite-estimates'])
            sub_df= sub_df.drop(columns=['ite-estimates-train'])

        # To ensure consistency that lower value for the metric is better
        for metric in SCORES_SIGN_FLIP:
            sub_df[metric] = -1 * sub_df[metric]

        dataset_df.append(sub_df)
    dataset_df= pd.concat(dataset_df, axis= 0)

    return dataset_df

# Load dataset for CATE estimators
def load_cate_df(
                    save_location: str,
                    analysis_case: str,
                    specific_meta_learner: str
                ):
    """Load the DataFrame and process it"""

    final_df= []
    for dataset_name in DATASET_LIST:
        print(dataset_name)
        for seed in range(TOTAL_SEEDS):            
            dataset_df= get_dataset_df(dataset_name= dataset_name, seed= seed)
            dataset_df.insert(0, 'oracle_pehe', dataset_df['pehe'].min())
            
            if analysis_case == 'single_level_selection':
                final_df.append(dataset_df)

            elif analysis_case == 'two_level_selection':
                for meta_estimator in ESTIMATOR_LIST:
                    estimator_arr= dataset_df['meta-estimator']
                    meta_estimator_df = dataset_df[estimator_arr == meta_estimator]
                    if meta_estimator in ['dr_learner', 'dml_learner', 's_learner_upd', 'x_learner']:
                        # Get DataFrame with final model hyperparameter selection done using appropriate score
                        final_model_opt_score= FINAL_MODEL_OPT_SCORE[meta_estimator]
                        score_arr= meta_estimator_df[final_model_opt_score]
                        # If there are multiple estimators that are minimum corresponding to the metric, then retain all of them                        
                        meta_estimator_df = meta_estimator_df[ score_arr == score_arr.min() ]
                    final_df.append(meta_estimator_df)
            
            elif analysis_case == 'meta_learner_selection':
                estimator_arr= dataset_df['meta-estimator']
                meta_estimator_df = dataset_df[estimator_arr == specific_meta_learner]                     
                final_df.append(meta_estimator_df)                

    # Insert dataset group column
    final_df= pd.concat(final_df, axis=0)
    final_df.insert(0, 'dataset_group', final_df['dataset'].apply(dataset_to_group))

    # Save the dataframe
    print(final_df.columns)
    final_df.to_csv(save_location + 'logs_cate.csv', float_format='%.2f', index=False)

    return

#Load dataset for Ensemble estimators
def load_ensemble_df(save_location='results/'):
    """Load the DataFrame and process it"""

    final_df= []
    for dataset_name in DATASET_LIST:
        print(dataset_name)
        for seed in range(TOTAL_SEEDS):
            curr_dir=  RESULTS_DIR + '/' + dataset_name + '/' 'seed_' + str(seed) + '/'
            sub_df= pickle.load(open(curr_dir + 'ensemble.p', 'rb'))
            final_df.append(sub_df)

    final_df= pd.concat(final_df, axis=0)

    # Insert dataset group column
    final_df.insert(0, 'dataset_group', final_df['dataset'].apply(dataset_to_group))

    # Save the dataframe
    final_df.to_csv(save_location + 'logs_cate.csv', float_format='%.2f', index=False)

    return

def aggregate_seed_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return the dataframes with aggregation over different dataset instances and random seeds for particular dataset group."""

    #Take the mean over different dataset instances conditioned on each dataset group and seed
    grouped_df = df.groupby(['dataset_group', 'seed'])
    cols = {score: grouped_df.apply(lambda x: x[score].mean()) for score in SCORES}
    dataset_agg_df = pd.DataFrame(cols)

    #Take the mean and standard error over different seeds
    grouped_df = dataset_agg_df.groupby(['dataset_group'])

    cols = {score: grouped_df.apply(lambda x: x[score].mean()) for score in SCORES}
    mean_df = pd.DataFrame(cols).T.round(2)

    cols = {score: grouped_df.apply(lambda x: x[score].std()) / np.sqrt(TOTAL_SEEDS) for score in SCORES}
    se_df = pd.DataFrame(cols).T.round(2)

    return mean_df, se_df


def argopt_estimator(x: pd.DataFrame, score: str) -> pd.DataFrame:
    """ Return the sliced dataframe from x corresponding to rows with the optimal score values. """

    argopt_indices= x[score].argmin()
    argopt_post_df= x.iloc[argopt_indices, :]

    return argopt_post_df

def argopt_softmax_temp(x: pd.DataFrame, score: str) -> pd.DataFrame:
    """ Return the sliced dataframe from x corresponding to optimal ensembles constructed using the specified score. """

    indices= x['ensemble_type'] == score
    argopt_ready_df= x[indices]
    convert_dict = {'ensemble_score': float,
                    'pehe': float,
                    'oracle_pehe': float
                    }
    argopt_ready_df= argopt_ready_df.astype(convert_dict)

    #Only to debug for selecting with oracle score
    # argopt_indices= argopt_ready_df['pehe'].argmin()

    argopt_indices= argopt_ready_df['ensemble_score'].argmin()
    argopt_post_df= argopt_ready_df.iloc[argopt_indices, :]

    return argopt_post_df

def get_optimal_pehe_per_metric(
                                    save_location: str= '', 
                                    analysis_case: str= '',
                                    normalization: bool=True,
                                ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ 
    Return dataframe with PEHE corresponding to the best estimators per metric.

    Inputs:
        save_location: Path containing the dataframe with all the CATE estimators and their evaluated scores
        analysis_case: Type of model selection strategy: single_level_selection; two_level_selection; ensemble_selection; meta_learner_selection
        normalization: Normalize the PEHE w.r.t the oracle PEHE
    
    Returns:
        mean_df: Dataframe containing the mean PEHE (aggregated over dataset instances and random seeds) of the best estimator for each metric
        se_df: Dataframe containing the standard error in PEHE (aggregated over dataset instances and random seeds) of the best estimator for each metric
    """

    #Read the dataframe file
    df = pd.read_csv(save_location + 'logs_cate.csv')
    grouped_df = df.groupby(['dataset_group', 'dataset', 'seed'])

    # scores= df['ensemble_type'].unique()
    cols={}
    for score in SCORES:
        if analysis_case in ['single_level_selection', 'two_level_selection', 'meta_learner_selection']:
            #Argmin w.r.t each score to get PEHE for the best estimator for that particular score
            argopt_post_df= grouped_df.apply(lambda x: argopt_estimator(x, score)['pehe'].mean())
        elif analysis_case in ['ensemble_selection']:
            # Argmin over the softmax temperature column to get the best performing ensemble based on each score
            argopt_post_df= grouped_df.apply(lambda x: argopt_softmax_temp(x, score)['pehe'].mean())                    
        cols[score]= argopt_post_df
    
    #Create Dataframe from the dictionary containing optimal PEHE for the estimators/ensemble corresponding each score
    cols['oracle_pehe']= grouped_df.apply(lambda x: x['oracle_pehe'].mean())
    pehe_df = pd.DataFrame(cols)
    
    #Normalization w.r.t Oracle PEHE
    if normalization:
        oracle_pehes = pehe_df['oracle_pehe']
        for score in SCORES:
            pehe_df[score] = (pehe_df[score] - oracle_pehes)/(1e-4+oracle_pehes)

    # Aggregate over different datasets and random seeds
    mean_df, se_df= aggregate_seed_df(copy.deepcopy(pehe_df))
    
    return mean_df, se_df


def compute_winner_estimators(save_location: str=''):
    """Compute the number of times each Meta Estimators wins or is optimal w.r.t the oracle score"""

    #Read the dataframe file
    df = pd.read_csv(save_location + 'logs_cate.csv')
    grouped_df = df.groupby(['dataset_group', 'dataset', 'seed'])

    #Argmin w.r.t oracle score to get best estimator for each (dataset group, dataset, seed) triplet
    cols = {}
    argopt_post_df= grouped_df.apply(lambda x: argopt_estimator(x, 'pehe'))
    cols['meta-estimator']= argopt_post_df['meta-estimator']

    #Create Dataframe from the dictionary containing optimal Meta Estimators w.r.t  oracle score
    winner_df= pd.DataFrame(cols)

    # Print the number of times each Meta Estimator wins
    print(100 * winner_df['meta-estimator'].value_counts() /  winner_df['meta-estimator'].value_counts().sum() )

    return

if __name__ == '__main__':

    # Input Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_df', type=int, default=0,
                    help= '')
    parser.add_argument('--analysis_case', type=str, default='single_level_selection',
                    help= 'single_level_selection; two_level_selection; ensemble_selection; meta_learner_selection')
    parser.add_argument('--specific_meta_learner', type=str, default='',
                        help= 'Specific meta learner to be analyzed: dr_learner, dml_learner, x_learner, s_learner_upd')
    parser.add_argument('--get_model_selection_stats', type=int, default=1,
                        help='Generate tables for model selection via metrics')
    parser.add_argument('--get_best_estimator_stats', type=int, default=0,
                        help='Generate plots for best CATE estimators as per the oracle metric')
    parser.add_argument('--compare_improvement', type=int, default=0,
                        help='Compare the different model selection strategy (single/two-level/ensemble)')

    args = parser.parse_args()
    analysis_case= args.analysis_case
    specific_meta_learner= args.specific_meta_learner

    save_location= os.getcwd() + '/results/' + analysis_case + '/'
    if analysis_case == 'meta_learner_selection':
        save_location= save_location + specific_meta_learner + '/'

    if not os.path.exists(save_location):
        os.makedirs(save_location)

    #Generate the overall dataframe containing all CATE Estimators along wih their evaluated scores from the logs
    if args.generate_df:
        if analysis_case in ['single_level_selection', 'two_level_selection', 'meta_learner_selection']:
            load_cate_df(
                            save_location= save_location, 
                            analysis_case= analysis_case,
                            specific_meta_learner= specific_meta_learner
                        )
        elif analysis_case in ['ensemble_selection']:
            load_ensemble_df(save_location= save_location)

    #Obtain results for selecting the best CATE Estimator per metrics
    if args.get_model_selection_stats:

        # sanity_check_cate_df(
        #                     save_location= save_location, 
        #                     analysis_case= analysis_case,
        #                     specific_meta_learner= specific_meta_learner
        #                  )

        mean_df, se_df= get_optimal_pehe_per_metric(
                                                        save_location= save_location, 
                                                        analysis_case= analysis_case
                                                    )
        merge_mean_sem_df(
                            mean_df= mean_df, 
                            se_df= se_df, 
                            save_location= save_location
                        )
    
    #Obtain results for distribution of best Meta Learners (given by oracle metric) per dataset
    if args.get_best_estimator_stats:
        compute_winner_estimators(save_location= save_location)
    
    # Compare the different model selection strategy (single/two-level/ensemble)
    if args.compare_improvement:
        single_level_df= pd.read_csv( os.getcwd() + '/results/' + 'single_level_selection/' + 'model_sel_cate_pehe.csv' )
        two_level_df= pd.read_csv( os.getcwd() + '/results/' + 'two_level_selection/' + 'model_sel_cate_pehe.csv' )
        ensemble_df= pd.read_csv( os.getcwd() + '/results/' + 'ensemble_selection/' + 'model_sel_cate_pehe.csv' )

        get_improvement_stats(single_level_df, two_level_df)
        get_improvement_stats(two_level_df, ensemble_df)
