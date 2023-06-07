import warnings
warnings.filterwarnings('ignore')

import sys
import pickle
import pathlib
import os
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

path= os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(path)

from utils.helpers import *
from utils.evaluation import *


SCORES_MAP= {

    'value_score' : 'Value Score',
    'value_dr_score' : 'Value DR Score',
    'value_dr_clip_prop_score': 'Value DR Clip Score',
    'influence_score' : 'Influence Score',
    'influence_clip_prop_score' : 'Influence Clip Score',
    'tau_t_score' : 'T Score',
    'tau_s_score' : 'S Score',
    'tau_match_score' : 'Match Score',

    'tau_iptw_score' : 'IPTW Score',
    'tau_iptw_clip_score' : 'IPTW Clip Score',
    'tau_switch_iptw_score' : 'IPTW Switch Score',
    'tau_cab_iptw_score' : 'IPTW CAB Score',
    'tau_switch_iptw_s_score': 'IPTW Switch S Score',
    'tau_switch_iptw_t_score': 'IPTW Switch T Score',
    'tau_cab_iptw_s_score': 'IPTW CAB S Score',
    'tau_cab_iptw_t_score': 'IPTW CAB T Score',

    'tau_dr_score' : 'DR Score',
    'tau_dr_clip_score' : 'DR Clip Score',
    'tau_switch_dr_score' : 'DR Switch Score',
    'tau_cab_dr_score' : 'DR CAB Score',

    'tau_dr_s_score': 'DR S Score',
    'tau_dr_s_clip_score': 'DR S Clip Score',
    'tau_switch_dr_s_score': 'DR Switch S Score',
    'tau_cab_dr_s_score': 'DR CAB S Score',

    'tau_dr_t_score': 'DR T Score',
    'tau_dr_t_clip_score': 'DR T Clip Score',
    'tau_switch_dr_t_score': 'DR Switch T Score',
    'tau_cab_dr_t_score': 'DR CAB T Score',

    'tau_tmle_score' : 'TMLE Score',
    'tau_tmle_s_score': 'TMLE S Score',
    'tau_tmle_t_score': 'TMLE T Score',

    'cal_dr_score' : 'Cal DR Score',
    'cal_tmle_score' : 'Cal TMLE Score',
    'cal_dr_s_score': 'Cal DR S Score',
    'cal_tmle_s_score': 'Cal TMLE S Score',
    'cal_dr_t_score': 'Cal DR T Score',
    'cal_tmle_t_score': 'Cal TMLE T Score',

    'qini_dr_score' : 'Qini DR Score',
    'qini_tmle_score' : 'Qini TMLE Score',
    'qini_dr_s_score': 'Qini DR S Score',
    'qini_tmle_s_score': 'Qini TMLE S Score',
    'qini_dr_t_score': 'Qini DR T Score',
    'qini_tmle_t_score': 'Qini TMLE T Score',

    'x_score' : 'X Score',
    'rscore' : 'R Score'
}

ESTIMATOR_LIST= ['dr_learner', 'dml_learner', 'causal_forest_learner', 's_learner', 's_learner_upd', 't_learner', 'x_learner']

TOTAL_SEEDS= 10

REAL_DATASET_NAMES = [
                'twins',
                'lalonde_psid1',
                'lalonde_cps1'
                ]

# #Heterogenous ACIC 2016
SYN_DATASET_NAMES = pickle.load(open('datasets/acic_2016_heterogenous_list.p', 'rb'))
# #Heterogenous ACIC 2018
SYN_DATASET_NAMES += pickle.load(open('datasets/acic_2018_heterogenous_list.p', 'rb'))

# SYN_DATASET_NAMES= []
DATASET_NAMES = REAL_DATASET_NAMES + SYN_DATASET_NAMES

root_dir = os.path.expanduser('~') + '/scratch/cate_eval_analysis/'
RESULTS_DIR = root_dir + str(Path('results_final'))

def compare_ensemble_improvement(ref_df, df):

    dataset_list= [
                    'acic_2016',
                    'lbidd',
                    'lalonde_cps1',
                    'lalonde_psid1',
                    'twins'
                ]
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

        for dataset in dataset_list:
            
            ref_mean= float(ref_pehe[dataset].split('(')[0])
            ref_sem= float(  ref_pehe[dataset].split('(')[-1].split(')')[0] )

            curr_mean= float( curr_pehe[dataset].split('(')[0])
            curr_sem= float(  curr_pehe[dataset].split('(')[-1].split(')')[0] )

            if ref_mean + 1.96 * ref_sem < curr_mean - 1.96 * curr_sem :
                comp_item[dataset]= -1    
            elif curr_mean + 1.96 * curr_sem < ref_mean - 1.96 * ref_sem :
                win_count+= 1
                comp_item[dataset]= 1
            else:
                tie_count+=1
                comp_item[dataset]= 0
            total_count+=1

        comp_df.append(comp_item)
    
    print('here')
    comp_df= pd.DataFrame.from_dict(comp_df)
    print(comp_df.to_latex())

    print('Win Percentage: ', 100*win_count/total_count)
    print('Tie Percentage: ', 100*tie_count/total_count)
    return 

def merge_mean_sem_df(mean_df, se_df, case='', save_location='results/', dominating_criteria= 'min'):

    dataset_list= [
                    'acic_2016',
                    'lbidd',
                    'lalonde_cps1',
                    'lalonde_psid1',
                    'twins'
                ]
    
    print(mean_df.head())
    print(se_df.head())

    mean_df = mean_df.rename(columns={"Unnamed: 0": "Metric"}).to_dict('records')
    se_df = se_df.rename(columns={"Unnamed: 0": "Metric"}).to_dict('records')

    merged_df= []
    for idx in range(len(mean_df)):

        item_mean= mean_df[idx]
        item_se= se_df[idx]

        if item_mean['Metric'] != item_se['Metric']:
            print('Error')
            sys.exit(-1)

        item_merged= {}
        item_merged['Metric']=  SCORES_MAP[ item_mean['Metric'].replace('_pc', '')]

        for key in dataset_list:
            item_merged[key]= str(item_mean[key]) + ' ( ' +  str(item_se[key]) +  ' ) '
        merged_df.append(item_merged)

    merged_df= pd.DataFrame.from_dict(merged_df)
    merged_df.to_csv(save_location + case + '.csv', float_format='%.2f', index=False)

    res = {}
    for idx in range(len(mean_df)):

        item_mean = mean_df[idx]
        item_se = se_df[idx]

        if item_mean['Metric'] != item_se['Metric']:
            print('Error')
            sys.exit(-1)

        metric = SCORES_MAP[ item_mean['Metric'].replace('_pc', '')]
        for key in dataset_list:
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

            if 'value' in metric:
                continue

            flag = 1
            for other_metric in metrics:

                if metric == other_metric:
                    continue
                
                if dominating_criteria == 'min':                
                    if res[dataset][other_metric]["upper"] < res[dataset][metric]["lower"] or res[dataset][metric]["se"] > 10.0:
                        flag = 0
                        break
                
                elif dominating_criteria == 'max':
                    if res[dataset][other_metric]["lower"] > res[dataset][metric]["upper"] or res[dataset][metric]["se"] > 10.0:
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
        for dataset in df[metric]:
            curr_df[dataset] = '\\textbf{' + curr_df[dataset] + '}'
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

#Load dataset for Ensemble estimators
def load_ensemble_df(save_location='results/'):
    """Load the DataFrame and process it"""

    final_df= []
    for dataset_name in DATASET_NAMES:
        print(dataset_name)
        for seed in range(TOTAL_SEEDS):
            curr_dir=  RESULTS_DIR + '/' + dataset_name + '/' 'seed_' + str(seed) + '/'
            # sub_df= pd.read_csv(curr_dir + 'ensemble.csv')
            sub_df= pd.read_csv(curr_dir + 'ensemble_upd.csv')
            final_df.append(sub_df)

    final_df= pd.concat(final_df, axis=0)

    # Insert dataset group column
    final_df.insert(0, 'dataset_group', final_df['dataset'].apply(dataset_to_group))

    # Save the dataframe
    final_df.to_csv(save_location + 'logs_ensemble.csv', float_format='%.2f', index=False)

    return 

# Load dataset for CATE estimators
def load_cate_df(save_location='results/', final_models_htune=0):
    """Load the DataFrame and process it"""

    final_df= []
    for dataset_name in DATASET_NAMES:
        print(dataset_name)
        for seed in range(TOTAL_SEEDS):

            dataset_df = []
            oracle_pehe = np.inf
            for estimator in ESTIMATOR_LIST:
                curr_dir = RESULTS_DIR + '/' + dataset_name + '/' 'seed_' + str(seed) + '/' + estimator

                sub_df = []
                for log_file in pathlib.Path(curr_dir).glob('logs.p'):
                    curr_df = pickle.load(open(log_file, 'rb'))
                    
                    #Find the best estimator for given dataset and meta estimator class
                    oracle_pehe= min( oracle_pehe, curr_df['pehe'].min() )

                    # Get DataFrame with final model hyperparameter selection done using appropriate score
                    if final_models_htune:
                        if estimator == 'dr_learner':
                            score_arr= curr_df['tau_dr_t_clip_score']
                            curr_df= curr_df[ score_arr == score_arr.min() ]
                        elif estimator == 'dml_learner':
                            score_arr= curr_df['rscore']
                            curr_df= curr_df[ score_arr == score_arr.min() ]
                        elif estimator == 'x_learner':
                            score_arr= curr_df['x_score']
                            curr_df= curr_df[ score_arr == score_arr.min() ]
                        elif estimator == 's_learner_upd':
                            score_arr= curr_df['tau_s_score']
                            curr_df= curr_df[ score_arr == score_arr.min() ]

                    sub_df.append(curr_df)

                if len(sub_df):
                    sub_df = pd.concat(sub_df, axis=0)
                    sub_df.drop(columns=['ite-estimates'])
                    sub_df.drop(columns=['ite-estimates-train'])

                # To ensure consistency that lower value for the metric is better
                for metric in ['cal_dr_s_score', 'cal_dr_t_score', 'cal_tmle_s_score', 'cal_tmle_t_score', 'qini_dr_s_score',
                               'qini_dr_t_score', 'qini_tmle_s_score', 'qini_tmle_t_score']:
                    sub_df[metric] = -1 * sub_df[metric]

                dataset_df.append(sub_df)

            dataset_df = pd.concat(dataset_df, axis=0)
            dataset_df.insert(0, 'oracle_pehe', oracle_pehe)
            final_df.append(dataset_df)

    final_df= pd.concat(final_df, axis=0)

    # Insert dataset group column
    final_df.insert(0, 'dataset_group', final_df['dataset'].apply(dataset_to_group))

    # Save the dataframe
    final_df.to_csv(save_location + 'logs_cate.csv', float_format='%.2f', index=False)

    # Fix issues with data
    # df.loc[df['model_t_score'].isnull(), 'prop-model'] = NONE_STR
    # df.loc[df['model_t_score'].isnull(), 'prop-model-hparam'] = NONE_STR
    # df.loc[df['model_y_score'].isnull(), 'outcome-model'] = NONE_STR        # shouldn't do anything
    # df.loc[df['model_y_score'].isnull(), 'outcome-model-hparam'] = NONE_STR # shouldn't do anything
    # df['value_dr_clip_prop_score'] = df['value_dr_clip_prop_score'] * -1

    return


#Aggregation over random seeds specific to our analysis
def aggregate_seed_df(df, scores=[]):

    '''
    Function: First we take mean over different datasets belonging to a particular configuration and then compute the mean and standard error over random seeds
    '''

    # Take the mean over different datasets belonging to each dataset group and seed
    grouped_df = df.groupby(['dataset_group', 'seed'])
    cols = {score: grouped_df.apply(lambda x: x[score].mean()) for score in scores}
    dataset_agg_df = pd.DataFrame(cols)
    print(dataset_agg_df.shape)

    # Take the mean and standard error over different seeds
    grouped_df = dataset_agg_df.groupby(['dataset_group'])

    cols = {score: grouped_df.apply(lambda x: x[score].mean()) for score in scores}
    mean_df = pd.DataFrame(cols).T.round(2)

    cols = {score: grouped_df.apply(lambda x: x[score].std()) / np.sqrt(TOTAL_SEEDS) for score in scores}
    se_df = pd.DataFrame(cols).T.round(2)

    return mean_df, se_df


# Compute Rank Correlation
def compute_rank_correlation(save_location= 'results/', merge_score=True):

    #Read the dataframe file
    if merge_score:
        df= pd.read_csv(save_location+'logs_cate_merged.csv')
    else:
        df= pd.read_csv(save_location+'logs_cate.csv')
    scores= [col for col in df if 'score' in col]

    # Conditioning on dataset group, instances, and seeds to compute the spearman rank correlation based on each score
    grouped_df = df.groupby(['dataset_group', 'dataset', 'seed'])
    cols = {score: grouped_df.apply(
        lambda x: spearmanr(x['pehe'].to_numpy(dtype='float32'), x[score].to_numpy(dtype='float32'))[0]) for score
        in scores}
    correlation_df = pd.DataFrame(cols)

    # Aggregate over different datasets and random seeds
    mean_df, se_df= aggregate_seed_df(copy.deepcopy(correlation_df), scores=scores)

    mean_df.to_csv(save_location + 'rank_corr_mean_df.csv')
    se_df.to_csv(save_location + 'rank_corr_se_df.csv')

    return


def argopt_softmax_temp(x, score):

    indices= x['ensemble_type'] == score
    argopt_ready_df= x[indices]
    convert_dict = {'ensemble_score': float,
                    'pehe': float,
                    'oracle_pehe': float
                    }
    argopt_ready_df= argopt_ready_df.astype(convert_dict)

    argopt_indices= argopt_ready_df['ensemble_score'].argmin()
    # argopt_indices= argopt_ready_df['pehe'].argmin()
    argopt_post_df= argopt_ready_df.iloc[argopt_indices, :]

    pehe= argopt_post_df['pehe'].mean()
    return pehe


#Compute Ensemble PEHE
def compute_ensemble_pehe(save_location='results/', merge_score= True):

    #Read the dataframe file
    if merge_score:
        df= pd.read_csv(save_location+'logs_ensemble_merged.csv')
    else:
        df= pd.read_csv(save_location+'logs_ensemble.csv')
    scores= df['ensemble_type'].unique()

    # Argmin over the softmax temperature column to get the best performing ensemble based on each score
    grouped_df = df.groupby(['dataset_group', 'dataset', 'seed'])
    cols = {score: grouped_df.apply(lambda x: argopt_softmax_temp(x, score)) for score in scores}
    cols['oracle_pehe']= grouped_df.apply(lambda x: x['oracle_pehe'].mean())
    ensemble_df = pd.DataFrame(cols)

    oracle_pehes = ensemble_df['oracle_pehe']
    for score in scores:
        ensemble_df[score] = (ensemble_df[score] - oracle_pehes)/(1e-4+oracle_pehes)

    print(ensemble_df.round(2))

    # Aggregate over different datasets and random seeds
    mean_df, se_df= aggregate_seed_df(copy.deepcopy(ensemble_df), scores=scores)

    mean_df.to_csv(save_location + 'ensemble_norm_pehe_mean_df.csv')
    se_df.to_csv(save_location +  'ensemble_norm_pehe_se_df.csv')

    return


if __name__ == '__main__':

    # Input Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis_case', type=str, default='cate_pehe',
                    help= 'cate_pehe; cate_rank_corr; cate_win_rate; ensemble_pehe')
    parser.add_argument('--merge_score', type=int, default=0,
                    help= '')
    parser.add_argument('--generate_df', type=int, default=0,
                    help= '')

    args = parser.parse_args()
    analysis_case= args.analysis_case
    generate_df= args.generate_df
    merge_score= args.merge_score

    if 'htune' in analysis_case:
        save_location= os.getcwd() + '/results/htune/'
    else:
        save_location= os.getcwd() + '/results/'

    if generate_df:
        #TODO: Get better naming for different analysis cases; cate_htune and cate will activate two statements if not for the order of if,else statements
        if 'cate_htune' in analysis_case:
            load_cate_df(save_location= save_location, final_models_htune=1)
        elif 'cate' in analysis_case:
            load_cate_df(save_location=save_location)
        elif 'ensemble' in analysis_case:
            load_ensemble_df(save_location=save_location)

    if analysis_case in ['cate_pehe', 'cate_htune_pehe']:
        mean_df = pd.read_csv(save_location + 'pehe_norm_mean_df.csv')
        se_df = pd.read_csv(save_location + 'pehe_norm_se_df.csv')
        merge_mean_sem_df(mean_df, se_df, case=analysis_case, save_location= save_location, dominating_criteria='min')
 
    elif analysis_case in ['cate_rank_corr', 'cate_htune_rank_corr']:
        compute_rank_correlation(save_location= save_location, merge_score= merge_score)
        mean_df = pd.read_csv(save_location + 'rank_corr_mean_df.csv')
        se_df = pd.read_csv(save_location + 'rank_corr_se_df.csv')
        merge_mean_sem_df(mean_df, se_df, case=analysis_case, save_location= save_location, dominating_criteria='max')

    elif analysis_case in ['ensemble_pehe', 'ensemble_htune_pehe']:
        compute_ensemble_pehe(save_location= save_location, merge_score= merge_score)
        mean_df = pd.read_csv(save_location + 'ensemble_norm_pehe_mean_df.csv')
        se_df = pd.read_csv(save_location + 'ensemble_norm_pehe_se_df.csv')
        merge_mean_sem_df(mean_df, se_df, case=analysis_case, save_location= save_location, dominating_criteria='min')
    
    elif analysis_case == 'compare_improvement':
        # ref_df= pd.read_csv(save_location + 'cate_pehe.csv')
        ref_df= pd.read_csv(save_location + '/htune/cate_htune_pehe.csv')
        df= pd.read_csv(save_location + '/htune/ensemble_htune_pehe.csv')
        compare_ensemble_improvement(ref_df, df)