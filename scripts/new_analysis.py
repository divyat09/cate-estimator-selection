import os
import copy
import sys
import argparse
import pandas as pd
from math import isclose
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

NONE_STR = 'none'

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


TOTAL_SEEDS= 10

def dataset_to_group(dataset):
    """Map from dataset name to dataset group (that are conditioned on)"""
    if dataset.startswith('acic_2016'):
        return 'acic_2016'
    elif dataset.startswith('lbidd'):
        return 'lbidd'
    else:
        return dataset


def load_df(save_location='results/'):
    """Load the DataFrame and process it"""
    df = pd.read_csv(save_location + 'logs_cate.csv')
    # # Insert dataset group column
    # df.insert(0, 'dataset_group', df['dataset'].apply(dataset_to_group))

    # Fix issues with data
    # df.loc[df['model_t_score'].isnull(), 'prop-model'] = NONE_STR
    # df.loc[df['model_t_score'].isnull(), 'prop-model-hparam'] = NONE_STR
    # df.loc[df['model_y_score'].isnull(), 'outcome-model'] = NONE_STR        # shouldn't do anything
    # df.loc[df['model_y_score'].isnull(), 'outcome-model-hparam'] = NONE_STR # shouldn't do anything
    # df['value_dr_clip_prop_score'] = df['value_dr_clip_prop_score'] * -1

    return df

def argopt_softmax_temp(x, score):

    argopt_indices= x[score].argmin()
    argopt_post_df= x.iloc[argopt_indices, :]

    pehe= argopt_post_df['pehe'].mean()

    return pehe

def argopt_over_metric(df, argopt_ready_df, opt_type, metric, argopt_cols=None):
    """Function that takes argmin over given metric inside grouped DataFrame"""
    opt_type = opt_type.lower()
    if opt_type == 'min':
        opt_func = min
    elif opt_type == 'max':
        opt_func = max
    else:
        raise ValueError(f'Invalid opt_type: {opt_type}')

    if argopt_cols is None:
        print(f'{metric}-arg{opt_type}ing and keeping ties')  # ({preargminned_cols} were argminned over in the experimental phase)')
    else:
        print(f'{metric}-arg{opt_type}ing over {argopt_cols} and keeping ties')   # ({preargminned_cols} were argminned over in the experimental phase)')
    none_idx = df[metric].isnull()
    argopt_idx = argopt_ready_df[metric].transform(opt_func) == df[metric]
    idx = argopt_idx | none_idx
    return df[idx]

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

    return mean_df, se_df, dataset_agg_df

'''
Major code changes:
- Differentiating between conditioning and aggregation columns is not needed as we use a very specific form of aggregation over dataset groups and seeds as done in the cate analysis script
- Argmin operations can perhaps be simplified? Seems complicated for now
- Oracle PEHE normalization was done incorrectly earlier as it was done post aggregation
- Change aggregation with new functions written in cate analysis file
- Check whether the oracle pehe are consistent across ensembles and non-ensembles; and if not make them consistent
'''


def run_analysis(df,
                 save_location='results/',
                 conditioning_cols=['dataset_group'],
                 aggregation_cols=['dataset', 'seed'],
                 argmin_cols=['meta-estimator', 'final-model', 'final-model-hparan'],
                 argmin_metrics= [],
                 final_metrics= [],
                 preopt_cols=[],
                 preargminned_cols=['outcome-model', 'outcome-model-hparam', 'prop-model', 'prop-model-hparam'],
                 agg_for_ties='mean',
                 agg_for_pehe='mean',
                 plot_chosen_ests=False,
                 normalize_est_counts=False,
                 export_local_vars=False,
                ):
    
    # Prepare DataFrame for argmin
    argmin_ready_df = df.groupby(conditioning_cols + aggregation_cols)
    print(f'# 1. Grouping by {conditioning_cols + aggregation_cols} #\n')
    n_cond_var_groups = len(argmin_ready_df)

    # Argmin using each metric and aggregate over ties
    argminned_dfs = {}
    argminned_agged_dfs = {}
    print(f'## 2. Argminning and then {agg_for_ties}-aggregating over ties:\n')
    print(argmin_metrics)
    for argmin_metric in argmin_metrics:
        # Argmin over the data
        argminned_df = argopt_over_metric(df=df,
                                          argopt_ready_df=argmin_ready_df,
                                          opt_type='min',
                                          metric=argmin_metric,
                                          argopt_cols=argmin_cols)
        argminned_dfs[argmin_metric] = argminned_df
        n_groups_after_argmin = len(argminned_df)

        if n_groups_after_argmin < n_cond_var_groups:
            raise Exception('Fewer groups then expected after argmin')
        elif n_groups_after_argmin > n_cond_var_groups: # Aggregate to get to 1 estimator per group
            print(f'Aggregating {n_groups_after_argmin} tied estimators down to {n_cond_var_groups} using {agg_for_ties} (mean of {n_groups_after_argmin / n_cond_var_groups:.1f} tied estimators per group)\n')
            regrouped_argminned_df = argminned_df.groupby(conditioning_cols + aggregation_cols)
            argminned_agged_df = regrouped_argminned_df.mean()
            argminned_agged_df.reset_index(inplace=True)
        argminned_agged_df.sort_values(conditioning_cols + aggregation_cols, inplace=True)

        argminned_agged_dfs[argmin_metric] = argminned_agged_df

    # Check that all argmin dfs have the same groups
    for argmin_metric in argmin_metrics[1:]:
        assert np.array_equal(argminned_agged_dfs[argmin_metrics[0]][conditioning_cols + aggregation_cols].to_numpy(),
                              argminned_agged_dfs[argmin_metric][conditioning_cols + aggregation_cols]),\
            f'{argmin_metrics[0]} != {argmin_metric}'

    # Put all PEHEs for estimators chosen from different argmin metrics into a single df with PEHE from best estimator
    print('### 3. Putting PEHEs for all estimators chosen from different argmin metrics into a single DF ###\n')
    pehe_df = argminned_agged_dfs[argmin_metrics[0]][conditioning_cols + aggregation_cols].copy()
    
    #TODO: Generate the DF again with oracle pehe for the no htune case; with unit test.
    if 'oracle_pehe' in argminned_agged_dfs['pehe'].columns:
        pehe_df['oracle_pehe'] = argminned_agged_dfs['pehe'].reset_index(drop=True)['oracle_pehe']
    else:
        pehe_df['oracle_pehe'] = argminned_agged_dfs['pehe'].reset_index(drop=True)['pehe']

    for argmin_metric in argmin_metrics[1:]:
        pehe_df[argmin_metric + '_pehe'] = argminned_agged_dfs[argmin_metric].reset_index(drop=True)['pehe']

    # print('##\n')
    # print(pehe_df)
    # pehe_df.round(2).to_csv('results/a.csv')
    #
    # # Debugging: Checking my implementation of argmin over final model columns is same as old implemetation
    # #My computation for argmin
    # cols = {argmin_metric: argmin_ready_df.apply(
    #     lambda x: argopt_softmax_temp(x, argmin_metric) )  for argmin_metric in argmin_metrics}
    # argmin_metric_df = pd.DataFrame(cols)
    # print(argmin_metric_df)
    # argmin_metric_df.round(2).to_csv('results/b.csv')
    # sys.exit()

    # Assert that the oracle PEHE column is the best PEHE (relative to PEHE from other metrics)
    np.testing.assert_array_almost_equal(pehe_df['oracle_pehe'].to_numpy(),
                                         pehe_df.drop(['dataset_group', 'dataset', 'seed'], axis='columns').min(axis=1).to_numpy(),
                                         decimal=8)

    #The correct way is to normalize PEHE before aggregation over datasets inside dataset group
    oracle_pehes = pehe_df['oracle_pehe']
    for column in pehe_df.columns:
        if column.endswith('_pehe') and column != 'oracle_pehe':
            pehe_df[column] = (pehe_df[column] - oracle_pehes)/(1e-4+oracle_pehes)
            pehe_df.rename(columns={column: column[:-5]}, inplace=True)

    print(pehe_df.round(2))
    #PEHE DF is the main object; this is for debugging it pre aggregations
    # pehe_df.to_csv('lets_debug.csv')
    print('##\n')

    # Calculate aggregate PEHE values over different datasets and random seeds
    print(f'#### 4a. {agg_for_pehe.capitalize()}-aggregating over PEHEs ####\n')
    mean_df, se_df, agg_df= aggregate_seed_df(copy.deepcopy(pehe_df), scores=final_metrics)

    print(mean_df.round(2))
    print('##\n')
    mean_df.round(2).to_csv(save_location + 'pehe_norm_mean_df.csv')

    print(se_df.round(2))
    print('##\n')
    se_df.round(2).to_csv(save_location + 'pehe_norm_se_df.csv')

    print(f'#### 4b. Calculating which metrics won in each {conditioning_cols + aggregation_cols} group ####\n')
    dropped_df = pehe_df.drop(['dataset_group', 'dataset', 'seed', 'oracle_pehe'], axis='columns')
    min_pehes = dropped_df.min(axis=1)
    win_df = pehe_df.copy().drop('oracle_pehe', axis='columns')
    for i in range(len(dropped_df)):
        for column in dropped_df.columns:
            if isclose(pehe_df.loc[i, column], min_pehes[i]):
                win_df.loc[i, column] = 1
            else:
                win_df.loc[i, column] = 0

    win_df.rename(lambda col: col[:-4] + 'win' if col.endswith('_pehe') else col, axis='columns', inplace=True)
    #For more readable score names
    win_df.rename(lambda col: SCORES_MAP[col]  if col in SCORES_MAP.keys() else col , axis='columns', inplace=True)

    # Calculate win rates
    print(f'##### 5b. Mean-aggregating wins over {aggregation_cols} within groups by {conditioning_cols} #####')
    win_rate_df = win_df.groupby(conditioning_cols).mean().drop('seed', axis='columns')\
        .rename(lambda col: col[:-4] if col.endswith('_win') else col, axis='columns')

    # win_rate_df.to_csv('win_rate_df.csv')
    # win_rate_df.transpose().to_csv('win_rate_df_trans.csv')
    print(win_rate_df.transpose().round(2).to_latex())

    # Compute the counts of how many meta-estimators of a given type each score chooses
    if plot_chosen_ests:
        total_est_counts = df['meta-estimator'].value_counts().sort_index()
        for argmin_metric, argminned_df in argminned_dfs.items():

            font = {'family': 'normal',
                    'size': 13
                    }

            matplotlib.rc('font', **font)

            if argmin_metric not in ['tau_dr_t_clip_score', 'x_score', 'rscore', 'tau_s_score', 'pehe']:
                continue

            if argmin_metric == 'rscore':
                argmin_metric= 'R_Score'
            if argmin_metric == 'tau_s_score':
                argmin_metric = 'S_Score'
            if argmin_metric == 'tau_dr_t_clip_score':
                argmin_metric = 'DR_Score'
            if argmin_metric == 'x_score':
                argmin_metric = 'X_Score'
            if argmin_metric == 'pehe':
                argmin_metric= 'Oracle_Score'

            counts = argminned_df['meta-estimator'].value_counts().sort_index()
            assert counts.index.equals(total_est_counts.index)
            if normalize_est_counts:
                counts = counts / total_est_counts
            counts.plot.bar()
            plt.title(argmin_metric)
            plt.subplots_adjust(bottom=.3)
            labels= ['Causal Forest', 'DML Learner', 'DR Learner', 'S Learner', 'Proj S Learner', 'T Learner', 'X Learner']
            plt.xticks([0, 1, 2, 3, 4, 5, 6], labels, rotation='30'
                                                               '')
            plt.grid(color='black', linewidth=1, axis='both', alpha=0.5)
            plt.tight_layout()
            plt.savefig(save_location + argmin_metric  + '.pdf')

    if export_local_vars:
        globals().update(locals())


if __name__ == '__main__':

    # Input Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, default='cate',
                    help= 'cate; cate_htune')
    
    args = parser.parse_args()
    case= args.case

    if 'htune' in case:
        save_location= os.getcwd() + '/results/htune/'
    else:
        save_location= os.getcwd() + '/results/'
    df = load_df(save_location= save_location)

    # SCORES= [col for col in df if 'score' in col]
    ARGMIN_METRICS= ['pehe']
    for score in SCORES:
        ARGMIN_METRICS.append(score)

    run_analysis(df=df,
                 save_location= save_location,
                 conditioning_cols=['dataset_group'],
                 aggregation_cols=['dataset', 'seed'],
                 argmin_cols=['meta-estimator', 'final-model', 'final-model-hparan'],
                 argmin_metrics= ARGMIN_METRICS,
                 final_metrics= SCORES,
                 preopt_cols=[],
                 plot_chosen_ests=False,
                 normalize_est_counts=True,
                 export_local_vars=True,
                 )