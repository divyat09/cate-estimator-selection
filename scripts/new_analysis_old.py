import os
import sys
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

ARGMIN_METRICS= ['pehe']
for score in SCORES:
    ARGMIN_METRICS.append(score)

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


def load_df(ensemble=False):
    """Load the DataFrame and process it"""
    if ensemble:
        df = pd.read_csv(os.getcwd() + '/results/' + '/logs-ensemble.csv')
    else:
        df = pd.read_csv(os.getcwd() + '/results/' + 'logs_cate.csv')

    # # Insert dataset group column
    # df.insert(0, 'dataset_group', df['dataset'].apply(dataset_to_group))

    # Fix issues with data
    # df.loc[df['model_t_score'].isnull(), 'prop-model'] = NONE_STR
    # df.loc[df['model_t_score'].isnull(), 'prop-model-hparam'] = NONE_STR
    # df.loc[df['model_y_score'].isnull(), 'outcome-model'] = NONE_STR        # shouldn't do anything
    # df.loc[df['model_y_score'].isnull(), 'outcome-model-hparam'] = NONE_STR # shouldn't do anything
    # df['value_dr_clip_prop_score'] = df['value_dr_clip_prop_score'] * -1

    return df


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


def run_analysis(df,
                 conditioning_cols=['dataset_group'],
                 aggregation_cols=['dataset', 'seed'],
                 argmin_cols=['meta-estimator', 'final-model', 'final-model-hparan'],
                 argmin_metrics= ARGMIN_METRICS,
                 preopt_cols=[],
                 preargminned_cols=['outcome-model', 'outcome-model-hparam', 'prop-model', 'prop-model-hparam'],
                 agg_for_ties='mean',
                 agg_for_pehe='mean',
                 plot_chosen_ests=False,
                 normalize_est_counts=False,
                 export_local_vars=False):
    
    # Get columns that specify a unique estimator
    n_estimator_cols = 10
    estimator_cols = df.columns[:n_estimator_cols]

    # Check that all columns used to specify an estimator are found above
    assert sorted(conditioning_cols + aggregation_cols + argmin_cols + preopt_cols + preargminned_cols) == sorted(estimator_cols)
    preargmin_ready_df = df.groupby(conditioning_cols + aggregation_cols + argmin_cols + preopt_cols)
    # Check that there is only 1 nuisance model per group (used AutoML)
    # assert preargmin_ready_df.size().min() == 1 == preargmin_ready_df.size().max()

    print(f'\n\nAnalysis overview:\n'
          f'1.Argopting over {preopt_cols}\n'
          f'2.Argmining over {argmin_cols}\n'
          f'3.Aggregating over {aggregation_cols}\n'
          f'4.Leaving conditioning over {conditioning_cols}\n'
          f'\n'
          f'Analysis details:\n')

    # Argmin/argmax over the preopt_cols over the final model score
    preopted_df = df
    if preopt_cols is not None and preopt_cols != []:
        for preopt_col in preopt_cols:
            assert 'final-model' in preopt_col

        # Partition df into 3 DataFrames: one where final model score is MSE, one where it's R2,
        # and one where there's no final model
        final_model_mse_df = df[(df['meta-estimator'] == 'dr_learner') | (df['meta-estimator'] == 'dml_learner')]
        final_model_r2_df = df[(df['meta-estimator'] == 's_learner_upd') | (df['meta-estimator'] == 'x_learner')]
        no_final_model_other_df = df[(df['meta-estimator'] == 'causal_forest_learner') |
                                     (df['meta-estimator'] == 's_learner') |
                                     (df['meta-estimator'] == 't_learner')]

        # Assert that the above created a partition
        assert len(final_model_mse_df) + len(final_model_r2_df) + len(no_final_model_other_df) == len(df)
        assert sorted(df['meta-estimator'].unique()) == sorted(list(final_model_mse_df['meta-estimator'].unique()) +
                                                               list(final_model_r2_df['meta-estimator'].unique()) +
                                                               list(no_final_model_other_df['meta-estimator'].unique()))

        print(f'# 0.1 Grouping by {conditioning_cols + aggregation_cols + argmin_cols} #\n')

        print(f'# 0.2.1 Argmining over {preopt_cols} for {final_model_mse_df["meta-estimator"].unique()} estimators #')
        mse_argminned_df = argopt_over_metric(df=final_model_mse_df,
                                              argopt_ready_df=final_model_mse_df.groupby(conditioning_cols + aggregation_cols + argmin_cols),
                                              opt_type='min',
                                              metric='model_final__score',
                                              argopt_cols=preopt_cols)

        print(f'\n# 0.2.2 Argmaxing over {preopt_cols} for {final_model_r2_df["meta-estimator"].unique()} estimators #')
        r2_argmaxed_df = argopt_over_metric(df=final_model_r2_df,
                                            argopt_ready_df=final_model_r2_df.groupby(conditioning_cols + aggregation_cols + argmin_cols),
                                            opt_type='max',
                                            metric='model_final__score',
                                            argopt_cols=preopt_cols)

        print(f'\n# 0.2.3 Leaving {no_final_model_other_df["meta-estimator"].unique()} estimators alone #')
        print(f'\n# 0.2.4 Combining best {final_model_mse_df["meta-estimator"].unique()}, {final_model_r2_df["meta-estimator"].unique()}, and {no_final_model_other_df["meta-estimator"].unique()} estimators together #\n')
        preopted_df = pd.concat([mse_argminned_df,
                                 r2_argmaxed_df,
                                 no_final_model_other_df])

    # Prepare DataFrame for argmin
    argmin_ready_df = preopted_df.groupby(conditioning_cols + aggregation_cols)
    print(f'# 1. Grouping by {conditioning_cols + aggregation_cols} #\n')
    n_cond_var_groups = len(argmin_ready_df)

    # Argmin using each metric and aggregate over ties
    argminned_dfs = {}
    argminned_agged_dfs = {}
    print(f'## 2. Argminning and then {agg_for_ties}-aggregating over ties:\n')
    for argmin_metric in argmin_metrics:
        # Argmin over the data
        argminned_df = argopt_over_metric(df=preopted_df,
                                          argopt_ready_df=argmin_ready_df,
                                          opt_type='min',
                                          metric=argmin_metric,
                                          argopt_cols=argmin_cols)
        # argopt_over_metric(df=df, argopt_ready_df=argmin_ready_df, opt_type='min', metric=argmin_metric)
        argminned_dfs[argmin_metric] = argminned_df
        n_groups_after_argmin = len(argminned_df)

        if n_groups_after_argmin < n_cond_var_groups:
            raise Exception('Fewer groups then expected after argmin')
        elif n_groups_after_argmin > n_cond_var_groups: # Aggregate to get to 1 estimator per group
            print(f'Aggregating {n_groups_after_argmin} tied estimators down to {n_cond_var_groups} using {agg_for_ties} (mean of {n_groups_after_argmin / n_cond_var_groups:.1f} tied estimators per group)\n')
            regrouped_argminned_df = argminned_df.groupby(conditioning_cols + aggregation_cols)
            if agg_for_ties == 'mean':
                argminned_agged_df = regrouped_argminned_df.mean()
            elif agg_for_ties == 'median':
                argminned_agged_df = regrouped_argminned_df.median()
            elif agg_for_ties == 'min':
                argminned_agged_df = regrouped_argminned_df.min()
            elif agg_for_ties == 'max':
                argminned_agged_df = regrouped_argminned_df.max()
            else:
                raise Exception(f'Currently unsupported aggregation for argmin ties: {agg_for_ties}')
            argminned_agged_df.reset_index(inplace=True)
        argminned_agged_df.sort_values(conditioning_cols + aggregation_cols, inplace=True)

        argminned_agged_dfs[argmin_metric] = argminned_agged_df

    print('##\n')

    # Check that all argmin dfs have the same groups
    for argmin_metric in argmin_metrics[1:]:
        assert np.array_equal(argminned_agged_dfs[argmin_metrics[0]][conditioning_cols + aggregation_cols].to_numpy(),
                              argminned_agged_dfs[argmin_metric][conditioning_cols + aggregation_cols]),\
            f'{argmin_metrics[0]} != {argmin_metric}'

    # Put all PEHEs for estimators chosen from different argmin metrics into a single df with PEHE from best estimator
    print('### 3. Putting PEHEs for all estimators chosen from different argmin metrics into a single DF ###\n')
    pehe_df = argminned_agged_dfs[argmin_metrics[0]][conditioning_cols + aggregation_cols].copy()
    pehe_df['oracle_pehe'] = argminned_agged_dfs['pehe'].reset_index(drop=True)['pehe']
    for argmin_metric in argmin_metrics[1:]:
        pehe_df[argmin_metric + '_pehe'] = argminned_agged_dfs[argmin_metric].reset_index(drop=True)['pehe']

    # Add ITE variance column
    pehe_df.merge(df.groupby(conditioning_cols + aggregation_cols)['true_ite_var'].mean().reset_index(), on=conditioning_cols + aggregation_cols)

    # Assert that the oracle PEHE column is the best PEHE (relative to PEHE from other metrics)
    np.testing.assert_array_almost_equal(pehe_df['oracle_pehe'].to_numpy(),
                                         pehe_df.drop(['dataset_group', 'dataset', 'seed'], axis='columns').min(axis=1).to_numpy(),
                                         decimal=8)

    # Calculate aggregate PEHE values
    print(f'#### 4a. {agg_for_pehe.capitalize()}-aggregating over PEHEs ####\n')
    grouped_pehe_df = pehe_df.groupby(conditioning_cols)

    if agg_for_pehe == 'mean':
        agg_pehe_df = grouped_pehe_df.mean()
    elif agg_for_pehe == 'min':
        agg_pehe_df = grouped_pehe_df.min()
    elif agg_for_pehe == 'max':
        agg_pehe_df = grouped_pehe_df.max()
    else:
        raise ValueError(f'Invalid agg_for_pehe: {agg_for_pehe}')
    # agg_pehe_df.drop('seed', axis='columns', inplace=True)

    print(agg_pehe_df.transpose().round(2).to_latex())
    agg_pehe_df.transpose().round(2).to_csv('results/pehe_df_trans_mean.csv')

    #Standard error across seeds
    # agg_pehe_df.transpose().agg("sem", axis="columns", level=0).round(2).to_csv('results/pehe_df_trans_sd.csv')
    # print(agg_pehe_df.transpose().agg("sem", axis="columns", level=0).round(2).to_latex())


    # Normalize PEHEs, using the oracle PEHE and calculating percentage change from it
    print(f'##### 5a. Normalizing aggregated PEHEs using percentage change from the oracle #####\n')
    oracle_pehes = agg_pehe_df['oracle_pehe']
    normalized_agg_pehe_df = agg_pehe_df.sub(oracle_pehes, axis=0).divide(oracle_pehes, axis=0)\
        .rename(lambda col: col[:-4] + 'pc' if col.endswith('_pehe') else col, axis='columns')

    normalized_agg_pehe_df.transpose().round(2).to_csv('results/pehe_norm_df_trans_mean.csv')
    print(normalized_agg_pehe_df.transpose().round(2).to_latex())

    #Standard error across seeds
    # normalized_agg_pehe_df.transpose().agg("sem", axis="columns", level=0).round(2).to_csv('results/pehe_norm_df_trans_sd.csv')
    # print(normalized_agg_pehe_df.transpose().agg("sem", axis="columns", level=0).round(2).to_latex())
    # sys.exit()
    
    # Calculate which metrics won
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

            if argmin_metric not in ['tau_dr_score', 'rscore', 'pehe']:
                continue

            if argmin_metric == 'rscore':
                argmin_metric= 'R Score'
            if argmin_metric == 'tau_dr_score':
                argmin_metric = 'Tau DR Score'
            if argmin_metric == 'pehe':
                argmin_metric= 'Oracle Score'

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
            plt.savefig('results/' + argmin_metric  + '.pdf')

    if export_local_vars:
        globals().update(locals())


if __name__ == '__main__':
    df = load_df()

    run_analysis(df=df,
                 conditioning_cols=['dataset_group'],
                 aggregation_cols=['dataset', 'seed'],
                 argmin_cols=['meta-estimator', 'final-model', 'final-model-hparan'],
                 preopt_cols=[],
                 plot_chosen_ests=False,
                 normalize_est_counts=True,
                 export_local_vars=True)

    # run_analysis(df=df,
    #              conditioning_cols=['dataset_group'],
    #              aggregation_cols=['dataset', 'seed'],
    #              argmin_cols=['meta-estimator', 'final-model'],
    #              preopt_cols=['final-model-hparan'],
    #              export_local_vars=True)
    #
    # run_analysis(df=df,
    #              conditioning_cols=['dataset_group'],
    #              aggregation_cols=['dataset', 'seed'],
    #              argmin_cols=['meta-estimator'],
    #              preopt_cols=['final-model', 'final-model-hparan'],
    #              export_local_vars=True)


sys.exit()


### Analyze the ensemble DataFrame ###

# Load ensemble DataFrame
df = load_df(ensemble=True)

# Remove rows with null ensemble scores
df = df[df['ensemble_score'].notnull()]

# TODO: try using the PEHE for this to debug why the ensembles are performing worse
# Argmin over ensemble_score
argmin_ready_df = df.groupby(['dataset_group', 'dataset', 'seed', 'ensemble_type'])
n_cond_var_groups = len(argmin_ready_df)
argminned_df = argopt_over_metric(df=df, argopt_ready_df=argmin_ready_df, opt_type='min',
                   metric='ensemble_score', argopt_cols=['softmax_temp'])

# TODO: try just ensembling these as a sanity check
# Aggregate over ties
n_groups_after_argmin = len(argminned_df)
agg_for_ties = 'mean'
print(f'Aggregating {n_groups_after_argmin} tied estimators down to {n_cond_var_groups} using {agg_for_ties} (mean of {n_groups_after_argmin / n_cond_var_groups:.1f} tied estimators per group)\n')
regrouped_argminned_df = argminned_df.groupby(['dataset_group', 'dataset', 'seed', 'ensemble_type'])
argminned_tie_agged_df = regrouped_argminned_df.mean().reset_index().drop('softmax_temp', axis='columns')

# Aggregate over dataset and seed
argminned_agged_df = argminned_tie_agged_df.groupby(['dataset_group', 'ensemble_type']).mean().reset_index().drop('seed', axis='columns')

print(argminned_agged_df.round(2).to_latex())

# TODO
# Calculate oracle
# normalized_agg_pehe_df = agg_pehe_df.sub(oracle_pehes, axis=0).divide(oracle_pehes, axis=0)\
#         .rename(lambda col: col[:-4] + 'pc' if col.endswith('_pehe') else col, axis='columns')

# Calculate wins by argminning over PEHE and count them up by group
win_df = argopt_over_metric(df=argminned_tie_agged_df,
                            argopt_ready_df=argminned_tie_agged_df.groupby(['dataset_group', 'dataset', 'seed']),
                            opt_type='min',
                            metric='pehe',
                            argopt_cols=['ensemble_type'])
win_counts = win_df.groupby('dataset_group')['ensemble_type'].value_counts()

# Build win_rate_df and calculate win rates from the above
dataset_groups = list(df['dataset_group'].unique())
datasets = dataset_groups
win_rate_df = pd.DataFrame({'dataset_group': dataset_groups, 'dataset': datasets})
for ensemble_type in df['ensemble_type'].unique():
    win_rate_df[ensemble_type] = 0

dataset_group_sizes = win_df['dataset_group'].value_counts()
for (dataset_group, score), n_wins in win_counts.iteritems():
    win_rate_df.loc[win_rate_df['dataset_group'] == dataset_group, score] = n_wins / dataset_group_sizes[dataset_group]

# print(win_rate_df.transpose().round(2).to_latex())