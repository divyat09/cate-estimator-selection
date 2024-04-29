import numpy as np
import pickle

TOTAL_SEEDS= 20
TOTAL_CATE_ESTIMATORS= 415
N_SAMPLE_SEEDS = 100
N_AGG_SEEDS = 100
CONF = 0.95
SOFTMAX_TEMP_GRID = np.logspace(-3.0, 5.0, num=10)

REGRESSION_SCORES = ['max_error', 'neg_mean_absolute_error', 'neg_median_absolute_error',
                     'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']
REGRESSION_SCORE_DEF = 'r2'
CLASSIFICATION_SCORES = ['accuracy', 'balanced_accuracy', 'average_precision',
                         'f1',
                         'precision',
                         'recall', 'roc_auc']
CLASSIFICATION_SCORE_DEF = 'accuracy'


REALCAUSE_DATASETS_FOLDER = 'realcause_datasets'
DATASET_GROUP_LIST= ['acic_2016', 'lalonde_cps1', 'lalonde_psid1', 'twins']
REAL_CAUSE_DATASET_LIST= ['lalonde_cps1', 'lalonde_psid1', 'twins']
SYN_DATASET_LIST= pickle.load(open('datasets/acic_2016_heterogenous_list.p', 'rb'))
DATASET_LIST= REAL_CAUSE_DATASET_LIST + SYN_DATASET_LIST

NUISANCE_MODEL_CASES= ['t_learner_0', 't_learner_1', 's_learner', 'dml', 'prop']
ESTIMATOR_LIST= ['dr_learner', 'dml_learner', 'causal_forest_learner', 's_learner', 's_learner_upd', 't_learner', 'x_learner']

SCORES = ['value_score', 'value_dr_score', 'value_dr_clip_score',
          'tau_match_score', 'tau_s_score', 'tau_t_score', 'x_score', 'r_score',
          'influence_score', 'influence_clip_score',
          'tau_iptw_score', 'tau_iptw_clip_score',
          'tau_switch_iptw_s_score', 'tau_switch_iptw_t_score', 'tau_cab_iptw_s_score', 'tau_cab_iptw_t_score',
          'tau_dr_s_score', 'tau_dr_s_clip_score', 'tau_dr_t_score', 'tau_dr_t_clip_score',
          'tau_switch_dr_s_score', 'tau_switch_dr_t_score', 'tau_cab_dr_s_score', 'tau_cab_dr_t_score',
          'tau_tmle_s_score', 'tau_tmle_t_score',
          'cal_dr_s_score', 'cal_dr_t_score', 'cal_tmle_s_score', 'cal_tmle_t_score',
          'qini_dr_s_score', 'qini_dr_t_score', 'qini_tmle_s_score', 'qini_tmle_t_score'
          ]

SCORES_SIGN_FLIP= ['cal_dr_s_score', 'cal_dr_t_score', 'cal_tmle_s_score', 'cal_tmle_t_score',
              'qini_dr_s_score', 'qini_dr_t_score', 'qini_tmle_s_score', 'qini_tmle_t_score']

FINAL_MODEL_OPT_SCORE = {
                            'dr_learner': 'tau_dr_t_clip_score',
                            'dml_learner': 'r_score',
                            'x_learner': 'x_score',
                            's_learner_upd': 'tau_s_score'
                       }

SCORES_MAP= {

    'value_score' : 'Value Score',
    'value_dr_score' : 'Value DR Score',
    'value_dr_clip_score': 'Value DR Clip Score',
    'influence_score' : 'Influence Score',
    'influence_clip_score' : 'Influence Clip Score',
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
    'r_score' : 'R Score'
}
