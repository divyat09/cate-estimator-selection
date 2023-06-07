from typing import List
from statistics import mean
from math import sqrt
import sys
import os
import pickle
import copy
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.model_selection._search import BaseSearchCV
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet, RidgeClassifier

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor,\
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from models.base import BaseGenModel
from causal_estimators.base import BaseEstimator, BaseIteEstimator

from econml.dml import CausalForestDML
from econml.score import RScorer
from econml.grf import CausalForest

from utils.helpers import sample_dataset

STACK_AXIS = 0
CONF = 0.95
REGRESSION_SCORES = ['max_error', 'neg_mean_absolute_error', 'neg_median_absolute_error',
                     'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']
REGRESSION_SCORE_DEF = 'r2'
CLASSIFICATION_SCORES = ['accuracy', 'balanced_accuracy', 'average_precision',
                         'f1',
                         'precision',
                         'recall', 'roc_auc']
CLASSIFICATION_SCORE_DEF = 'accuracy'

def get_nuisance_propensity_pred(w, t, save_dir=''):

    if os.path.isfile(save_dir + 'prop' + '.p'):

        #Propensity Model
        data_size= w.shape[0]
        t= np.reshape(t, (data_size))
        
        prop_model= pickle.load( open(save_dir + 'prop' + '.p', "rb") )                  
        pred_prob= prop_model.predict_proba(w)
        prop_score= pred_prob[:, 0] * (1-t) + pred_prob[:, 1] * (t)
    
    else:        
        print('Error: Nuisance model not trained for evaluation purposes on this dataset')
    
    return pred_prob, prop_score
    
def get_nuisance_outcome_t_pred(w, t, save_dir=''):

    if os.path.isfile(save_dir + 't_learner_0' + '.p') and os.path.isfile(save_dir + 't_learner_1' + '.p'):
        
        data_size= w.shape[0]
        t= np.reshape(t, (data_size))

        #Outcome Models
        out_model_0= pickle.load( open(save_dir + 't_learner_0' + '.p', "rb") )
        out_model_1= pickle.load( open(save_dir + 't_learner_1' + '.p', "rb") )

        mu_0= out_model_0.predict(w)
        mu_1= out_model_1.predict(w)
    
    else:
        print('Error: Nuisance model not trained for evaluation purposes on this dataset')        
        
    return (mu_0, mu_1)

def get_nuisance_outome_s_pred(w, t, save_dir=''):

    print(save_dir)
    if os.path.isfile(save_dir + 's_learner' + '.p'):
        
        data_size= w.shape[0]
        t= np.reshape(t, (data_size, 1))
        
        t0= t*0
        t1= t*0 + 1        
        w0= np.hstack([w, t0])
        w1= np.hstack([w, t1])

        out_model= pickle.load( open(save_dir + 's_learner' + '.p', "rb") )
        mu_0= out_model.predict(w0)
        mu_1= out_model.predict(w1)

    else:        
        print('Error: Nuisance model not trained for evaluation purposes on this dataset')
        
    return (mu_0, mu_1)


def get_nuisance_outcome_dml_pred(w, save_dir=''):

    if os.path.isfile(save_dir + 'dml' + '.p'):

        out_model = pickle.load(open(save_dir + 'dml' + '.p', "rb"))
        mu_dml= out_model.predict(w)

    else:
        print('Error: Nuisance model not trained for evaluation purposes on this dataset')

    return mu_dml

def calculate_metrics(dataset_name, dataset_obj,
                      estimator_name, estimator: BaseEstimator, nuisance_model_config,
                      seed: int, conf_ints=True,
                      return_ite_vectors=False,
                      nuisance_stats_dir= '',
                      debug_save_dir= '',
                      inv_prop_threshold=10):


    fitted_estimators=[]
    fitted_estimators.append(copy.deepcopy(estimator))

    #Training Data
    dataset_samples = sample_dataset(dataset_name, dataset_obj, seed=seed, case='train')
    train_w, train_t, train_y = dataset_samples['w'], dataset_samples['t'], dataset_samples['y']

    #Evaluation Data
    dataset_samples= sample_dataset(dataset_name, dataset_obj, seed=seed, case='eval')
    eval_w, eval_t, eval_y, ate, ite= dataset_samples['w'], dataset_samples['t'], dataset_samples['y'], dataset_samples['ate'], dataset_samples['ite']

    #Nuisance Models
    prop_prob, prop_score= get_nuisance_propensity_pred(eval_w, eval_t, save_dir=nuisance_stats_dir)
    outcome_s_pred= get_nuisance_outome_s_pred(eval_w, eval_t, save_dir=nuisance_stats_dir)
    outcome_t_pred= get_nuisance_outcome_t_pred(eval_w, eval_t, save_dir=nuisance_stats_dir)
    outcome_dml_pred= get_nuisance_outcome_dml_pred(eval_w, save_dir=nuisance_stats_dir)

    #ITE Metrics
    ite_metrics = calculate_ite_metrics(ite, fitted_estimators, eval_w, eval_t)
    ite_mean_metrics = {k: np.mean(v) for k, v in ite_metrics.items()}
#         ite_std_metrics = {'std_of_' + k: np.std(v) for k, v in ite_metrics.items()}    

    metrics= {}
    metrics.update(ite_mean_metrics)
    if return_ite_vectors:
        metrics.update(ite_metrics)

    # Estimator's ITE on validation set
    eval_t0 = eval_t * 0
    eval_t1 = eval_t * 0 + 1

    ite_estimates = np.stack([fitted_estimator.effect(eval_w, eval_t0, eval_t1) for
                              fitted_estimator in fitted_estimators],
                             axis=STACK_AXIS)

    # Estimator's ITE on training set
    train_t0 = train_t * 0
    train_t1 = train_t * 0 + 1

    ite_estimates_train= np.stack([fitted_estimator.effect(train_w, train_t0, train_t1) for
                              fitted_estimator in fitted_estimators],
                             axis=STACK_AXIS)

    #Saving estimators ITE estimate
    metrics.update({'ite-estimates-train': ite_estimates_train})
    metrics.update({'ite-estimates': ite_estimates})

    # #Debugging ITE Logging Error
    # print(ite_metrics['pehe'])
    #
    # pehe_squared = calc_vector_mse(ite_estimates, ite, reduce_axis=(1 - STACK_AXIS))
    # print(np.sqrt(pehe_squared))
    #
    # pehe_squared= np.mean((ite_estimates - ite) ** 2)
    # print(np.sqrt(pehe_squared))
    #
    # log_file= debug_save_dir + 'logs.p'
    # debug_df= pickle.load(open(log_file, 'rb'))
    # ite_estimates= debug_df['ite-estimates'][0]
    # pehe_squared= np.mean((ite_estimates - ite) ** 2)
    # print(np.sqrt(pehe_squared))
    #
    # print(ite.shape)
    # print(ite[:5])
    # print(ite_estimates.shape)
    # print(ite_estimates[0, :5])
    # sys.exit()

    # Compute Value Risk
    score= calculate_value_risk(ite_estimates, eval_w, eval_t, eval_y, dataset_name, prop_score= prop_score)
    metrics.update(score)

    #Compute Value DR Risk
    score= calculate_value_dr_risk(ite_estimates, eval_w, eval_t, eval_y, dataset_name, outcome_pred= outcome_t_pred, prop_score= copy.deepcopy(prop_score))
    metrics.update(score)

    #Compute Value DR Risk
    score= calculate_value_dr_risk(ite_estimates, eval_w, eval_t, eval_y, dataset_name, outcome_pred= outcome_t_pred, prop_score= copy.deepcopy(prop_score), min_propensity= 0.1)
    metrics.update(score)

    #Compute Tau Matching Risk
    score= calculate_tau_match_risk(ite_estimates, eval_w, eval_t, eval_y)
    metrics.update(score)

    # Compute Plug In Tau Score from Van der Schaar paper using S-Learner
    score= calculate_all_tau_risk(ite_estimates,eval_w, eval_t, eval_y, outcome_pred= outcome_s_pred,  prop_score= copy.deepcopy(prop_score), case='s_score')
    metrics.update(score)

    # Compute Plug In Tau Score from Van der Schaar paper using T-Learner
    score= calculate_all_tau_risk(ite_estimates,eval_w, eval_t, eval_y, outcome_pred= outcome_t_pred, prop_score= copy.deepcopy(prop_score), case='t_score')
    metrics.update(score)

    # Compute IPTW Score
    score= calculate_all_tau_risk(ite_estimates,eval_w, eval_t, eval_y, prop_score= copy.deepcopy(prop_score), case='iptw_score')
    metrics.update(score)

    # Compute IPTW Score with propensity clipping
    score= calculate_all_tau_risk(ite_estimates,eval_w, eval_t, eval_y, prop_score= copy.deepcopy(prop_score), case='iptw_clip_score', inv_prop_threshold= inv_prop_threshold)
    metrics.update(score)

    # Compute Switch IPTW Score with S Learner
    score= calculate_all_tau_risk(ite_estimates,eval_w, eval_t, eval_y, outcome_pred= outcome_s_pred, prop_score= copy.deepcopy(prop_score), case='switch_iptw_s_score',  inv_prop_threshold= inv_prop_threshold)
    metrics.update(score)

    # Compute Switch IPTW Score with T Learner
    score= calculate_all_tau_risk(ite_estimates,eval_w, eval_t, eval_y, outcome_pred= outcome_t_pred, prop_score= copy.deepcopy(prop_score), case='switch_iptw_t_score',  inv_prop_threshold= inv_prop_threshold)
    metrics.update(score)

    # Compute CAB IPTW Score with S Learner
    score= calculate_all_tau_risk(ite_estimates,eval_w, eval_t, eval_y, outcome_pred= outcome_s_pred, prop_score= copy.deepcopy(prop_score), case='cab_iptw_s_score',  inv_prop_threshold= inv_prop_threshold)
    metrics.update(score)

    # Compute CAB IPTW Score with T Learner
    score= calculate_all_tau_risk(ite_estimates,eval_w, eval_t, eval_y, outcome_pred= outcome_t_pred, prop_score= copy.deepcopy(prop_score), case='cab_iptw_t_score',  inv_prop_threshold= inv_prop_threshold)
    metrics.update(score)

    # Compute DR Score with outcome function as S Learner
    score= calculate_all_tau_risk(ite_estimates,eval_w, eval_t, eval_y, outcome_pred= outcome_s_pred, prop_score= copy.deepcopy(prop_score), case='dr_s_score')
    metrics.update(score)

    # Compute DR Score with propensity clipping, outcome function as S Learner
    score= calculate_all_tau_risk(ite_estimates,eval_w, eval_t, eval_y, outcome_pred= outcome_s_pred, prop_score= copy.deepcopy(prop_score), case='dr_s_clip_score', inv_prop_threshold= inv_prop_threshold)
    metrics.update(score)

    # Compute DR Score with outcome function as T Learner
    score= calculate_all_tau_risk(ite_estimates,eval_w, eval_t, eval_y, outcome_pred= outcome_t_pred, prop_score= copy.deepcopy(prop_score), case='dr_t_score')
    metrics.update(score)

    # Compute DR Score with propensity clipping, outcome function as T Learner
    score= calculate_all_tau_risk(ite_estimates,eval_w, eval_t, eval_y, outcome_pred= outcome_t_pred, prop_score= copy.deepcopy(prop_score), case='dr_t_clip_score', inv_prop_threshold= inv_prop_threshold)
    metrics.update(score)

    # Compute Switch DR Score with S Learner
    score= calculate_all_tau_risk(ite_estimates,eval_w, eval_t, eval_y, outcome_pred= outcome_s_pred, prop_score= copy.deepcopy(prop_score), case='switch_dr_s_score',  inv_prop_threshold= inv_prop_threshold)
    metrics.update(score)

    # Compute Switch DR Score with T Learner
    score = calculate_all_tau_risk(ite_estimates,eval_w, eval_t, eval_y, outcome_pred= outcome_t_pred, prop_score= copy.deepcopy(prop_score), case='switch_dr_t_score',  inv_prop_threshold= inv_prop_threshold)
    metrics.update(score)

    # Compute CAB DR Score with S Learner
    score= calculate_all_tau_risk(ite_estimates,eval_w, eval_t, eval_y, outcome_pred= outcome_s_pred, prop_score= copy.deepcopy(prop_score), case='cab_dr_s_score',  inv_prop_threshold= inv_prop_threshold)
    metrics.update(score)

    # Compute CAB DR Score with T Learner
    score= calculate_all_tau_risk(ite_estimates,eval_w, eval_t, eval_y, outcome_pred= outcome_t_pred, prop_score= copy.deepcopy(prop_score), case='cab_dr_t_score',  inv_prop_threshold= inv_prop_threshold)
    metrics.update(score)

    #Compute TMLE score with S Learner
    score= calculate_all_tau_risk(ite_estimates,eval_w, eval_t, eval_y, outcome_pred= outcome_s_pred, outcome_dml_pred= outcome_dml_pred, prop_prob= copy.deepcopy(prop_prob), prop_score= copy.deepcopy(prop_score), case='tmle_s_score',  inv_prop_threshold= inv_prop_threshold, save_dir= nuisance_stats_dir)
    metrics.update(score)

    #Compute TMLE score with T Learner
    score= calculate_all_tau_risk(ite_estimates,eval_w, eval_t, eval_y, outcome_pred= outcome_t_pred, outcome_dml_pred= outcome_dml_pred, prop_prob= copy.deepcopy(prop_prob), prop_score= copy.deepcopy(prop_score), case='tmle_t_score',  inv_prop_threshold= inv_prop_threshold, save_dir= nuisance_stats_dir)
    metrics.update(score)

    #Compute Van de Schaar Influence function
    score= calculate_influence_risk(ite_estimates, eval_w, eval_t, eval_y, outcome_pred= outcome_t_pred, prop_prob= prop_prob, min_propensity= 0.01)
    metrics.update(score)

    #Compute RScore Metric
    score= calculate_r_risk(ite_estimates, eval_w, eval_t, eval_y, outcome_pred= outcome_dml_pred, treatment_prob=prop_prob[:, 1])
    metrics.update(score)

    #Compute X Learner Metric
    score= calculate_x_risk(ite_estimates, eval_w, eval_t, eval_y, outcome_pred= outcome_t_pred)
    metrics.update(score)

    #Compute Calibration Score with DR S Learner as base
    score = calculate_calibration_risk(ite_estimates, eval_w, eval_t, eval_y, ite_estimates_train=ite_estimates_train, outcome_pred=outcome_s_pred, prop_score=copy.deepcopy(prop_score), case='dr_s_score')
    metrics.update(score)

    #Compute Calibration Score with DR T Learner as base
    score= calculate_calibration_risk(ite_estimates,eval_w, eval_t, eval_y, ite_estimates_train= ite_estimates_train, outcome_pred= outcome_t_pred, prop_score= copy.deepcopy(prop_score), case='dr_t_score')
    metrics.update(score)

    #Compute Calibration Score with TMLE S Learner as base
    score= calculate_calibration_risk(ite_estimates,eval_w, eval_t, eval_y, ite_estimates_train= ite_estimates_train, outcome_pred= outcome_s_pred, outcome_dml_pred= outcome_dml_pred, prop_prob= copy.deepcopy(prop_prob), prop_score= copy.deepcopy(prop_score), case='tmle_s_score', save_dir= nuisance_stats_dir)
    metrics.update(score)

    #Compute Calibration Score with TMLE T Learner as base
    score= calculate_calibration_risk(ite_estimates,eval_w, eval_t, eval_y, ite_estimates_train= ite_estimates_train, outcome_pred= outcome_t_pred, outcome_dml_pred= outcome_dml_pred, prop_prob= copy.deepcopy(prop_prob), prop_score= copy.deepcopy(prop_score), case='tmle_t_score', save_dir= nuisance_stats_dir)
    metrics.update(score)

    #Compute Qini Score with DR S Learner as base
    score = calculate_qini_risk(ite_estimates, eval_w, eval_t, eval_y, outcome_pred=outcome_s_pred, prop_score=copy.deepcopy(prop_score), case='dr_s_score')
    metrics.update(score)

    #Compute Qini Score with DR T Learner as base
    score= calculate_qini_risk(ite_estimates,eval_w, eval_t, eval_y, outcome_pred= outcome_t_pred, prop_score= copy.deepcopy(prop_score), case='dr_t_score')
    metrics.update(score)

    #Compute Qini Score with TMLE S Learner as base
    score= calculate_qini_risk(ite_estimates,eval_w, eval_t, eval_y, outcome_pred= outcome_s_pred, outcome_dml_pred= outcome_dml_pred, prop_prob= copy.deepcopy(prop_prob), prop_score= copy.deepcopy(prop_score), case='tmle_s_score', save_dir= nuisance_stats_dir)
    metrics.update(score)

    #Compute Qini Score with TMLE T Learner as base
    score= calculate_qini_risk(ite_estimates,eval_w, eval_t, eval_y, outcome_pred= outcome_t_pred, outcome_dml_pred= outcome_dml_pred, prop_prob= copy.deepcopy(prop_prob), prop_score= copy.deepcopy(prop_score), case='tmle_t_score', save_dir= nuisance_stats_dir)
    metrics.update(score)

    return metrics


def calculate_qini_risk(ite_estimates, w, t, y, outcome_pred=[], outcome_dml_pred=[], prop_prob= [], prop_score=[], case='dr_t_learner', save_dir=''):

    #TODO: Defining (t0, t1) assumes the treatment is binary
    t0= t*0
    t1= t*0 + 1

    data_size= w.shape[0]
    t= np.reshape(t, (data_size))
    y= np.reshape(y, (data_size))
    ite_estimates= np.reshape(ite_estimates, (data_size))

    if outcome_pred:
        mu_0, mu_1 = outcome_pred
        mu = mu_0 * (1 - t) + mu_1 * (t)

    #Compute ITE Approx
    if case in ['dr_s_score', 'dr_t_score']:
        ite_approx =(mu_1 - mu_0) + (2 * t - 1) * (y - mu) / prop_score

    elif case in ['tmle_s_score', 'tmle_t_score']:

        ite_approx_plugin= mu_1 - mu_0
        y_res= y - outcome_dml_pred
        t_res= t - prop_prob[:, 1]
        t_var= prop_prob[:, 1] * ( 1- prop_prob[:, 1] )
        clvr= t_res / t_var

        if os.path.isfile(save_dir + 'tmle_cf.p'):
            tmle_= pickle.load(open(save_dir + 'tmle_cf.p', "rb"))
        else:
            tmle_ = CausalForest(min_samples_leaf=20)
            tmle_.fit(w, clvr, y_res)
            pickle.dump(tmle_, open(save_dir + 'tmle_cf.p', "wb"))

        ite_approx = ite_approx_plugin + tmle_.predict(w).flatten() * clvr

    #Qini Score
    ugrid = np.linspace(1, 99, 50)
    qs = np.percentile(ite_estimates, ugrid)
    ate = np.zeros((len(qs), 3))
    true_ate = np.zeros(len(qs))
    psi = np.zeros((len(qs), ite_approx.shape[0]))
    n = len(ite_approx)

    all_ate = np.mean(ite_approx)
    for it in range(len(qs)):
        inds = (qs[it] <= ite_estimates)
        prob = np.mean(inds)
        psi[it, :] = (ite_approx - all_ate) * (inds - prob)
        ate[it, 0] = np.mean(psi[it])
        psi[it, :] -= ate[it, 0]
        ate[it, 1] = np.sqrt(np.mean(psi[it]**2) / n)
        ate[it, 2] = prob

    qini_psi = np.sum(psi[:-1] * np.diff(ugrid).reshape(-1, 1) / 100, 0)
    qini = np.sum(ate[:-1, 0] * np.diff(ugrid) / 100)
    qini_stderr = np.sqrt(np.mean(qini_psi**2) / n)

    out= {}
    out['qini_' + case] = qini - qini_stderr
    return out


def calculate_calibration_risk(ite_estimates, w, t, y, ite_estimates_train=[], outcome_pred=[], outcome_dml_pred=[], prop_prob= [], prop_score=[], case='dr_t_learner', save_dir=''):

    #TODO: Defining (t0, t1) assumes the treatment is binary
    t0= t*0
    t1= t*0 + 1

    data_size= w.shape[0]
    t= np.reshape(t, (data_size))
    y= np.reshape(y, (data_size))
    ite_estimates= np.reshape(ite_estimates, (data_size))
    ite_estimates_train= np.reshape(ite_estimates_train, (ite_estimates_train[0].shape[0]))
    if outcome_pred:
        mu_0, mu_1 = outcome_pred
        mu = mu_0 * (1 - t) + mu_1 * (t)

    #Compute ITE Approx
    if case in ['dr_s_score', 'dr_t_score']:
        ite_approx =(mu_1 - mu_0) + (2 * t - 1) * (y - mu) / prop_score

    elif case in ['tmle_s_score', 'tmle_t_score']:

        ite_approx_plugin= mu_1 - mu_0
        y_res= y - outcome_dml_pred
        t_res= t - prop_prob[:, 1]
        t_var= prop_prob[:, 1] * ( 1- prop_prob[:, 1] )
        clvr= t_res / t_var

        if os.path.isfile(save_dir + 'tmle_cf.p'):
            tmle_= pickle.load(open(save_dir + 'tmle_cf.p', "rb"))
        else:
            tmle_ = CausalForest(min_samples_leaf=20)
            tmle_.fit(w, clvr, y_res)
            pickle.dump(tmle_, open(save_dir + 'tmle_cf.p', "wb"))

        ite_approx = ite_approx_plugin + tmle_.predict(w).flatten() * clvr

    #Calibration Score
    ugrid = np.arange(0, 101, 25)
    qs = np.percentile(ite_estimates_train, ugrid)
    qs[-1] = np.inf
    qs[0] = -np.inf

    all_ate = np.mean(ite_approx)
    ate = np.zeros((len(qs) - 1, 3))
    for it in range(len(qs) - 1):
        inds = (qs[it] <= ite_estimates) & (ite_estimates < qs[it + 1])
        ate[it, 2] = np.mean(inds)
        if ate[it, 2] > 0:
            ate[it, 0] = np.mean(ite_approx[inds])
            ate[it, 1] = np.mean(ite_estimates[inds])

    cal_score = np.sum(ate[:, 2] * np.abs(ate[:, 0] - ate[:, 1]))

    out= {}
    out['cal_' + case] = 1 - cal_score
    return out


def calculate_r_risk(ite_estimates, w, t, y, outcome_pred= [], treatment_prob=[]):

    data_size = w.shape[0]
    t = np.reshape(t, (data_size))
    y = np.reshape(y, (data_size))
    ite_estimates = np.reshape(ite_estimates, (data_size))

    mu= outcome_pred
    # print(y.shape, mu.shape, t.shape, ite_estimates.shape, prop_score.shape)

    # R Score
    r_score= np.mean(( (y-mu) - ite_estimates*(t-treatment_prob)) ** 2)

    out = {}
    out['rscore'] = r_score

    return out

def calculate_x_risk(ite_estimates, w, t, y, outcome_pred= []):

    data_size= w.shape[0]
    t = np.reshape(t, (data_size))
    y = np.reshape(y, (data_size))
    ite_estimates = np.reshape(ite_estimates, (data_size))

    mu_0, mu_1= outcome_pred
    ite_approx=  t * ( y - mu_0 ) +  (1-t) * ( mu_1 - y )

    out= {}
    out['x_score'] = np.mean((ite_estimates - ite_approx) ** 2)

    return out

def calculate_value_risk(ite_estimates, w, t, y, dataset_name, prop_score=[]):
    # TODO: Defining (t0, t1) assumes the treatment is binary. Are we going to work with non binary treatments?

    t0 = t * 0
    t1 = t * 0 + 1

    data_size = w.shape[0]
    ite_estimates = np.reshape(ite_estimates, (data_size))
    t = np.reshape(t, (data_size))
    y = np.reshape(y, (data_size))

    #     print('W', w.shape)
    #     print('T', t.shape)
    #     print('Y', y.shape)
    #     print('Prop Score', prop_score.shape)
    #     print('ITE', ite_estimates.shape)

    # Decision policy: Recommend treatment based ITE. Check whether positive ITE is desirable for the datsaet or not
    if dataset_name not in ['twins']:
        decision_policy = 1 * (ite_estimates > 0)
    else:
        decision_policy = 1 * (ite_estimates < 0)

    decision_policy = np.reshape(decision_policy, (data_size))
    #     print('Decision Policy', decision_policy.shape)

    #     print(np.unique(t, return_counts=True))
    #     print(np.unique(decision_policy, return_counts=True))
    #     print(np.sum(t == decision_policy))

    indices = t == decision_policy
    weighted_outcome = y / prop_score

    #     print(np.sum(indices), data_size)
    #     print(np.mean(weighted_outcome[indices]), np.sum(weighted_outcome[indices])/np.sum(indices))
    #     print(np.mean(weighted_outcome), np.sum(weighted_outcome)/data_size)
    value_score = np.sum(weighted_outcome[indices]) / data_size
    if dataset_name not in ['twins']:
        value_score = -1*value_score

    out = {}
    out['value_score'] = value_score

    return out


def calculate_value_dr_risk(ite_estimates, w, t, y, dataset_name, outcome_pred=[], prop_score=[], min_propensity=0):
    # TODO: Defining (t0, t1) assumes the treatment is binary
    t0 = t * 0
    t1 = t * 0 + 1

    data_size = w.shape[0]
    t = np.reshape(t, (data_size))
    y = np.reshape(y, (data_size))
    ite_estimates = np.reshape(ite_estimates, (data_size))

    mu_0, mu_1 = outcome_pred
    mu = mu_0 * (1 - t) + mu_1 * (t)

    # Decision Policy: Recommend treatment based ITE. Check whether positive ITE is desirable for the datsaet or not
    if dataset_name not in ['twins']:
        decision_policy = 1 * (ite_estimates > 0)
    else:
        decision_policy = 1 * (ite_estimates < 0)
    decision_policy = np.reshape(decision_policy, (data_size))
    #     print('Decision Policy', decision_policy.shape)

    # Propensity clipping version
    if min_propensity:
        indices = prop_score < min_propensity
        prop_score[indices]= min_propensity

        indices= prop_score > 1 - min_propensity
        prop_score[indices]= 1-min_propensity

    # Value DR Score
    value_dr_score = decision_policy * (mu_1 - mu_0 + (2 * t - 1) * (y - mu) / prop_score)
    if dataset_name not in ['twins']:
        value_dr_score = -1*value_dr_score

    out={}
    if min_propensity:
        out['value_dr_clip_prop_score']= np.mean(value_dr_score)
    else:
        out['value_dr_score'] = np.mean(value_dr_score)

    # if min_propensity:
    #     indices= np.where(np.logical_and(prop_score >= min_propensity, prop_score <= 1-min_propensity))[0]
    #     out['value_dr_clip_prop_score']= np.mean(value_dr_score[indices])

    return out


def nearest_observed_counterfactual(x, X, Y):

    dist= np.sum((X-x)**2, axis=1)
    idx= np.argmin(dist)
    return Y[idx]

def calculate_tau_match_risk(ite_estimates, w, t, y, iptw=False, prop_score=[]):

    #TODO: Defining (t0, t1) assumes the treatment is binary
    t0= t*0
    t1= t*0 + 1

    data_size= w.shape[0]
    t= np.reshape(t, (data_size))
    y= np.reshape(y, (data_size))
    ite_estimates= np.reshape(ite_estimates, (data_size))

    X0= w[t==0, :]
    X1= w[t==1, :]
    Y0= y[t==0]
    Y1= y[t==1]

    match_estimates_ite= np.zeros(data_size)

    cf_y=np.zeros(data_size)
    for idx in range(data_size):
        curr_x= w[idx]
        curr_t= t[idx]
        curr_y= y[idx]
        #Approximating counterfactual by mathching
        if curr_t == 1:
            cf_y[idx] = nearest_observed_counterfactual(curr_x, X0, Y0)
        elif curr_t == 0:
            cf_y[idx] = nearest_observed_counterfactual(curr_x, X1, Y1)

    match_estimates_ite= (2*t -1)*(y - cf_y)

    tau_score= np.mean((ite_estimates - match_estimates_ite)**2)

    out={}
    out['tau_match_score']= tau_score

    return out


def calculate_all_tau_risk(ite_estimates, w, t, y, outcome_pred=[], outcome_dml_pred=[], prop_prob= [], prop_score=[], case='s_learner', inv_prop_threshold= 0, save_dir=''):

    #TODO: Defining (t0, t1) assumes the treatment is binary
    t0= t*0
    t1= t*0 + 1

    data_size= w.shape[0]
    t= np.reshape(t, (data_size))
    y= np.reshape(y, (data_size))
    ite_estimates= np.reshape(ite_estimates, (data_size))

    if outcome_pred:
        mu_0, mu_1 = outcome_pred
        mu = mu_0 * (1 - t) + mu_1 * (t)

    #Propensity Clipping
    if case in ['iptw_clip_score', 'dr_s_clip_score', 'dr_t_clip_score'] and inv_prop_threshold:
        indices = prop_score < 1/inv_prop_threshold
        prop_score[indices]= 1/inv_prop_threshold

        indices= prop_score > 1 - 1/inv_prop_threshold
        prop_score[indices]= 1 - 1/inv_prop_threshold

    if case in ['tmle_s_score', 'tmle_t_score'] and inv_prop_threshold:
        indices = prop_prob[:, 1] < 1/inv_prop_threshold
        prop_prob[indices, 1]= 1/inv_prop_threshold

        indices= prop_prob[:, 1] > 1 - 1/inv_prop_threshold
        prop_prob[indices, 1]= 1 - 1/inv_prop_threshold

    if case in ['s_score', 't_score']:
        ite_approx = mu_1 - mu_0

    elif case in ['iptw_score', 'iptw_clip_score']:
        ite_approx = (2 * t - 1) * (y) / prop_score

    elif case in ['dr_s_score', 'dr_t_score', 'dr_s_clip_score', 'dr_t_clip_score']:
        ite_approx =(mu_1 - mu_0) + (2 * t - 1) * (y - mu) / prop_score

    elif case in ['tmle_s_score', 'tmle_t_score']:

        # #NOTE: Debug
        # ite_approx= mu_1 - mu_0

        ite_approx_plugin= mu_1 - mu_0
        y_res= y - outcome_dml_pred
        t_res= t - prop_prob[:, 1]
        t_var= prop_prob[:, 1] * ( 1- prop_prob[:, 1] )
        clvr= t_res / t_var

        if os.path.isfile(save_dir + 'tmle_cf.p'):
            tmle_= pickle.load(open(save_dir + 'tmle_cf.p', "rb"))
        else:
            tmle_ = CausalForest(min_samples_leaf=20)
            tmle_.fit(w, clvr, y_res)
            pickle.dump(tmle_, open(save_dir + 'tmle_cf.p', "wb"))

        ite_approx = ite_approx_plugin + tmle_.predict(w).flatten() * clvr

    elif 'switch' in case:
        ite_approx_plugin= mu_1 - mu_0

        if 'iptw' in case:
            ite_approx_robust= (2 * t - 1) * (y) / prop_score
        elif 'dr' in case:
            ite_approx_robust = (mu_1 - mu_0) + (2 * t - 1) * (y - mu) / prop_score

        indices= 1*(prop_score < 1/inv_prop_threshold)
        ite_approx= ite_approx_plugin * indices  + ite_approx_robust * (1-indices)

    elif 'cab' in case:
        ite_approx_plugin= mu_1 - mu_0

        if 'iptw' in case:
            ite_approx_robust= (2 * t - 1) * (y) / prop_score
        elif 'dr' in case:
            ite_approx_robust = (mu_1 - mu_0) + (2 * t - 1) * (y - mu) / prop_score

        alpha= prop_score * inv_prop_threshold
        indices = 1 * (prop_score < 1 / inv_prop_threshold)
        ite_approx= ( alpha * ite_approx_robust + (1-alpha) * ite_approx_plugin ) * indices  + ite_approx_robust * (1-indices)

    # Log the score
    out={}
    out['tau_' + case]= np.mean((ite_estimates - ite_approx) ** 2)

    return out

def calculate_influence_risk(ite_estimates, w, t, y, outcome_pred=[], prop_prob=[], min_propensity=0):

    #TODO: Defining (t0, t1) assumes the treatment is binary
    t0= t*0
    t1= t*0 + 1

    data_size= w.shape[0]
    t= np.reshape(t, (data_size))
    y= np.reshape(y, (data_size))
    ite_estimates= np.reshape(ite_estimates, (data_size))

    mu_0, mu_1= outcome_pred
    mu= mu_0 * (1-t) + mu_1 * (t)
    prop_score = prop_prob[:, 0] * (1 - t) + prop_prob[:, 1] * (t)

    t_learner_ite= mu_1 - mu_0
    plug_in_estimate= t_learner_ite - ite_estimates
    A= t - prop_prob[:, 1]
    C= prop_prob[:, 0] * prop_prob[:, 1]
    B= 2*t*(t- prop_prob[:, 1])*(1/C)

    #Influence Score
    influence_score = (1 - B) * (t_learner_ite ** 2) + B * y * plug_in_estimate - A * (plug_in_estimate ** 2) + ite_estimates ** 2

    out= {}
    out['influence_score'] = np.mean(influence_score)

    #Propesnity clipping version
    if min_propensity:
        indices= np.where(np.logical_and(prop_score >= min_propensity, prop_score <= 1-min_propensity))[0]
        out['influence_clip_prop_score']= np.mean(influence_score[indices])

    return out


def calculate_ite_metrics(ite: np.ndarray, fitted_estimators: List[BaseIteEstimator], w, t):
    
    #TODO: Defining (t0, t1) assumes the treatment is binary  
    t0= t*0
    t1= t*0 + 1
    
    ite_estimates = np.stack([fitted_estimator.effect(w, t0, t1) for
                              fitted_estimator in fitted_estimators],
                             axis=STACK_AXIS)

    # Calculated for each unit/individual, this is the a vector of num units
    mean_ite_estimate = ite_estimates.mean(axis=STACK_AXIS)
    ite_bias = mean_ite_estimate - ite
    ite_abs_bias = np.abs(ite_bias)
    ite_squared_bias = ite_bias**2
    ite_variance = calc_vector_variance(ite_estimates, mean_ite_estimate)
    ite_std_error = np.sqrt(ite_variance)
    ite_mse = calc_vector_mse(ite_estimates, ite)
    ite_rmse = np.sqrt(ite_mse)

    # Calculated for a single dataset, so this is a vector of num datasets
    pehe_squared = calc_vector_mse(ite_estimates, ite, reduce_axis=(1 - STACK_AXIS))
    pehe = np.sqrt(pehe_squared)

    #True Variance of ITE estimates
    true_ite_var= [np.var(ite)]

#     #Standard Scale
#     pehe_vec= np.sqrt((ite_estimates - ite)**2)
#     standard_scale= [(pehe_vec - np.mean(pehe_vec))/np.sqrt(np.var(pehe_vec))]

    # TODO: ITE coverage
    # ate_coverage = calc_coverage(ate_conf_ints, ate)
    # ate_mean_int_length = calc_mean_interval_length(ate_conf_ints)

    return {
        'ite_bias': ite_bias,
        # 'ite_abs_bias': ite_abs_bias,
        # 'ite_squared_bias': ite_squared_bias,
        'ite_variance': ite_variance,
        'ite_std_error': ite_std_error,
#         'ite_mse': ite_mse,
#         'ite_rmse': ite_rmse,
        # 'ite_coverage': ite_coverage,
        # 'ite_mean_int_length': ite_mean_int_length,
        'true_ite_var': true_ite_var,
        'pehe_squared': pehe_squared,
        'pehe': pehe,
    }


def calc_variance(estimates, mean_estimate):
    return calc_mse(estimates, mean_estimate)


def calc_mse(estimates, target):
    if isinstance(estimates, (list, tuple)):
        estimates = np.array(estimates)
    return ((estimates - target) ** 2).mean()


def calc_coverage(intervals: List[tuple], estimand):
    n_covers = sum(1 for interval in intervals if interval[0] <= estimand <= interval[1])
    return n_covers / len(intervals)


def calc_mean_interval_length(intervals: List[tuple]):
    return mean(interval[1] - interval[0] for interval in intervals)


def calc_vector_variance(estimates: np.ndarray, mean_estimate: np.ndarray):
    return calc_vector_mse(estimates, mean_estimate)


def calc_vector_mse(estimates: np.ndarray, target: np.ndarray, reduce_axis=STACK_AXIS):
    assert isinstance(estimates, np.ndarray) and estimates.ndim == 2
    assert isinstance(target, np.ndarray) and target.ndim == 1
    assert target.shape[0] == estimates.shape[1 - STACK_AXIS]

    n_seeds = estimates.shape[STACK_AXIS]
    target = np.expand_dims(target, axis=STACK_AXIS).repeat(n_seeds, axis=STACK_AXIS)
    return ((estimates - target) ** 2).mean(axis=reduce_axis)
