from typing import List
from statistics import mean
from math import sqrt
import sys
import os
import pickle
import numpy as np
from typing import Tuple

from econml.grf import CausalForest

def get_nuisance_propensity_pred(w, t, save_dir='') -> Tuple[np.ndarray, np.ndarray]:
    """ Compute predictions of propensity model seleted by AutoML for metrics on evaluation set

    Inputs:
        w: Covariates; Expected Shape: (n, d)
        t: Treatment; Expected Shape: (n, 1)
        save_dir: Directory containing the nuisance models based on AutoML selection
    
    Returns:
        pred_prob: Predicted probabilities of treatment; Shape: (n, 2)
        prop_score: Propensity scores; Shape: (n)
    """

    if os.path.isfile(save_dir + 'prop' + '.p'):
        data_size= w.shape[0]
        t= np.reshape(t, (data_size))
        
        prop_model= pickle.load( open(save_dir + 'prop' + '.p', "rb") )                  
        pred_prob= prop_model.predict_proba(w)
        prop_score= pred_prob[:, 0] * (1-t) + pred_prob[:, 1] * (t)    
    else:        
        print('Error: Nuisance model not trained for evaluation purposes on this dataset')
    
    return pred_prob, prop_score
    
def get_nuisance_outcome_t_pred(w, t, save_dir='') -> Tuple[np.ndarray, np.ndarray]:
    """ Compute predictions of T Learner outcome model seleted by AutoML for metrics on evaluation set
    
    Inputs:
        w: Covariates; Expected Shape: (n, d)
        t: Treatment; Expected Shape: (n, 1)
        save_dir: Directory containing the nuisance models based on AutoML selection
    
    Returns:
        mu_0: Predicted outcomes for t=0; Shape: (n)
        mu_1: Predicted outcomes for t=1; Shape: (n)
    """

    if os.path.isfile(save_dir + 't_learner_0' + '.p') and os.path.isfile(save_dir + 't_learner_1' + '.p'):        
        data_size= w.shape[0]
        t= np.reshape(t, (data_size))

        out_model_0= pickle.load( open(save_dir + 't_learner_0' + '.p', "rb") )
        out_model_1= pickle.load( open(save_dir + 't_learner_1' + '.p', "rb") )
        mu_0= out_model_0.predict(w)
        mu_1= out_model_1.predict(w)    
    else:
        print('Error: Nuisance model not trained for evaluation purposes on this dataset')        
        
    return (mu_0, mu_1)

def get_nuisance_outome_s_pred(w, t, save_dir='') -> Tuple[np.ndarray, np.ndarray]:
    """ Compute predictions of S Learner outcome model seleted by AutoML for metrics on evaluation set
    
    Inputs:
        w: Covariates; Expected Shape: (n, d)
        t: Treatment; Expected Shape: (n, 1)
        save_dir: Directory containing the nuisance models based on AutoML selection
    
    Returns:
        mu_0: Predicted outcomes for t=0; Shape: (n)
        mu_1: Predicted outcomes for t=1; Shape: (n)
    """

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


def get_nuisance_outcome_dml_pred(w, save_dir='') -> np.ndarray:
    """ Compute predictions of R Learner outcome model seleted by AutoML for metrics on evaluation set
    
    Inputs:
        w: Covariates; Expected Shape: (n, d)
        save_dir: Directory containing the nuisance models based on AutoML selection
    
    Returns:
        mu_dml: Predicted outcomes; Shape: (n)
    """    

    if os.path.isfile(save_dir + 'dml' + '.p'):
        out_model = pickle.load(open(save_dir + 'dml' + '.p', "rb"))
        mu_dml= out_model.predict(w)
    else:
        print('Error: Nuisance model not trained for evaluation purposes on this dataset')

    return mu_dml

def calculate_metrics(
                    eval_w: np.ndarray,
                    eval_t: np.ndarray,
                    eval_y: np.ndarray,
                    prop_prob: np.ndarray,
                    prop_score: np.ndarray,
                    outcome_s_pred: Tuple[np.ndarray, np.ndarray],
                    outcome_t_pred: Tuple[np.ndarray, np.ndarray],
                    outcome_dml_pred: np.ndarray,
                    ite_estimates: np.ndarray,
                    ite_estimates_train: np.ndarray,
                    score: str,
                    dataset_name: str,
                    nuisance_stats_dir: str,
                    inv_prop_threshold: float= 10
                ) -> dict:    
    """Compute the evaluation metrics (score)

    Inputs:
        eval_w: Covariates; Expected Shape: (n, d)
        eval_t: Treatment; Expected Shape: (n, 1)
        eval_y: Outcome; Expected Shape: (n, 1)
        prop_prob: Predicted probabilities of treatment; Shape: (n, 2)
        prop_score: Propensity scores; Shape: (n)
        outcome_s_pred: Predicted outcomes for S Learner; Shape: [(n), (n)]
        outcome_t_pred: Predicted outcomes for T Learner; Shape: [(n), (n)]
        outcome_dml_pred: Predicted outcomes for R Learner; Shape: (n)
        ite_estimates: ITE estimates; Shape: (n)
        ite_estimates_train: ITE estimates on training set; Shape: (n)
        score: Evaluation metric to compute
        dataset_name: Name of the dataset
        nuisance_stats_dir: Directory containing the nuisance models based on AutoML selection
        inv_prop_threshold: Threshold for propensity clipping
    
    Returns:
        Dictionary containing the evaluation metric
    """
    
    if score == 'value_score':
        # Compute Value Risk
        return calculate_value_risk(
                                    ite_estimates= ite_estimates, 
                                    w= eval_w, 
                                    t= eval_t, 
                                    y= eval_y, 
                                    dataset_name= dataset_name, 
                                    prop_score= prop_score
                                )    
    elif score == 'value_dr_score':
        #Compute Value DR Risk
        return calculate_value_dr_risk(
                                        ite_estimates= ite_estimates, 
                                        w= eval_w, 
                                        t= eval_t, 
                                        y= eval_y, 
                                        dataset_name= dataset_name, 
                                        outcome_pred= outcome_t_pred, 
                                        prop_score= prop_score
                                    )
    elif score == 'value_dr_clip_score':
        #Compute Value DR Risk with propensity clipping
        return calculate_value_dr_risk(
                                        ite_estimates= ite_estimates, 
                                        w= eval_w, 
                                        t= eval_t, 
                                        y= eval_y, 
                                        dataset_name= dataset_name, 
                                        outcome_pred= outcome_t_pred, 
                                        prop_score= prop_score, 
                                        inv_prop_threshold= inv_prop_threshold
                                    )
    elif score == 'r_score':
        #Compute RScore Metric
        return calculate_r_risk(
                                    ite_estimates= ite_estimates, 
                                    w= eval_w,
                                    t= eval_t, 
                                    y= eval_y, 
                                    outcome_dml_pred= outcome_dml_pred, 
                                    treatment_prob=prop_prob[:, 1]
                                )
    elif score == 'x_score':
        #Compute X Learner Metric
        return calculate_x_risk(
                                    ite_estimates= ite_estimates, 
                                    w= eval_w,
                                    t= eval_t, 
                                    y= eval_y,
                                    outcome_pred= outcome_t_pred
                                )
    elif score == 'tau_match_score':
        #Compute Tau Matching Risk
        return calculate_tau_match_risk(
                                            ite_estimates= ite_estimates, 
                                            w= eval_w, 
                                            t= eval_t, 
                                            y= eval_y                                          
                                        )
    elif score == 'tau_s_score':
        # Compute Plug In Tau Score from Van der Schaar paper using S-Learner
        return calculate_all_tau_risk(
                                        ite_estimates= ite_estimates,
                                        w= eval_w, 
                                        t= eval_t, 
                                        y= eval_y, 
                                        outcome_pred= outcome_s_pred,  
                                        prop_score= prop_score, 
                                        case='s_score'
                                    )    
    elif score == 'tau_t_score':
        # Compute Plug In Tau Score from Van der Schaar paper using T-Learner
        return calculate_all_tau_risk(
                                        ite_estimates= ite_estimates,
                                        w= eval_w, 
                                        t= eval_t, 
                                        y= eval_y, 
                                        outcome_pred= outcome_t_pred, 
                                        prop_score= prop_score, 
                                        case='t_score'
                                    )
    elif score == 'tau_iptw_score':
        # Compute IPTW Score
        return calculate_all_tau_risk(
                                        ite_estimates= ite_estimates, 
                                        w= eval_w, 
                                        t= eval_t, 
                                        y= eval_y, 
                                        prop_score= prop_score, 
                                        case='iptw_score'
                                    )
    elif score == 'tau_iptw_clip_score':
        # Compute IPTW Score with propensity clipping
        return calculate_all_tau_risk(
                                        ite_estimates= ite_estimates,
                                        w= eval_w, 
                                        t= eval_t, 
                                        y= eval_y, 
                                        prop_score= prop_score, 
                                        case='iptw_clip_score', 
                                        inv_prop_threshold= inv_prop_threshold
                                    )
    elif score == 'tau_switch_iptw_s_score':   
        # Compute Switch IPTW Score with S Learner
        return calculate_all_tau_risk(
                                        ite_estimates= ite_estimates,
                                        w= eval_w, 
                                        t= eval_t, 
                                        y= eval_y, 
                                        outcome_pred= outcome_s_pred, 
                                        prop_score= prop_score, 
                                        case='switch_iptw_s_score',  
                                        inv_prop_threshold= inv_prop_threshold
                                    )
    elif score == 'tau_switch_iptw_t_score':
        # Compute Switch IPTW Score with T Learner
        return calculate_all_tau_risk(
                                        ite_estimates= ite_estimates,
                                        w= eval_w, 
                                        t= eval_t, 
                                        y= eval_y, 
                                        outcome_pred= outcome_t_pred, 
                                        prop_score= prop_score, 
                                        case='switch_iptw_t_score',  
                                        inv_prop_threshold= inv_prop_threshold
                                    )
    elif score == 'tau_cab_iptw_s_score':
       # Compute CAB IPTW Score with S Learner
        return calculate_all_tau_risk(
                                        ite_estimates= ite_estimates,
                                        w= eval_w, 
                                        t= eval_t, 
                                        y= eval_y, 
                                        outcome_pred= outcome_s_pred, 
                                        prop_score= prop_score, 
                                        case='cab_iptw_s_score',  
                                        inv_prop_threshold= inv_prop_threshold
                                    )
    elif score == 'tau_cab_iptw_t_score':  
        # Compute CAB IPTW Score with T Learner
        return calculate_all_tau_risk(
                                        ite_estimates= ite_estimates,
                                        w= eval_w, 
                                        t= eval_t, 
                                        y= eval_y, 
                                        outcome_pred= outcome_t_pred, 
                                        prop_score= prop_score, 
                                        case='cab_iptw_t_score',  
                                        inv_prop_threshold= inv_prop_threshold
                                    )
    elif score == 'tau_dr_s_score':
        # Compute DR Score with outcome function as S Learner
        return calculate_all_tau_risk(
                                        ite_estimates= ite_estimates, 
                                        w= eval_w, 
                                        t= eval_t, 
                                        y= eval_y, 
                                        outcome_pred= outcome_s_pred, 
                                        prop_score= prop_score, 
                                        case='dr_s_score'
                                    )
    elif score == 'tau_dr_s_clip_score':
        # Compute DR Score with propensity clipping, outcome function as S Learner
        return calculate_all_tau_risk(
                                        ite_estimates= ite_estimates,
                                        w= eval_w, 
                                        t= eval_t, 
                                        y= eval_y, 
                                        outcome_pred= outcome_s_pred, 
                                        prop_score= prop_score, 
                                        case='dr_s_clip_score', 
                                        inv_prop_threshold= inv_prop_threshold
                                    )
    elif score == 'tau_dr_t_score':
        # Compute DR Score with outcome function as T Learner
        return calculate_all_tau_risk(
                                        ite_estimates= ite_estimates,
                                        w= eval_w, 
                                        t= eval_t, 
                                        y= eval_y, 
                                        outcome_pred= outcome_t_pred, 
                                        prop_score= prop_score, 
                                        case='dr_t_score'
                                    )
    elif score == 'tau_dr_t_clip_score':
        # Compute DR Score with propensity clipping, outcome function as T Learner
        return calculate_all_tau_risk(
                                        ite_estimates= ite_estimates,
                                        w= eval_w, 
                                        t= eval_t, 
                                        y= eval_y, 
                                        outcome_pred= outcome_t_pred, 
                                        prop_score= prop_score, 
                                        case='dr_t_clip_score', 
                                        inv_prop_threshold= inv_prop_threshold
                                    )
    elif score == 'tau_switch_dr_s_score':
        # Compute Switch DR Score with S Learner
        return calculate_all_tau_risk(
                                        ite_estimates= ite_estimates,
                                        w= eval_w, 
                                        t= eval_t, 
                                        y= eval_y, 
                                        outcome_pred= outcome_s_pred, 
                                        prop_score= prop_score, 
                                        case='switch_dr_s_score',  
                                        inv_prop_threshold= inv_prop_threshold
                                    )
    elif score == 'tau_switch_dr_t_score':
        # Compute Switch DR Score with T Learner
        return calculate_all_tau_risk(
                                        ite_estimates= ite_estimates,
                                        w= eval_w, 
                                        t= eval_t, 
                                        y= eval_y, 
                                        outcome_pred= outcome_t_pred, 
                                        prop_score= prop_score, 
                                        case='switch_dr_t_score',  
                                        inv_prop_threshold= inv_prop_threshold
                                    )
    elif score == 'tau_cab_dr_s_score':
        # Compute CAB DR Score with S Learner
        return calculate_all_tau_risk(
                                        ite_estimates= ite_estimates,
                                        w= eval_w, 
                                        t= eval_t, 
                                        y= eval_y, 
                                        outcome_pred= outcome_s_pred, 
                                        prop_score= prop_score, 
                                        case='cab_dr_s_score',  
                                        inv_prop_threshold= inv_prop_threshold
                                    )
    elif score == 'tau_cab_dr_t_score':
        # Compute CAB DR Score with T Learner
        return calculate_all_tau_risk(
                                        ite_estimates= ite_estimates,
                                        w= eval_w, 
                                        t= eval_t, 
                                        y= eval_y, 
                                        outcome_pred= outcome_t_pred, 
                                        prop_score= prop_score, 
                                        case='cab_dr_t_score',  
                                        inv_prop_threshold= inv_prop_threshold
                                    )
    elif score == 'tau_tmle_s_score':
       #Compute TMLE score with S Learner
        return calculate_all_tau_risk(
                                        ite_estimates= ite_estimates,
                                        w= eval_w, 
                                        t= eval_t, 
                                        y= eval_y, 
                                        outcome_pred= outcome_s_pred, 
                                        outcome_dml_pred= outcome_dml_pred, 
                                        prop_prob= prop_prob, 
                                        prop_score= prop_score, 
                                        case='tmle_s_score',  
                                        inv_prop_threshold= inv_prop_threshold, 
                                        save_dir= nuisance_stats_dir
                                    )
    elif score == 'tau_tmle_t_score':
        #Compute TMLE score with T Learner
        return calculate_all_tau_risk(
                                        ite_estimates= ite_estimates,
                                        w= eval_w, 
                                        t= eval_t, 
                                        y= eval_y, 
                                        outcome_pred= outcome_t_pred, 
                                        outcome_dml_pred= outcome_dml_pred, 
                                        prop_prob= prop_prob, 
                                        prop_score= prop_score, 
                                        case='tmle_t_score',  
                                        inv_prop_threshold= inv_prop_threshold, 
                                        save_dir= nuisance_stats_dir
                                    )
    elif score == 'cal_dr_s_score':
        #Compute Calibration Score with DR S Learner as base
        return calculate_calibration_risk(
                                            ite_estimates= ite_estimates, 
                                            w= eval_w, 
                                            t= eval_t, 
                                            y= eval_y, 
                                            ite_estimates_train=ite_estimates_train, 
                                            outcome_pred=outcome_s_pred, 
                                            prop_score= prop_score, 
                                            case='dr_s_score'
                                        )
    elif score == 'cal_dr_t_score':
        #Compute Calibration Score with DR T Learner as base
        return calculate_calibration_risk(
                                            ite_estimates= ite_estimates,
                                            w= eval_w, 
                                            t= eval_t, 
                                            y= eval_y, 
                                            ite_estimates_train= ite_estimates_train, 
                                            outcome_pred= outcome_t_pred, 
                                            prop_score= prop_score, 
                                            case='dr_t_score'
                                        )
    elif score == 'cal_tmle_s_score':
        #Compute Calibration Score with TMLE S Learner as base
        return calculate_calibration_risk(
                                            ite_estimates= ite_estimates,
                                            w= eval_w, 
                                            t= eval_t, 
                                            y= eval_y, 
                                            ite_estimates_train= ite_estimates_train, 
                                            outcome_pred= outcome_s_pred, 
                                            outcome_dml_pred= outcome_dml_pred, 
                                            prop_prob= prop_prob, 
                                            case='tmle_s_score', 
                                            save_dir= nuisance_stats_dir
                                        )
    elif score == 'cal_tmle_t_score':
        #Compute Calibration Score with TMLE T Learner as base
        return calculate_calibration_risk(
                                            ite_estimates= ite_estimates,
                                            w= eval_w, 
                                            t= eval_t, 
                                            y= eval_y, 
                                            ite_estimates_train= ite_estimates_train, 
                                            outcome_pred= outcome_t_pred, 
                                            outcome_dml_pred= outcome_dml_pred, 
                                            prop_prob= prop_prob,
                                            case='tmle_t_score',
                                            save_dir= nuisance_stats_dir
                                        )
    elif score == 'qini_dr_s_score':
        #Compute Qini Score with DR S Learner as base
        return calculate_qini_risk(
                                    ite_estimates= ite_estimates, 
                                    w= eval_w, 
                                    t= eval_t, 
                                    y= eval_y, 
                                    outcome_pred=outcome_s_pred, 
                                    prop_score= prop_score, 
                                    case='dr_s_score'
                                )
    elif score == 'qini_dr_t_score':
        #Compute Qini Score with DR T Learner as base
        return calculate_qini_risk(
                                    ite_estimates= ite_estimates,
                                    w= eval_w, 
                                    t= eval_t, 
                                    y= eval_y, 
                                    outcome_pred= outcome_t_pred, 
                                    prop_score= prop_score, 
                                    case='dr_t_score'
                                )
    elif score == 'qini_tmle_s_score':
        #Compute Qini Score with TMLE S Learner as base
        return calculate_qini_risk(
                                    ite_estimates= ite_estimates,
                                    w= eval_w, 
                                    t= eval_t, 
                                    y= eval_y, 
                                    outcome_pred= outcome_s_pred, 
                                    outcome_dml_pred= outcome_dml_pred, 
                                    prop_prob= prop_prob, 
                                    case='tmle_s_score', 
                                    save_dir= nuisance_stats_dir
                                )
    elif score == 'qini_tmle_t_score':
        #Compute Qini Score with TMLE T Learner as base
        return calculate_qini_risk(
                                    ite_estimates= ite_estimates,
                                    w= eval_w, 
                                    t= eval_t, 
                                    y= eval_y, 
                                    outcome_pred= outcome_t_pred, 
                                    outcome_dml_pred= outcome_dml_pred, 
                                    prop_prob= prop_prob, 
                                    case='tmle_t_score', 
                                    save_dir= nuisance_stats_dir
                                )
    elif score == 'influence_score':
        #Compute Van de Schaar Influence function
        return calculate_influence_risk(
                                        ite_estimates= ite_estimates, 
                                        w= eval_w, 
                                        t= eval_t, 
                                        y= eval_y, 
                                        outcome_pred= outcome_t_pred, 
                                        prop_prob= prop_prob, 
                                    )
    elif score == 'influence_clip_score':
        #Compute Van de Schaar Influence function with propensity clipping
        return calculate_influence_risk(
                                        ite_estimates= ite_estimates, 
                                        w= eval_w, 
                                        t= eval_t, 
                                        y= eval_y, 
                                        outcome_pred= outcome_t_pred, 
                                        prop_prob= prop_prob, 
                                        inv_prop_threshold= inv_prop_threshold
                                    )
    else:
        NotImplementedError("Score not supported")


def calculate_all_tau_risk(
                            ite_estimates: np.ndarray, 
                            w: np.ndarray, 
                            t: np.ndarray, 
                            y: np.ndarray, 
                            outcome_pred:  Tuple[np.ndarray, np.ndarray]= [], 
                            outcome_dml_pred:  np.ndarray= [], 
                            prop_prob: np.ndarray= [], 
                            prop_score: np.ndarray= [], 
                            case: str= '',
                            save_dir: str= '',
                            inv_prop_threshold: float= 0, 
                        ) -> dict:
    """
    Inputs:
        ite_estimates: ITE estimates; Shape: (n)
        w: Covariates; Expected Shape: (n, d)
        t: Treatment; Expected Shape: (n, 1)
        y: Outcome; Expected Shape: (n, 1)
        outcome_pred: Predicted outcomes for S and T Learner; Shape: [(n), (n)]
        outcome_dml_pred: Predicted outcomes for R Learner; Shape: (n)
        prop_prob: Predicted probabilities of treatment; Shape: (n, 2)
        prop_score: Propensity scores; Shape: (n)
        case: Evaluation metric to compute
        inv_prop_threshold: Threshold for propensity clipping
        save_dir: Directory containing the nuisance models based on AutoML selection
    """

    data_size= w.shape[0]
    t= np.reshape(t, (data_size))
    y= np.reshape(y, (data_size))

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

def calculate_calibration_risk(
                                ite_estimates: np.ndarray, 
                                w: np.ndarray, 
                                t: np.ndarray, 
                                y: np.ndarray, 
                                ite_estimates_train: np.ndarray, 
                                outcome_pred: np.ndarray, 
                                outcome_dml_pred: np.ndarray= [], 
                                prop_prob: np.ndarray= [], 
                                prop_score: np.ndarray =[],
                                case='', 
                                save_dir=''
                            ) -> dict:
    """
    Inputs:
        ite_estimates: ITE estimates; Shape: (n)
        w: Covariates; Expected Shape: (n, d)
        t: Treatment; Expected Shape: (n, 1)
        y: Outcome; Expected Shape: (n, 1)
        ite_estimates_train: ITE estimates on training dataset; Shape: (n)
        outcome_pred: Predicted outcomes for S and T Learner; Shape: [(n), (n)]
        outcome_dml_pred: Predicted outcomes for R Learner; Shape: (n)
        prop_prob: Predicted probabilities of treatment; Shape: (n, 2)
        prop_score: Propensity scores; Shape: (n)
        case: Evaluation metric to compute
        save_dir: Directory containing the nuisance models based on AutoML selection
    """

    data_size= w.shape[0]
    t= np.reshape(t, (data_size))
    y= np.reshape(y, (data_size))

    mu_0, mu_1 = outcome_pred

    #Compute ITE Approx
    if case in ['dr_s_score', 'dr_t_score']:
        mu = mu_0 * (1 - t) + mu_1 * (t)
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

def calculate_qini_risk(
                        ite_estimates: np.ndarray, 
                        w: np.ndarray, 
                        t: np.ndarray,
                        y: np.ndarray, 
                        outcome_pred: np.ndarray, 
                        outcome_dml_pred: np.ndarray=[], 
                        prop_prob: np.ndarray= [], 
                        prop_score: np.ndarray= [], 
                        case='', 
                        save_dir=''
                    ) -> dict:
    """
    Inputs:
        ite_estimates: ITE estimates; Shape: (n)
        w: Covariates; Expected Shape: (n, d)
        t: Treatment; Expected Shape: (n, 1)
        y: Outcome; Expected Shape: (n, 1)
        outcome_pred: Predicted outcomes for S and T Learner; Shape: [(n), (n)]
        outcome_dml_pred: Predicted outcomes for R Learner; Shape: (n)
        prop_prob: Predicted probabilities of treatment; Shape: (n, 2)
        prop_score: Propensity scores; Shape: (n)
        case: Evaluation metric to compute
        save_dir: Directory containing the nuisance models based on AutoML selection
    """

    data_size= w.shape[0]
    t= np.reshape(t, (data_size))
    y= np.reshape(y, (data_size))

    mu_0, mu_1 = outcome_pred

    #Compute ITE Approx
    if case in ['dr_s_score', 'dr_t_score']:
        mu = mu_0 * (1 - t) + mu_1 * (t)
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

def calculate_x_risk(
                        ite_estimates: np.ndarray, 
                        w: np.ndarray, 
                        t: np.ndarray, 
                        y: np.ndarray, 
                        outcome_pred:  Tuple[np.ndarray, np.ndarray]
                    ) -> dict:
    """
    Inputs:
        ite_estimates: ITE estimates; Shape: (n)
        w: Covariates; Expected Shape: (n, d)
        t: Treatment; Expected Shape: (n, 1)
        y: Outcome; Expected Shape: (n, 1)
        outcome_pred: Predicted outcomes for S and T Learner; Shape: [(n), (n)]
    """

    data_size= w.shape[0]
    t = np.reshape(t, (data_size))
    y = np.reshape(y, (data_size))

    mu_0, mu_1= outcome_pred
    ite_approx=  t * ( y - mu_0 ) +  (1-t) * ( mu_1 - y )

    x_score = np.mean((ite_estimates - ite_approx) ** 2)

    return {'x_score': x_score}

def calculate_r_risk(
                        ite_estimates: np.ndarray, 
                        w: np.ndarray, 
                        t: np.ndarray, 
                        y: np.ndarray, 
                        outcome_dml_pred: np.ndarray, 
                        treatment_prob: np.ndarray
                    ) -> dict:
    """
    Inputs:
        ite_estimates: ITE estimates; Shape: (n)
        w: Covariates; Expected Shape: (n, d)
        t: Treatment; Expected Shape: (n, 1)
        y: Outcome; Expected Shape: (n, 1)
        outcome_dml_pred: Predicted outcomes for DML Learner; (n)
        treatment_prob: Probability of T=1; Shape: (n)
    """

    data_size = w.shape[0]
    t = np.reshape(t, (data_size))
    y = np.reshape(y, (data_size))

    # R Score
    mu= outcome_dml_pred
    r_score= np.mean(( (y-mu) - ite_estimates*(t-treatment_prob)) ** 2)

    return {'r_score': r_score}


def nearest_observed_counterfactual(x, X, Y):
    dist= np.sum((X-x)**2, axis=1)
    idx= np.argmin(dist)
    return Y[idx]

def calculate_tau_match_risk(
                                ite_estimates: np.ndarray,
                                w: np.ndarray, 
                                t: np.ndarray, 
                                y: np.ndarray, 
                            ) -> dict:
    """
    Inputs:
        ite_estimates: ITE estimates; Shape: (n)
        w: Covariates; Expected Shape: (n, d)
        t: Treatment; Expected Shape: (n, 1)
        y: Outcome; Expected Shape: (n, 1)
    """

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
    match_score= np.mean((ite_estimates - match_estimates_ite)**2)

    return {'tau_match_score': match_score}

def calculate_value_risk(
                            ite_estimates: np.ndarray, 
                            w: np.ndarray, 
                            t: np.ndarray, 
                            y: np.ndarray, 
                            dataset_name: str, 
                            prop_score: np.ndarray
                        ) -> dict:
    """
    Inputs:
        ite_estimates: ITE estimates; Shape: (n)
        w: Covariates; Expected Shape: (n, d)
        t: Treatment; Expected Shape: (n, 1)
        y: Outcome; Expected Shape: (n, 1)
        dataset_name: Name of the dataset
        prop_score: Propensity scores; Shape: (n)
    """

    data_size = w.shape[0]
    ite_estimates = np.reshape(ite_estimates, (data_size))
    t = np.reshape(t, (data_size))
    y = np.reshape(y, (data_size))

    # Decision policy: Recommend treatment based ITE. Check whether positive ITE is desirable for the datsaet or not
    if dataset_name not in ['twins']:
        decision_policy = 1 * (ite_estimates > 0)
    else:
        decision_policy = 1 * (ite_estimates < 0)

    decision_policy = np.reshape(decision_policy, (data_size))
    indices = t == decision_policy
    weighted_outcome = y / prop_score

    value_score = np.sum(weighted_outcome[indices]) / data_size
    if dataset_name not in ['twins']:
        value_score = -1*value_score

    return {'value_score' : value_score}


def calculate_value_dr_risk(
                                ite_estimates: np.ndarray, 
                                w: np.ndarray, 
                                t: np.ndarray, 
                                y: np.ndarray, 
                                dataset_name: str, 
                                outcome_pred: Tuple[np.ndarray, np.ndarray], 
                                prop_score: np.ndarray, 
                                inv_prop_threshold: float=0
                            ) -> dict:
    """
    Inputs:
        ite_estimates: ITE estimates; Shape: (n)
        w: Covariates; Expected Shape: (n, d)
        t: Treatment; Expected Shape: (n, 1)
        y: Outcome; Expected Shape: (n, 1)
        dataset_name: Name of the dataset
        outcome_pred: Predicted outcomes T Learner; Shape: [(n), (n)]
        prop_score: Propensity scores; Shape: (n)
        inv_prop_threshold: Propensity scores; Shape: (n)
    """

    data_size = w.shape[0]
    t = np.reshape(t, (data_size))
    y = np.reshape(y, (data_size))

    mu_0, mu_1 = outcome_pred
    mu = mu_0 * (1 - t) + mu_1 * (t)

    # Decision Policy: Recommend treatment based ITE. Check whether positive ITE is desirable for the datsaet or not
    if dataset_name not in ['twins']:
        decision_policy = 1 * (ite_estimates > 0)
    else:
        decision_policy = 1 * (ite_estimates < 0)
    decision_policy = np.reshape(decision_policy, (data_size))

    # Propensity clipping version
    if inv_prop_threshold:
        indices = prop_score < 1/inv_prop_threshold
        prop_score[indices]= 1/inv_prop_threshold

        indices= prop_score > 1 - 1/inv_prop_threshold
        prop_score[indices]= 1-1/inv_prop_threshold

    # Value DR Score
    value_dr_score = decision_policy * (mu_1 - mu_0 + (2 * t - 1) * (y - mu) / prop_score)
    if dataset_name not in ['twins']:
        value_dr_score = -1*value_dr_score

    out={}
    if inv_prop_threshold:
        out['value_dr_clip_score']= np.mean(value_dr_score)
    else:
        out['value_dr_score'] = np.mean(value_dr_score)

    return out


def calculate_influence_risk(
                                ite_estimates: np.ndarray, 
                                w: np.ndarray, 
                                t: np.ndarray, 
                                y: np.ndarray, 
                                outcome_pred: Tuple[np.ndarray, np.ndarray],
                                prop_prob: np.ndarray, 
                                inv_prop_threshold: float=0
                            ) -> dict:
    """
    Inputs:
        ite_estimates: ITE estimates; Shape: (n)
        w: Covariates; Expected Shape: (n, d)
        t: Treatment; Expected Shape: (n, 1)
        y: Outcome; Expected Shape: (n, 1)
        outcome_pred: Predicted outcomes T Learner; Shape: [(n), (n)]
        prop_prob: Predicted probabilities of treatment; Shape: (n, 2)
        inv_prop_threshold: Propensity scores; Shape: (n)
    """

    data_size= w.shape[0]
    t= np.reshape(t, (data_size))
    y= np.reshape(y, (data_size))

    mu_0, mu_1= outcome_pred
    t_learner_ite= mu_1 - mu_0
    plug_in_estimate= t_learner_ite - ite_estimates
    A= t - prop_prob[:, 1]
    C= prop_prob[:, 0] * prop_prob[:, 1]
    B= 2*t*(t- prop_prob[:, 1])*(1/C)
        
    #Influence Score
    influence_score = (1 - B) * (t_learner_ite ** 2) + B * y * plug_in_estimate - A * (plug_in_estimate ** 2) + ite_estimates ** 2

    # Propensity clipping version
    prop_score = prop_prob[:, 0] * (1 - t) + prop_prob[:, 1] * (t)
    if inv_prop_threshold:
        indices= np.where(np.logical_and(prop_score >= 1/inv_prop_threshold, prop_score <= 1-1/inv_prop_threshold))[0]
        influence_score= np.mean(influence_score[indices])        

    out={}
    if inv_prop_threshold:
        out['influence_clip_score']= np.mean(influence_score)
    else:
        out['influence_score'] = np.mean(influence_score)

    return out