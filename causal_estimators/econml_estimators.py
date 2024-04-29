import numpy as np
import pandas as pd
import scipy
import sparse as sp
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.base import clone

#EconML Estimators
from econml.dml import DML, CausalForestDML
from econml.dr import DRLearner
from econml.metalearners import XLearner, TLearner, SLearner
from econml._cate_estimator import LinearCateEstimator

# Pass a list of nuisance models required for the Meta Estimator under the following scheme: ( (model_name, model_y), (model_name, model_t) )
def get_estimator_class(estimator_name):

    if estimator_name == 'dml_learner':
        return lambda model_dict, model_final: DML(
                                                    model_t= model_dict['model_t']['model_func'], 
                                                    model_y= model_dict['model_y']['model_func'], 
                                                    model_final= model_final['model_func'],  
                                                    discrete_treatment=True, 
                                                    linear_first_stages=False, cv=3
                                                   )

    if estimator_name == 'dr_learner':
        return lambda model_dict, model_final: DRLearner(
                                                        model_propensity= model_dict['model_t']['model_func'], 
                                                        model_regression= model_dict['model_y']['model_func'],
                                                        model_final= model_final['model_func'], cv=3
                                                        )

    elif estimator_name == 'x_learner':
        return lambda model_dict, model_final: XLearner(
                                                        propensity_model= model_dict['model_t']['model_func'],
                                                        models= model_dict['model_y']['model_func'],
                                                        cate_models= model_final['model_func']
                                                        )

    elif estimator_name == 's_learner_upd':
        return lambda model_dict, model_final: SLearnerUpdated(
                                                                overall_model=model_dict['model_y']['model_func'],
                                                                final_model=model_final['model_func']
                                                            )

    elif estimator_name == 'causal_forest_learner':
        return lambda model_dict: CausalForestDML(
                                                    model_t=model_dict['model_t']['model_func'],
                                                    model_y=model_dict['model_y']['model_func'],
                                                    discrete_treatment=True,
                                                    cv=3
                                                )

    elif estimator_name == 's_learner':
        return lambda model_dict: SLearner(overall_model=model_dict['model_y']['model_func'])

    elif estimator_name == 't_learner':    
        return lambda model_dict: TLearner(models=model_dict['model_y']['model_func'])

#Projected S-Learner building upon LinearCateEstimator from EconML
class SLearnerUpdated(LinearCateEstimator):

    def __init__(self, *, overall_model, final_model):
        """Projected S Learner

        Inputs:
            overall_model: Regression model for predicting outcomes using S-Learner approach
            final_model: Regression model for the treatment effect estmation
        """
        self.overall_model = overall_model
        self.final_model = final_model
        return

    def fit(self, y, T, X, W=None):
        """ Train the CATE estimator on data

        Inputs:
            X: Covariates; Expected Shape: (n, d)
            W: Additional Covariates; Expected Shape: (n, d')
            T: Treatment; Expected Shape: (n, 1)
            y: Outcome; Expected Shape: (n)
        """

        XW = X
        if W is not None:
            XW = np.hstack([X, W])
        self.model_ = clone(self.overall_model)
        self.model_.fit(np.hstack([T.reshape(-1, 1), XW]), y)
        ones = np.hstack([np.ones((X.shape[0], 1)), XW])
        zeros = np.hstack([np.zeros((X.shape[0], 1)), XW])
        diffs = self.model_.predict(ones) - self.model_.predict(zeros)

        self.model_final_ = clone(self.final_model)
        self.model_final_.fit(X, diffs)
        return self

    def effect(self, X, T0=0, T1=1):
        return self.const_marginal_effect(X)

    def const_marginal_effect(self, X):
        return self.model_final_.predict(X)


#Meta class over different various EconML CATE estimators to support a common interface
class EconMLEstimator():
    
    def __init__(self, 
                 estimator_name: str, 
                 nuisance_model: dict, 
                 final_model: dict
                 ):
        """ Create the CATE estimator for the specified estimator and corresponding nuisance and final models.

        Inputs:
            estimator_name: Name of the CATE estimator
            nuisance_model: Nuisance models for the CATE estimator
            final_model: Final model for the CATE estimator
        """
        
        estimator_class=  get_estimator_class(estimator_name)
        if final_model['name'] != 'none':
            self.cate_estimator = estimator_class(nuisance_model, final_model)
        else:
            self.cate_estimator = estimator_class(nuisance_model)

        return
    
    def fit(self, w, t, y):
        """ Train the CATE estimator on data

        Inputs:
            w: Covariates; Expected Shape: (n, d)
            t: Treatment; Expected Shape: (n, 1)
            y: Outcome; Expected Shape: (n)
        """
        
        self.cate_estimator.fit(y, t, X=w)
        
        return
        
    def ate(self, w, t0, t1) -> float:
        """ Compute the Average Treatment Effect (ATE) for each sample in the dataset.

        Inputs:
            w: Covariates; Expected Shape: (n, d)
            t0: Control Class; Expected Shape: (n, 1)
            t1: Treatment Class; Expected Shape: (n, 1)

        Returns:
            ite: Individual Treatment Effect; Shape: (n)
        """
               
        return self.cate_estimator.ate(w, T0=t0, T1=t1)
        

    def effect(self, w, t0, t1) -> np.ndarray:
        """ Compute the Individual Treatment Effect (ITE) for each sample in the dataset.

        Inputs:
            w: Covariates; Expected Shape: (n, d)
            t0: Control Class; Expected Shape: (n, 1)
            t1: Treatment Class; Expected Shape: (n, 1)
        
        Returns:
            ite: Individual Treatment Effect; Shape: (n)
        """
        
        return self.cate_estimator.effect(w, T0=t0, T1=t1)