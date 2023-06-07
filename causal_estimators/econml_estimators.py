import numpy as np
import pandas as pd
import scipy
import sparse as sp
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


#Class
class EconMLEstimator():
    
    def __init__(self, estimator_type, estimator):
        
        self.estimator_type= estimator_type
        self.estimator= estimator
        
        return
    
    def fit(self, w, t, y):
        
        self.estimator.fit(y, t, X=w)
        
        return 
        
    def ate(self, w, t0, t1):
        
        return self.estimator.ate(w, T0=t0, T1=t1)
        

    def effect(self, w, t0, t1):
        
        return self.estimator.effect(w, T0=t0, T1=t1)

    def get_nuisance_model(self, case='regression'):
        
        '''
        case: type of nuisance model; regression or classification
        
        return: list of nuisance models corresponding to different splits
        
        '''
        
        out=[]
        
        if self.estimator_type in ['dml_learner', 'causal_forest_learner']:
            
            #Return type of estimator.model is [[ model_seed_0, ..., model_seed_N ]]; hence return estimator.model[0]
            
            if case == 'regression':            
                out= self.estimator.models_y[0]
            elif case == 'classification':
                out= self.estimator.models_t[0]
            
        elif self.estimator_type in ['dr_learner', 'dr_learner_tune_0.1', 'dr_learner_tune_0.01']:
            
            #Return type of estimator.model is [[ model_seed_0, ..., model_seed_N ]]; hence return estimator.model[0]
            
            if case == 'regression':            
                out= self.estimator.models_regression[0]
            elif case == 'classification':
                out= self.estimator.models_propensity[0]

        elif self.estimator_type in ['s_learner_upd']:

            # Return type of estimator.model is model, hence return [estimator.model]

            if case == 'regression':
                out = [self.estimator.model_]

            elif case == 'classification':
                print('Error: Propensity score models not supported by the estimator')

        elif self.estimator_type in ['s_learner']:
            
            #Return type of estimator.model is model, hence return [estimator.model]
            
            if case == 'regression':                
                out= [self.estimator.overall_model]
            
            elif case == 'classification':
                print('Error: Propensity score models not supported by the estimator')
                        
        elif self.estimator_type in ['t_learner', 'x_learner']:
            
            #Return type of estimator.model is [model_t0, model_t1]
            
            if case == 'regression':                
                out= self.estimator.models
            
            elif case == 'classification':
                print('Error: Propensity score models not supported by the estimator')

        return out

    def eval_final_model(self, w, t, y):

        data_size = w.shape[0]
        y = np.reshape(y, (data_size))

        # Compute final model score for dml, dr again without econml
        # This computes final model score on the training split by default
        if self.estimator_type in ['dml_learner', 'linear_dml', 'sparse_linear_dml', 'causal_forest_learner', 'dr_learner', 'linear_dr', 'sparse_linear_dr', 'forest_dr', 'dr_learner_tune_0.1', 'dr_learner_tune_0.01']:
            final_model_score= self.estimator.score(y, t, X=w)

        elif self.estimator_type in ['s_learner_upd']:

            # Check inputs
            if w is None:
                w = np.zeros((y.shape[0], 1))

            ones = np.hstack([np.ones((data_size, 1)), w])
            zeros = np.hstack([np.zeros((data_size, 1)), w])
            diffs = self.estimator.model_.predict(ones) - self.estimator.model_.predict(zeros)
            final_model_score= self.estimator.model_final_.score(w, diffs)

        elif self.estimator_type in ['x_learner']:

            # Get Nuisance Models
            models= self.get_nuisance_model(case= 'regression')

            # Check inputs
            if y.ndim == 2 and y.shape[1] == 1:
                y = y.flatten()
            if w is None:
                w = np.zeros((y.shape[0], 1))

            #         y, t, w, _ = check_inputs(y, t, w, multi_output_T=False)

            categories = self.estimator.categories

            # Would not be executed; check copied from econml implementation
            if categories != 'auto':
                categories = [categories]  # OneHotEncoder expects a 2D array with features per column
            # Transforms treatment (N,) to shape (N, 1)
            transformer = OneHotEncoder(categories=categories, sparse=False, drop='first')
            t = transformer.fit_transform(t.reshape(-1, 1))

            self._d_t = t.shape[1:]
            t = self.inverse_onehot(t)

            #                 self.models = check_models(self.models, self._d_t[0] + 1)

            pred = np.zeros(data_size)
            final_model_score = 0.0
            # Unlike other estimators, iterating over models gives model trained corresponding to treatment == idx
            for idx in range(self._d_t[0]):

                imputed_effect_on_controls= models[idx+1].predict(w[t==0]) - y[t==0]
                imputed_effect_on_treated= y[t==idx+1] - models[0].predict(w[t==idx+1])
                final_model_score += self.estimator.cate_controls_models[idx].score(w[t==0], imputed_effect_on_controls)
                final_model_score += self.estimator.cate_treated_models[idx].score(w[t==idx+1], imputed_effect_on_treated)

            final_model_score = final_model_score / (2*self._d_t[0])

        else:
            print('Error: Final model score not implemented for this estimator')

        print(final_model_score)
        return final_model_score

    def eval_nuisance_model(self, w, t, y, case='regression'):
        
        '''                
        '''
        
        out_pred=[]
        out_score= []
        
        data_size= w.shape[0]
        y= np.reshape(y, (data_size))
        
        if self.estimator_type in ['dml_learner', 'linear_dml', 'sparse_linear_dml', 'causal_forest_learner']:
            
            models= self.get_nuisance_model(case= case)
            
            if case == 'regression':                
                for model in models:
                    pred= model.predict(w)
                    out_pred.append(pred)
                    
                    score= model.score(w, y)
                    out_score.append(score)
                    
            elif case == 'classification':
                for model in models:
                    pred= model.predict(w)
                    out_pred.append(pred)
                    
                    score= model.score(w, t)
                    out_score.append(score)   

        elif self.estimator_type in ['dr_learner', 'linear_dr', 'sparse_linear_dr', 'forest_dr', 'dr_learner_tune_0.1', 'dr_learner_tune_0.01']:
                        
            models= self.get_nuisance_model(case= case)
            
            if case == 'regression':             
                
                #Since we need to stack both (w, t); we should have t as (N, 1) shape
                t= np.reshape(t, (data_size, 1))
            
                for model in models:
                    pred= model.predict(np.hstack([w,t]))
                    out_pred.append(pred)
                    
                    score= model.score(np.hstack([w,t]), y)
                    out_score.append(score)
                    
            elif case == 'classification':
                for model in models:
                    pred= model.predict(w)
                    out_pred.append(pred)
                    
                    score= model.score(w, t)
                    out_score.append(score)   

        elif self.estimator_type in ['s_learner_upd']:

            models= self.get_nuisance_model(case=case)

            if case == 'regression':

                # Check inputs
                if w is None:
                    w = np.zeros((y.shape[0], 1))

                feat_arr = np.hstack([t.reshape(-1,1), w])

                for model in models:

                    pred = model.predict(feat_arr)
                    out_pred.append(pred)

                    score = model.score(feat_arr, y)
                    out_score.append(score)

            elif case == 'classification':
                print('Error: Propensity score models not supported by the estimator')

                        
        elif self.estimator_type in ['s_learner']:    
            
            models= self.get_nuisance_model(case= case)
        
            if case == 'regression':
                
                # Check inputs
                if w is None:
                    w = np.zeros((y.shape[0], 1))

        #         y, t, w, _ = check_inputs(y, t, w, multi_output_T=False)

                categories = self.estimator.categories

                #Would not be executed; check copied from econml implementation
                if categories != 'auto':
                    categories = [categories]  # OneHotEncoder expects a 2D array with features per column
                # Transforms treatment (N,) to shape (N, 1)
                transformer = OneHotEncoder(categories=categories, sparse=False, drop='first')
                t = transformer.fit_transform(t.reshape(-1, 1))
                
                # Note: unlike other Metalearners, we need the controls' encoded column for training
                # Thus, we append the controls column before the one-hot-encoded T
                # We might want to revisit, though, since it's linearly determined by the others
                feat_arr = np.concatenate((w, 1 - np.sum(t, axis=1).reshape(-1, 1), t), axis=1)    
                
                for model in models:
                    
                    pred= model.predict(feat_arr)
                    out_pred.append(pred)
                    
                    score= model.score(feat_arr, y)
                    out_score.append(score)                    
            
            elif case == 'classification':
                print('Error: Propensity score models not supported by the estimator')            
            
            
        elif self.estimator_type in ['t_learner', 'x_learner']:
            
            models= self.get_nuisance_model(case= case)
        
            if case == 'regression':
                
                # Check inputs
                if y.ndim == 2 and y.shape[1] == 1:
                    y = y.flatten()
                if w is None:
                    w = np.zeros((y.shape[0], 1))

        #         y, t, w, _ = check_inputs(y, t, w, multi_output_T=False)

                categories = self.estimator.categories

                #Would not be executed; check copied from econml implementation
                if categories != 'auto':
                    categories = [categories]  # OneHotEncoder expects a 2D array with features per column
                # Transforms treatment (N,) to shape (N, 1)
                transformer = OneHotEncoder(categories=categories, sparse=False, drop='first')
                t = transformer.fit_transform(t.reshape(-1, 1))
                
                self._d_t = t.shape[1:]
                t = self.inverse_onehot(t)
                
#                 self.models = check_models(self.models, self._d_t[0] + 1)

                pred= np.zeros(data_size)
                score= 0.0
                # Unlike other estimators, iterating over models gives model trained corresponding to treatment == idx        
                for idx in range(self._d_t[0] + 1):
                    inds= t == idx
                    pred[inds] = models[idx].predict(w[inds])                    
                    score += models[idx].score(w[inds], y[inds])                    
                score = score/(self._d_t[0] + 1)
                
                out_pred.append(pred)
                out_score.append(score)
            
            elif case == 'classification':
                print('Error: Propensity score models not supported by the estimator')


        return out_pred, out_score

    
    def issparse(self, X):
        """Determine whether an input is sparse.
        For the purposes of this function, both `scipy.sparse` matrices and `sparse.SparseArray`
        types are considered sparse.
        Parameters
        ----------
        X : array-like
            The input to check
        Returns
        -------
        bool
            Whether the input is sparse
        """
        return scipy.sparse.issparse(X) or isinstance(X, sp.SparseArray)

    def ndim(self, X):
        """Return the number of array dimensions."""
        return X.ndim if self.issparse(X) else np.ndim(X)


    def inverse_onehot(self, T):
        """
        Given a one-hot encoding of a value, return a vector reversing the encoding to get numeric treatment indices.
        Note that we assume that the first column has been removed from the input.
        Parameters
        ----------
        T : array (shape (n, d_t-1))
            The one-hot-encoded array
        Returns
        -------
        A : vector of int (shape (n,))
            The un-encoded 0-based category indices
        """
        assert self.ndim(T) == 2
        # note that by default OneHotEncoder returns float64s, so need to convert to int
        return (T @ np.arange(1, T.shape[1] + 1)).astype(int)
    
