import pickle
import random
import numpy as np

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet, RidgeClassifier
from sklearn.svm import SVR, LinearSVR, SVC, LinearSVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor,\
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.exceptions import UndefinedMetricWarning

#Hyperparam search grids                    
alphas = {'alpha': np.logspace(-4, 5, 10)}
# gammas = [] + ['scale']
Cs = np.logspace(-4, 5, 10)
d_Cs = {'C': Cs}
SVM = 'svm'
d_Cs_pipeline = {SVM + '__C': Cs}
max_depths = list(range(2, 10 + 1)) + [None]
d_max_depths = {'max_depth': max_depths}
d_max_depths_base = {'base_estimator__max_depth': max_depths}
# Ks = {'n_neighbors': [1, 2, 3, 5, 10, 15, 25, 50, 100, 200]}
Ks = {'n_neighbors': [1, 2, 3, 5, 10, 15, 20, 25, 40, 50]}


OUTCOME_MODEL_GRID = { 'no_hparam' : [], 'regularized_lr': [],  'svr': [], 'forest': [], 'misc': [] }
PROP_SCORE_MODEL_GRID = { 'no_hparam' : [], 'logistic': [],  'svm': [], 'misc': []  }

OUTCOME_MODEL_GRID['no_hparam']= [
    
    ('LinearRegression', LinearRegression(), {}),
    ('LinearRegression_interact',
     make_pipeline(PolynomialFeatures(degree=2, interaction_only=True),
                   LinearRegression()),
     {}),
    ('LinearRegression_degree2',
     make_pipeline(PolynomialFeatures(degree=2), LinearRegression()), {}),    
    
]

OUTCOME_MODEL_GRID['regularized_lr']= [
    
    ('Ridge', lambda x: Ridge(alpha=x),  alphas),
    ('Lasso', lambda x: Lasso(alpha=x), alphas),
    ('ElasticNet', lambda x: ElasticNet(alpha=x), alphas),
    ('KernelRidge', lambda x: KernelRidge(alpha=x), alphas),    
    
]

OUTCOME_MODEL_GRID['svr']= [
    
    ('SVR_rbf', lambda x: SVR(kernel='rbf', C=x), d_Cs),
    ('SVR_sigmoid', lambda x: SVR(kernel='sigmoid', C=x), d_Cs),    
    ('LinearSVR', lambda x: LinearSVR(), d_Cs),
  
]

OUTCOME_MODEL_GRID['forest']= [
    
    # TODO: also cross-validate over min_samples_split and min_samples_leaf
    ('DecisionTree', lambda x: DecisionTreeRegressor(max_depth= x), d_max_depths),
    ('RandomForest', lambda x: RandomForestRegressor(max_depth= x), d_max_depths),
    
]
    
OUTCOME_MODEL_GRID['misc']= [
    
    # TODO: also cross-validate over learning_rate
    ('GradientBoosting', lambda x: GradientBoostingRegressor(max_depth=x), d_max_depths),
    
]



PROP_SCORE_MODEL_GRID['no_hparam']= [
    
    ('LogisticRegression',  LogisticRegression(penalty='none'), {}),    
    ('LDA', LinearDiscriminantAnalysis(), {}),
    ('LDA_shrinkage', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), {}),
    ('QDA', QuadraticDiscriminantAnalysis(), {}),    
    ('GaussianNB', GaussianNB(), {}),    
    
]

PROP_SCORE_MODEL_GRID['logistic']= [
    
    ('LogisticRegression_l2', lambda x: LogisticRegression(penalty='l2', C=x),  d_Cs),
    ('LogisticRegression_l1_liblinear', lambda x: LogisticRegression(penalty='l1', solver='liblinear', C=x), d_Cs),
    ('LogisticRegression_l2_liblinear', lambda x: LogisticRegression(penalty='l2', solver='liblinear', C=x), d_Cs),
    ('LogisticRegression_l1_saga', lambda x: LogisticRegression(penalty='l1', solver='saga', C=x), d_Cs),
    
]

PROP_SCORE_MODEL_GRID['svm']= [
    
    ('SVM_rbf', lambda x: SVC(kernel='rbf', probability=True, C=x), d_Cs),    
    ('SVM_sigmoid', lambda x: SVC(kernel='sigmoid', probability=True, C=x), d_Cs),   
    
]

PROP_SCORE_MODEL_GRID['misc']= [
    
    ('kNN', lambda x: KNeighborsClassifier(n_neighbors=x), Ks),    
    
    # TODO: also cross-validate over learning_rate
    ('GradientBoosting', lambda x: GradientBoostingClassifier(max_depth=x), d_max_depths),    

]

def get_nuisance_models_list():

    # Loop over different hyperparams to construct list: (model_name, model(hparam))
    res={}
    res['outcome_models']= {}
    res['prop_score_models']= {}
    
    for key in res.keys():
        if key == 'outcome_models':
            meta_list= OUTCOME_MODEL_GRID
        elif key == 'prop_score_models':
            meta_list= PROP_SCORE_MODEL_GRID
        
        for sub_key in meta_list.keys():

            for (model_name, model, param_grid) in meta_list[sub_key]:
                if not param_grid:
                    if 'no_hparam' not in res[key].keys():
                        res[key]['no_hparam']= []
                    res[key]['no_hparam'].append( {'name': model_name, 'hparam': 'none', 'model_func': model} )
                else:
                    hparam_list= list(param_grid.values())[0]
                    hparam_name= list(param_grid.keys())[0]
                    if model_name not in res[key].keys():
                        res[key][model_name]= []
                    for hparam in hparam_list:
                        res[key][model_name].append( {'name': model_name, 'hparam': hparam_name + '_' + str(hparam), 'model_func': model(hparam)} )

    # print('No Hparam Case', key, res[key]['no_hparam'])
    return res['outcome_models'], res['prop_score_models']


def get_cate_nuisance_models(estimator_name: str, nuisance_stats_dir: str) -> dict:
    """ Get the nuisance models for the CATE estimator based on the AutoML selection.

    Inputs:
        estimator_name" name of the CATE estimator
        nuisance_stats_dir: path to the directory containing the nuisance models selected by AutoML

    Returns:
        Dictionary containing the nuisance (propensity, regression) models with attributes like name, hyperparameter, and the model class.
        Used to instantiate the CATE estimator EconML object.
    """

    # Propensity Model
    if estimator_name in ['dr_learner', 'dml_learner', 'causal_forest_learner', 'x_learner']:
        model_class = clone(pickle.load(open(nuisance_stats_dir + 'prop.p', "rb")))
        model_t= {}
        model_t['name']=  str(model_class).split('(')[0]
        model_t['hparam']= str(model_class).split('(')[-1].replace(')','')
        model_t['model_func']= model_class
    else:
        #S & T Learner variants do not require the propensity model
        model_t={'name':'none', 'hparam':'none'}

    # Outcome Model
    if estimator_name in ['dml_learner', 'causal_forest_learner']:
        #Important to clone since we want to train nusiance models as per EconML estimaor's requirements
        model_class = clone(pickle.load(open(nuisance_stats_dir + 'dml.p', "rb")))
    elif estimator_name in ['dr_learner', 's_learner', 's_learner_upd', 't_learner', 'x_learner']:
        #The optimal S-learner based nuisance model is used as the outcome model for the DR/T/X Meta Learner during the training phase of CATE estimators
        #Important to clone since we want to train nusiance models as per EconML estimaor's requirements
        model_class = clone(pickle.load(open(nuisance_stats_dir + 's_learner.p', "rb")))
    model_y= {}
    model_y['name']=  str(model_class).split('(')[0]
    model_y['hparam']= str(model_class).split('(')[-1].replace(')','')
    model_y['model_func']= model_class

    return {'model_t': model_t, 'model_y': model_y}

def get_cate_final_models(estimator_name: str, grid_size: int) -> dict:
    """ Get the final regression models for the CATE estimator based on a randomly sampled list of outcome models.

    Inputs:
        estimator_name: name of the CATE estimator
        grid_size: number of models to sample per type of outcome model

    Returns:
        Dictionary containing the final regression models with attributes like name, hyperparameter, and the model class.
        Used to instantiate the CATE estimator EconML object.
    """

    #Obtain list of nuisance (outcome, propensity) models
    outcome_models, _= get_nuisance_models_list()
    models_dict= outcome_models

    #Sample from the list of different types of nuisance models
    models_approx_list=[]
    for key in models_dict.keys():
        if key == 'no_hparam':
            models_approx_list += models_dict[key]
        else:
            models_approx_list += random.sample(models_dict[key], grid_size)
    
    random.shuffle(models_approx_list)

    if estimator_name in ['dr_learner', 'dml_learner', 'x_learner', 's_learner_upd']:
        return models_approx_list
    else:
        return [{'name':'none', 'hparam':'none'}]
