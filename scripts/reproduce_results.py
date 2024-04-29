import os

dataset_name= 'twins'
root_dir= 'cate_eval_analysis'
total_seeds= 20

#Generate Dataset
for seed in range(total_seeds):
    script= 'python generate_data.py ' + ' --seed ' + str(seed) + ' --root_dir ' +  str(root_dir) + ' --meta_dataset ' + 'realcause'

#Train Nuisance Models
for seed in range(total_seeds):
    script= 'python nuisance_model_selection.py --res_dir results_final ' + ' --seed ' + str(seed) + ' --root_dir ' +  str(root_dir) + ' --dataset ' + str(dataset_name)
    os.system(script)

#Train Estimators
for seed in range(total_seeds):
    for estimator in ['s_learner', 't_learner', 'x_learner', 'causal_forest_learner', 'dml_learner', 'dr_learner', 's_learner_upd']:
        script= 'python train.py --res_dir results_final ' + ' --estimator ' + str(estimator) + ' --seed ' + str(seed) + ' --root_dir ' +  str(root_dir) + ' --dataset ' + str(dataset_name)
        os.system(script)

#Train Ensembles over Meta-Estimators using Surrogate Metrics
for seed in range(total_seeds):
    script= 'python ensemble_train.py --res_dir results_final ' + ' --seed ' + str(seed) + ' --root_dir ' +  str(root_dir) + ' --dataset ' + str(dataset_name)
    os.system(script)

#Note that generate_df argument can be ignored in subsequent calls once the results have been generated. 
#It is required to set the flag as 1 when launching the command for the first time.

#Analyse results with single level model selection strategy
script= 'python cate_analysis.py --analysis_case single_level_selection ' + ' --generate_df 1 '
os.system(script)

#Analyse results with two level model selection strategy
script= 'python cate_analysis.py --analysis_case two_level_selection ' + ' --generate_df 1 '
os.system(script)

#Analyse results with ensemble selection strategy
script= 'python cate_analysis.py --analysis_case ensemble_selection ' + ' --generate_df 1 '
os.system(script)
