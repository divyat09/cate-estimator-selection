import os

dataset_name= 'twins'
root_dir= 'causal_val_project'
total_seeds= 5

#Train Nuisance Models
for seed in range(total_seeds):
    script= 'python nuisance_model_selection.py --res_dir results_final ' + ' --seed ' + str(seed) + ' --root_dir ' +  str(root_dir) + ' --dataset ' + str(dataset_name)
    os.system(script)

#Train Estimators
for seed in range(total_seeds):
    for estimator in ['s_learner', 't_learner', 'x_learner', 'causal_forest_learner', 'dml_learner', 'dr_learner', 's_learner_upd']:
        script= 'python train.py --res_dir results_final --train_estimators 1 --automl_selection 1 ' + ' --estimator ' + str(estimator) + ' --seed ' + str(seed) + ' --root_dir ' +  str(root_dir) + ' --dataset ' + str(dataset_name)
        os.system(script)

#Analyse Results
script= 'python main_analysis.py'
os.system(script)

script= 'python scripts/new_analysis.py'
os.system(script)