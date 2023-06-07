# Reproducing results of the paper

## Setup

Please download the ACIC 2016 and ACIC 2018 datasets and place them in `$HOME/scratch/` directory.

Use the requirements.txt file for installing the dependencies.

## Reproduce Results

```python scripts/reproduce_results.py```

## Analyzing results using logged DataFrame

For ease, we have provided the final dataframe that contains the data after training all the CATE estimators across all the datasets. The corresponding file (logs_ensemble.csv) can be found in the results directory.

- To generate the main table, execute the following command:
 ``` python scripts/cate_analysis.py --generate_df 0 --analysis_case ensemble_htune_pehe  ```

## Training CATE estimators

To train CATE estimators and reproduce the logged DataFrame, we describe the commands ahead for the dataset ``twins`` and seed `0`. The same commands can be executed for the reamining seed values and the complete list of datasets can be constructed as follows:

- Real Cause Datasets: `['twins', 'lalonde_psid1', 'lalonde_cps1']`
- ACIC 2016 Datasets: Load the file `datasets/acic_2016_heterogenous_list.p`
- ACIC 2018 Datasets: Load the file `datasets/acic_2018_heterogenous_list.p`

Before training the CATE estimator, we first need to ensure that we have selected the corresponding nuisance models via AutoML. To do the nuisance model selection for a given dataset and seed value, execute the following command:

`python nuisance_model_selection.py  --seed 0 --dataset twins `

After the nuisance model selection, we can execute the following command to train a particular CATE estimator for this dataset and seed. 

`python train.py  --seed 0 --dataset twins --estimator dml_learner `

The complete list of CATE estimators to be trained is as follows: `['dml_learner', 'dr_learner', 'x_learner', 'causal_forest_learner', 's_learner', 't_learner', 's_learner_upd' ]`

After training all the CATE estimators for this datasets and seed, we will train the Ensemble CATE estimators.

`python ensemble_train.py  --seed 0 --dataset twins `

