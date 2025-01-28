# Empirical Analysis of Model Selection for Heterogeneous Causal Effect Estimation

Code accompanying the paper [Empirical Analysis of Model Selection for Heterogeneous Causal Effect Estimation](https://arxiv.org/abs/2211.01939).
The paper has been accepted at ICLR 2024 for a spotlight presentation. [OpenReview](https://openreview.net/forum?id=yuy6cGt3KL) [Talk](https://iclr.cc/virtual/2024/poster/17398).

# Brief note about the paper

<!-- <img src="main_fig.png" width=65% align="center"> -->

We study the problem of model selection in causal inference, specifically for conditional average treatment effect (CATE) estimation. Unlike machine learning, there is no perfect analogue of cross-validation for model selection as we do not observe the counterfactual potential outcomes. Towards this, a variety of surrogate metrics have been proposed for CATE model selection that use only observed data. However, we do not have a good understanding regarding their effectiveness due to limited comparisons in prior studies. We conduct an extensive empirical analysis to benchmark the surrogate model selection metrics introduced in the literature, as well as the novel ones introduced in this work. We ensure a fair comparison by tuning the hyperparameters associated with these metrics via AutoML, and provide more detailed trends by incorporating realistic datasets via generative modeling. Our analysis suggests novel model selection strategies based on careful hyperparameter selection of CATE estimators and causal ensembling.

# Reproducing results of the paper

A script to reproduce results of the paper can be executed as follows.

`python scripts/reproduce_results.py`

## Setup

Please download the ACIC 2016 datasets and place them in `root_dir/acic_2016/` directory.

- ACIC 2016 benchmark link: https://jenniferhill7.wixsite.com/acic-2016/competition

Use the requirements.txt file for installing the dependencies.

## Training CATE estimators

To train CATE estimators we describe the commands ahead for the dataset ``twins`` and seed `0`. The same commands can be executed for the remaining seed values and datasets.

First we create the train/val splits for each dataset by executing the following command for the Real Cause datasets.

- `python generate_date.py --seed 0 --meta_dataset realcause` 

For the case datasets in the ACIC 2016 benchmark, we create the train/val splits by executing the following command.

- `python generate_date.py --seed 0 --meta_dataset acic`

Before training the CATE estimator, we first need to ensure that we have selected the corresponding nuisance models via AutoML. To do the nuisance model selection for a given dataset and seed value, execute the following command:

`python nuisance_model_selection.py  --seed 0 --dataset twins `

After the nuisance model selection, we can execute the following command to train a particular CATE estimator for this dataset and seed. 

`python train.py  --seed 0 --dataset twins --estimator dml_learner `

The complete list of CATE estimators to be trained is as follows: `['dml_learner', 'dr_learner', 'x_learner', 'causal_forest_learner', 's_learner', 't_learner', 's_learner_upd' ]`

After training all the CATE estimators for this datasets and seed, we will train the Ensemble CATE estimators.

`python ensemble_train.py  --seed 0 --dataset twins `

