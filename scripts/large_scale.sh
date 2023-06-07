#!/bin/sh
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-65
#SBATCH --time=1:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

#Force slurm output
export PYTHONUNBUFFERED=1

#Load module
module load python/3.8 scipy-stack

#Load python environment
source ~/causal_val_env/bin/activate

echo "Done transfering data"

# python3 nuisance_model_selection.py --seed $1 --slurm_exp 1 --res_dir results_final

python3 ensemble_train.py --seed $1 --slurm_exp 1 --res_dir results_final

#python3 train.py --seed $1 --estimator $2 --train_estimators 1 --slurm_exp 1 --res_dir results_final --automl_selection 1

#python3 train.py --seed $1 --estimator $2 --train_estimators 1 --slurm_exp 2 --res_dir results_final --dataset lalonde_cps1
