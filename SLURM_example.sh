#!/bin/bash
# Resource allocation (see SLURM docs)
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=16
#SBATCH --time 10:00:00

# Other flags to set for SLURM (see docs)
#SBATCH --ntasks=1
#SBATCH --priority=TOP
#SBATCH --job-name=JADE-example
#SBATCH -D <path_to_your_home_directory>
#SBATCH --output=log.txt
#SBATCH --verbose

# Setting the env_variables
export CUDA_CACHE_PATH="$TMPDIR"/.cuda_cache/
export CUDA_CACHE_DISABLE=0
export WANDB_USER=<YOUR_WANDB_USERNAME>
export WANDB_PROJECT=<YOUR_WANDB_PROJECT>
export EXPERIMENTS_ROOT=<YOUR_EXPERIMENT_ROOT>
export N_WORKERS=16
export DEVICE=cuda

# Pointing to the directory in which the dataset is stored
export DATA_ROOT=<YOUR_DATA_ROOT> # /hddstore/datasets

# here I use temp to save the backups since they are uploaded and deleted afterwards anyway
mkdir /tmp/experiments
export EXPERIMENTS_ROOT=/tmp/experiments

#Generate cuda stats to check cuda is found
nvidia-smi
echo Starting

# Make sure you are in the project directory before trying to run the agent
cd <PATH_TO_THE_PROJECT_ROOT>

# Run the agent
echo Starting agent $WANDB_USER/$WANDB_PROJECT/$SWEEP_ID
wandb agent $SWEEP_ID
wait

# Remove all the files for the model backups since wandb is uploading them anyway
# You can change the experiments directory if you want to keep local versions
rm -r /tmp/experiments