#!/bin/bash
export WANDB_USER=<YOUR_USERNAME>
export WANDB_PROJECT=<YOUR_PROJECT>
export DATA_ROOT=<DATASET_ROOT_DIRECTORY>
export EXPERIMENTS_ROOT='/tmp' # change to the directory in which the experiments need to be stored
export NUM_WORKERS=6

python train.py \
        --data_conf=configurations/data/MNIST_valid.yml \
        --eval_conf=configurations/eval/VIB_simple.yml \
        --model_conf=configurations/models/VIB.yml \
        --code-dir=./example
        --epochs=10 --seed=42 --device=cpu


#python train.py --run_id=<RUN_ID> --epochs=10
