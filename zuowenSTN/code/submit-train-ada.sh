#!/bin/bash -l

module load modules/Python3
module load tensorflow-gpu/1.12.0-python3.6

export CUDA_VISIBLE_DEVICES=0

config_name='configs/christinaconfig_cifar100_nat.json'
repo_dir='/userdata/heinzec/simple_transformations/repo_dir'
local_json_folder_name='cifar100_nat'
WORSTOFK=1
LAMBDA_CORE=0
USE_CORE=0
SEED=2
NUM_IDS=64
FO_EPSILON=8
FO_STEP_SIZE=2
FO_NUM_STEPS=5

python train.py \
--config=$config_name \
--save-root-path=$repo_dir \
--local_json_dir_name=$local_json_folder_name \
--worstofk=$WORSTOFK \
--lambda-core=$LAMBDA_CORE \
--use_core=$USE_CORE \
--seed=$SEED \
--num-ids=$NUM_IDS \
--fo_epsilon=$FO_EPSILON \
--fo_step_size=$FO_STEP_SIZE \
--fo_num_steps=$FO_NUM_STEPS
