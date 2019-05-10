#!/bin/bash -l

python3 train_reg.py \
--config=$config_name \
--save-root-path=$repo_dir \
--local_json_dir_name=$local_json_folder_name \
--worstofk=$WORSTOFK \
--lambda-core=$LAMBDA_CORE \
--use_core=$USE_CORE \
--seed=$SEED \
--num-ids=$NUM_IDS \
--fo_epsilon=$FO_EPSILON \
--fo_num_steps=$FO_NUM_STEPS
