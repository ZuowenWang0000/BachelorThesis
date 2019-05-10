#!/bin/bash -l

bsub -J $JOB_NAME -n 1 -W $ttime -R "rusage[mem=$mem_per_gpu, ngpus_excl_p=$num_gpu_per_job]" -R "select[gpu_model1==GeForceGTX1080Ti]" \
python train_with_stn.py \
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
