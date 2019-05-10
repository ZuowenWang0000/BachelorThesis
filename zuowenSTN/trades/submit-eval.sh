#!/bin/bash -l

module load python_gpu/3.6.4

JOB_NAME='eval_linf_core'
ttime=24:00
mem_per_gpu=15000

config_name='configs/christinaconfig_cifar10_linf_gr4_eval.json'
repo_dir='/cluster/work/math/fanyang-broglil/CoreRepo/runs_christina/'
EVAL_ON_TRAIN=0
SAVE_FNAME='eval_linf_core_400steps_rand_eps16_steps4.json'

bsub -J $JOB_NAME -n 1 -W $ttime -R "rusage[mem=$mem_per_gpu, ngpus_excl_p=1]" -R "select[gpu_model1==GeForceGTX1080Ti]" \
python eval.py \
--config=$config_name \
--save_root_path=$repo_dir \
--exp_id_list '7BHWj7Yf7e_1097090' '7VRCyehxRR_1097092' 'QebrkierBZ_1097094' 'gD8ngxZSm9_1097096' 'jsBpXT5WEX_1097088' \
--eval_on_train=$EVAL_ON_TRAIN \
--save_filename=$SAVE_FNAME \
--linf_attack=1
