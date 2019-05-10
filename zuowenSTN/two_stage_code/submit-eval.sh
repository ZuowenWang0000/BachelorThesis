#!/bin/bash -l

module load python_gpu/3.6.4

JOB_NAME='eval_two_stage'
ttime=12:00
mem_per_gpu=15000

config_name='configs/christinaconfig_cifar10_spatial_eval.json'
repo_dir='/cluster/home/wangzu/zuowenSTN/two_stage_code/noAdvCoreRepo/logdir'
EVAL_ON_TRAIN=0
SAVE_FNAME='eval_two_stage.json'

bsub -J $JOB_NAME -n 1 -W $ttime -R "rusage[mem=$mem_per_gpu, ngpus_excl_p=1]" -R "select[gpu_model1==GeForceGTX1080Ti]" \
python eval_two_stage.py \
--config=$config_name \
--save_root_path=$repo_dir \
--exp_id_list 'WXgywpiVKs_1757588' \
--eval_on_train=$EVAL_ON_TRAIN \
--save_filename=$SAVE_FNAME \
--linf_attack=0 \
--reg_model_path='../regression_code/regRepo/logdir/EwFkq8PSdS_1831255/checkpoint-180000'
