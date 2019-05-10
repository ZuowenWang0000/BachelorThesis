#!/bin/bash -l

module load python_gpu/3.6.4

JOB_NAME='eval_linf_rob50'
ttime=24:00
mem_per_gpu=15000

config_name='configs/christinaconfig_cifar10_linf_gr4_eval.json'
repo_dir='/cluster/work/math/fanyang-broglil/CoreRepo/runs_christina/'
EVAL_ON_TRAIN=0
SAVE_FNAME='eval_linf_rob50_400steps_rand_eps16_steps4.json'

bsub -J $JOB_NAME -n 1 -W $ttime -R "rusage[mem=$mem_per_gpu, ngpus_excl_p=1]" -R "select[gpu_model1==GeForceGTX1080Ti]" \
python eval.py \
--config=$config_name \
--save_root_path=$repo_dir \
--exp_id_list '3gusbQzNmS_1097052' 'HcWWVSYMVF_1097054' 'Lewp2D75WV_1097056' 'QNeJE7eHdv_1097048' 'Y4uzCwsgUf_1097050' \
--eval_on_train=$EVAL_ON_TRAIN \
--save_filename=$SAVE_FNAME \
--linf_attack=1
