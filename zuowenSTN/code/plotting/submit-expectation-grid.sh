#!/bin/bash -l

module load python_gpu/3.6.4

JOB_NAME='eval_svhn_exp'
ttime=4:00
mem_per_gpu=15000

config_name='../configs/svhn_spatial.json'
repo_dir='/cluster/work/math/fanyang-broglil/CoreRepo/runs_christina/'
EVAL_ON_TRAIN=0
out_folder='/cluster/work/math/fanyang-broglil/CoreRepo/runs_christina/cdf_pkl_files/'

bsub -J $JOB_NAME -n 1 -W $ttime -R "rusage[mem=$mem_per_gpu, ngpus_excl_p=1]" -R "select[gpu_model1==GeForceGTX1080Ti]" \
python compute_exp_over_grid_eval.py \
--config=$config_name \
--save_root_path=$repo_dir \
--save_folder_pkl=$out_folder \
--exp_id '3HXnyE5mCt_1072386' \
--method 'core' \
--k=10 \
--eval_on_train=$EVAL_ON_TRAIN
