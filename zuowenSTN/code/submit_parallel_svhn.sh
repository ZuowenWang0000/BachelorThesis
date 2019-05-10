#!/bin/sh

# Hard coded settings for resources
# time limit
export ttime=24:00
# number of gpus per job
export num_gpu_per_job=1
# memory per job
export mem_per_gpu=15000

export JOB_NAME='spatial_svhn_vgg'
export config_name='configs/svhn_spatial_vgg.json'
export local_json_folder_name='local_svhn_spatial_vgg'
export repo_dir='/cluster/scratch/heinzec/simple_transformations/repo_dir_19jan2'

# load python
module load python_gpu/3.6.4
echo "Using configuration: ${config_name}"

for VAR_WORSTOFK in 1 5 10 20
do
    export WORSTOFK=$VAR_WORSTOFK
    for VAR_LAMBDA_CORE in 0 .25 .5 .75 1 1.5 2
    do
        export LAMBDA_CORE=$VAR_LAMBDA_CORE
        for VAR_USE_CORE in 1
        do
            export USE_CORE=$VAR_USE_CORE
            for VAR_SEED in 1 2 3 4 5
            do
                export SEED=$VAR_SEED
                for VAR_NUM_IDS in 64
                do
                    export NUM_IDS=$VAR_NUM_IDS
                    for VAR_FO_EPSILON in 8
                    do
                        export FO_EPSILON=$VAR_FO_EPSILON
                        for VAR_FO_NUM_STEPS in 5
                        do
                            export FO_NUM_STEPS=$VAR_FO_NUM_STEPS
                            sh submit-train.sh
                        done
                    done
                done
            done
        done
    done
done
