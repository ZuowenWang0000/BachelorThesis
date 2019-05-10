#!/bin/sh

# Hard coded settings for resources
# time limit
export ttime=12:00
# number of gpus per job
export num_gpu_per_job=1
# memory per job
export mem_per_gpu=15000


export JOB_NAME='trades_trail'
export config_name='configs/tradesconfig_cifar10_spatial_trail.json'
export local_json_folder_name='local_cifar10_trades_trail'
export repo_dir='/cluster/home/wangzu/zuowenSTN/tradeRepo'

# load python
module load python_gpu/3.6.4
echo "Using configuration: ${config_name}"

for VAR_WORSTOFK in 1
do
    export WORSTOFK=$VAR_WORSTOFK
    for VAR_LAMBDA_CORE in 1
    do
        export LAMBDA_CORE=$VAR_LAMBDA_CORE
        for VAR_USE_CORE in 1
        do
            export USE_CORE=$VAR_USE_CORE
#            for VAR_SEED in 1 2 3 4 5
            for VAR_SEED in 1
            do
                export SEED=$VAR_SEED
                for VAR_NUM_IDS in 64
                do
                    export NUM_IDS=$VAR_NUM_IDS
                    for VAR_FO_EPSILON in 16
                    do
                        export FO_EPSILON=$VAR_FO_EPSILON
                        for VAR_FO_NUM_STEPS in 10
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
