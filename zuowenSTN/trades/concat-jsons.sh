#!/bin/bash -l

repo_dir='/cluster/work/math/fanyang-broglil/CoreRepo/runs_christina'
local_json_folder_name='experiments_cifar10_fo_vgg_all_jsons'
experiment_json_fname='experiments_cifar10_fo_vgg_all.json'

module load python_gpu/3.6.4
python concatenate_jsons.py --repo_dir ${repo_dir} --local_json_dir_name ${local_json_folder_name} --output_dir ${repo_dir} --output_filename ${experiment_json_fname}
