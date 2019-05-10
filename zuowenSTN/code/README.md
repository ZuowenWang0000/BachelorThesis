# RECOVARIT

This repository contains code to train and evaluate models against
adversarially chosen rotations and translations. It can be used to reproduce the
main experiments of:

**RECOVARIT: Leveraging image identity improves
    adversarial robustness against simple transformations**<br>
*Christina Heinze-Deml, Luzius Brogli and Fanny Yang*<br>

The code is based on https://github.com/MadryLab/adversarial_spatial. 

The main scipts to run are `train.py` and `eval.py`, which will train and
evaluate a model respectively. 

**Note:** `train.py` only supports groups of size 2. For larger groups, need to use
`train_larger_groups.py`. The latter uses the more generic but slower implementation
of the conditional variance penalty.

Options are all included in `configs/cifar10_spatial.json`
annotated below. 

```
{
  "model": {
      # padding mode, passed directly to tf.pad
      "pad_mode": "constant",
      # architecture - can be "resnet" or "vgg"
      "model_family": "resnet",
      # depth parameter for the CIFAR resnet architecture; depth = 6n+2
      "resnet_depth_n": 5,
      # resnet filters
      "filters": [16, 16, 32, 64],
      # size of image fed to classifier, set to 64 for black-canvas setting (no
      # information loss during rotation and translation)
      "pad_size": 32,
      # number of output classes
      "n_classes": 10
  },

  "training": {
      "tf_random_seed": 1,
      "np_random_seed": 1,
      "max_num_training_steps": 80000,
      "num_output_steps": 5000,
      "num_summary_steps": 5000,
      "num_easyeval_steps": 5000,
      "num_eval_steps": 80000,
      "num_checkpoint_steps": 5000,
      "num_ids": 64,
      "batch_size": 128,
      "lr" : 0.1,
      "step_size_schedule": [[0, 0.1], [40000, 0.01], [60000, 0.001]],
      "momentum": 0.9,
      "weight_decay": 0.0002,
      # interleaves evaluation steps during training
      "eval_during_training": true,
      # include Linf and spatial attacks during training
      "adversarial_training": true,
      # if true, use only adversarial examples in cross entropy loss
      "adversarial_ce": false,
      # use random left-right flip (see note below)
      "data_augmentation": true,
      # use additional data augmentation for core (not fully implemented)
      "data_augmentation_core": false,
      # flag whether to use core
      "use_core": true,
      # group size used for core grouping
      "group_size": 2,
      # weight of core penalty term 
      "lambda_": 1
  },

  "eval": {
      "num_eval_examples": 10000,
      "batch_size": 128,
      # useful for quickly computing standard accuracy if set to false
      "adversarial_eval": true
  },

  "attack": {
      # perform Linf-bounded PGD attack
      "use_linf": false,
      # perform adversarial rotations and translations
      "use_spatial": true,

      # parameters for PGD attacks
      "loss_function": "xent", # can also be set to "cw" for Carlini-Wagner
      "epsilon": 8.0,
      "num_steps": 5,
      # step size for PGD attack, needs to be a list for spatial, e.g. [0.03, 0.03, 0.3]
      "step_size": 2.0,
      # whether to use a random init for PGD; recommended for spatial 
      "random_start": false,

      # parameters for spatial attack
      # can either be chosen using a few random tries or PGD
      "spatial_method": "random", # or "fo"
      "spatial_limits": [3, 3, 30], # trans_x pix, trans_y pix, rotation degrees
      "random_tries": 10, # if method is random choose the worst of x tries
      "grid_granularity": [5, 5, 31] # controls how many points are in the grid
  },

  "data": {
      # currently, dataset_name can be "cifar-10", "cifar-100" or "svhn
      "dataset_name": "cifar-10",
      "data_path": "/scratch/datasets/cifar10" 
    }
}
```

Run with a particular config file
```
python train.py --config PATH/TO/FILE
```

## Standard CIFAR data augmentation
By default data augmentation only includes random left-right flips. Standard CIFAR10
augmentation (+-4 pixel crops) can be achieved by setting
`adversarial_training: true`, `spatial_method: random`, `random_tries: 1`,
`spatial_limits: [4, 4, 0]`.

## Running RECOVARIT with mixed batches
For RECOVARIT, the number of adversarial examples in a given batch is computed as `batch_size - num_ids`.  

### Groups of size 2

The following runs core with 64 groups of size 2 and a total batch size of 128 if `batch_size: 128` and `group_size: 2`:
```
python train.py --use_core 1 --lambda-core 0.1 --num_ids 64
```

For only using e.g. 32 adversarial examples and a total batch_size of 128, 
set `batch_size: 128` and `group_size: 2`
```
python train.py --use_core 1 --lambda-core 0.1 --num_ids 96
```

### Groups of size 4
The following runs core with 32 groups of size 4 and a total batch size of 128 if `batch_size: 128` and `group_size: 4`:
```
python train_larger_groups.py --use_core 1 --lambda-core 0.1 --num_ids 32
```

## Running RECOVARIT with full adversarial batch
To run RECOVARIT using the full adversarial batch for the cross entropy loss, 
set `batch_size: 256`, `group_size: 2` and `adversarial_ce: true`. This will yield 
128 adversarial examples entering the cross entropy loss; the conditional 
variance penalty will be computed on 128 original examples and 128 adversarial
examples.
```
python train.py --use_core 1 --lambda-core 0.1 --num_ids 128
```


## Running robust optimization with 50% adversarial examples
For robust optimization with a 50/50 batch (i.e. 50% adversarial examples and
50% natural examples), run with `num_ids=batch_size/2`. I.e. the following runs
a total batch size of 128 if `batch_size: 128`:
```
python train.py --use_core 1 --lambda-core 0 --num_ids 64
```

## Running robust optimization with 100% adversarial examples 
For robust optimization with full adversarial batch the batch size needs to be
set via `num_ids` (i.e. the following runs with a total batch size of 128):
```
python train.py --use_core 0 --num_ids 128
```
The option `adversarial_training` needs to be set to `true`.


For more options see `train.py`


# Experiment-Repo Readme

Our repo dir path: `/cluster/work/math/fanyang-broglil/CoreRepo`
Datasets are stored in the common folder `/cluster/work/math/fanyang-broglil/datasets`

## Current usage on Leonhard
In `submit_parallel_wo_ray.sh` we set the `local_json_folder_name`  which is 
created within the `repo_dir`.

Each run of `train.py` creates its own `{repo_dir}/{local_json_folder_name}/{exp_ID}_{lsf_job_id}_experiments.json`.  

After all jobs finish, *all* individual json files contained in `{repo_dir}/{local_json_folder_name}` can be concatenated to a single json file using `concat-jsons.sh`
which calls `concatenate_jsons.py`.

All checkpoint IDs and metadata corresponding to an experiment ID are saved in `{repo_dir}/training_metadata/{exp_ID}_{lsf_job_id}_training_metadata.pkl`

All data corresponding to checkpoints are stored in `{repo_dir}/logdir/{checkpoint_ID}_checkpoint.xxx`.

## Current local usage
Locally, run with `train.py --save_in_local_json=0`. Then the results will be saved in `experiment_json_fname` and when creating a new ID, it is checked whether an
experiment with this ID already exists in `experiment_json_fname`.

When running locally with `train.py --save_in_local_json=1`, the results are saved in
`{repo_dir}/{local_json_folder_name}/{exp_ID}__experiments.json`. Since no LSF job id
can be used, the filename may not be unique (even though the probability of clashes is low).
Concatenating the results of several runs can be done with running `concatenate_jsons.py`.


# Notes on running Ray submit script (not used in experiments)

**Note:** We have not been using Ray since it led to problems when launching
multiple sets of experiments. 

Running ray-based parallel script on the LSF platform
```
bsub -W {time limit for ray head node} -R "rusage[mem=20000,ngpus_excl_p=1]" -R "select[gpu_model1==GeForceGTX1080Ti]"
"bash {path/to/codefolder/}submit_parallel.sh -r -n 20 ”
```
Flags: `-r` to run ray, `-l` to use a limit for number of requested GPUs
Options: `-m` memory, `-n` max number of GPUs (automatically sets `-l` flag), `-t` time limit, `-c` config script, `-g` # gpus per job

Make sure that time limit for ray head node is at least as long as the option value for `-t`
The script `submit_parallel.sh` calls `ray_cluster.sh` which connects the nodes.

# Notes on code that might become relevant

Standardization is done for each batch. This means there could be a problem
when group size is increased so that the mean might not be meaningful anymore.


# Current construction sites

Use `train_stn.py` and `configs/config_stn.json` only in conjunction with each other.
```
python train_stn.py --config configs/config_stn.json
```
Changes in the evaluation and output part of `train_stn.py` will be transfered
to `train.py` later today (19th)