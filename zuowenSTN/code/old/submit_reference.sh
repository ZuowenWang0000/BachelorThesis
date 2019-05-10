#!/bin/bash -l

module load python_gpu/3.6.4
cd reference/adversarial_spatial-master/
python train_timed.py

