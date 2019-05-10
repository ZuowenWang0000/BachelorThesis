#!/bin/bash -l

bsub -o test1core.txt -W 24:00 -R "rusage[mem=9000,ngpus_excl_p=1]" -R "select[gpu_model1==GeForceGTX1080Ti]"  < submit_core.sh
