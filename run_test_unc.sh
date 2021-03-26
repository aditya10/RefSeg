#!/bin/bash
# a file for job output, you can check job progress
#SBATCH --output=/ubc/cs/research/shield/projects/aditya10/RefSeg/results/unc/9/unc_testB_800.out
 
# a file for errors
#SBATCH --error=/ubc/cs/research/shield/projects/aditya10/RefSeg/results/unc/9/unc_testB_800.out
 
# gpus per node
#SBATCH --gres=gpu:1
 
# number of requested nodes
#SBATCH --nodes=1
 
# memory per node
#SBATCH --mem=8192
#SBATCH --job-name=RefSeg
#SBATCH --cpus-per-task=4
#SBATCH --partition=edith
#SBATCH --time=5-10:00:00

python2 -u /ubc/cs/research/shield/projects/aditya10/RefSeg/trainval_model.py -m test2 -d unc -t testB -n CMPC_model_graphmod_3 -i 800000 -c -emb -f /ubc/cs/research/shield/projects/aditya10/RefSeg/ckpts/unc/9
