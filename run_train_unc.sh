#!/bin/bash
# a file for job output, you can check job progress
#SBATCH --output=/ubc/cs/research/shield/projects/aditya10/RefSeg/results/unc/18/train.out

# a file for errors
#SBATCH --error=/ubc/cs/research/shield/projects/aditya10/RefSeg/results/unc/18/train.err
 
# gpus per node
#SBATCH --gres=gpu:4
 
# number of requested nodes
#SBATCH --nodes=1
 
# memory per node
#SBATCH --mem=8192
#SBATCH --job-name=RefSeg
#SBATCH --cpus-per-task=4
#SBATCH --partition=edith
#SBATCH --time=5-10:00:00

pip2 list
python2 -u /ubc/cs/research/shield/projects/aditya10/RefSeg/trainval_model.py -m train -d unc -t train -n RS0 -emb -f /ubc/cs/research/shield/projects/aditya10/RefSeg/ckpts/unc/18