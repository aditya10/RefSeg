#!/bin/bash
# a file for job output, you can check job progress
#SBATCH --output=/ubc/cs/research/shield/projects/aditya10/RefSeg/results/Gref/5/Gref_train.out

# a file for errors
#SBATCH --error=/ubc/cs/research/shield/projects/aditya10/RefSeg/results/Gref/5/Gref_train.err
 
# gpus per node
#SBATCH --gres=gpu:8
 
# number of requested nodes
#SBATCH --nodes=1
 
# memory per node
#SBATCH --mem=8192
#SBATCH --job-name=RefSeg
#SBATCH --cpus-per-task=4
#SBATCH --partition=edith
#SBATCH --time=5-10:00:00

pip2 list
python2 -u /ubc/cs/research/shield/projects/aditya10/RefSeg/trainval_model.py -m train -d Gref -t train -n CMPC_model_graphmod_duplicated -emb -f /ubc/cs/research/shield/projects/aditya10/RefSeg/ckpts/Gref/5