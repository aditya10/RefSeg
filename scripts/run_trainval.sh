#!/bin/bash
# a file for job output, you can check job progress
#SBATCH --output=/ubc/cs/research/shield/projects/aditya10/RefSeg/results/unc+/6/unc+_trainval.out

# a file for errors
#SBATCH --error=/ubc/cs/research/shield/projects/aditya10/RefSeg/results/unc+/6/unc+_trainval.err
 
# gpus per node
#SBATCH --gres=gpu:4
 
# number of requested nodes
#SBATCH --nodes=1
 
# memory per node
#SBATCH --mem=8192
#SBATCH --job-name=RefSeg
#SBATCH --cpus-per-task=4
#SBATCH --partition=edith
#SBATCH --time=7-10:00:00

pip2 list
python2 -u /ubc/cs/research/shield/projects/aditya10/RefSeg/trainval_model.py -m train -d unc+ -t train -n CMPC_model_graphmod_dup_loss -emb -f /ubc/cs/research/shield/projects/aditya10/RefSeg/ckpts/unc+/6
python2 -u /ubc/cs/research/shield/projects/aditya10/RefSeg/trainval_model.py -m test -d unc+ -t val -n CMPC_model_graphmod_dup_loss -i 700000 -c -emb -f /ubc/cs/research/shield/projects/aditya10/RefSeg/ckpts/unc+/6
python2 -u /ubc/cs/research/shield/projects/aditya10/RefSeg/trainval_model.py -m test -d unc+ -t testA -n CMPC_model_graphmod_dup_loss -i 700000 -c -emb -f /ubc/cs/research/shield/projects/aditya10/RefSeg/ckpts/unc+/6
python2 -u /ubc/cs/research/shield/projects/aditya10/RefSeg/trainval_model.py -m test -d unc+ -t testB -n CMPC_model_graphmod_dup_loss -i 700000 -c -emb -f /ubc/cs/research/shield/projects/aditya10/RefSeg/ckpts/unc+/6