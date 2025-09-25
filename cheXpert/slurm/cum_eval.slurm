#!/bin/bash
#SBATCH --gres=gpu:8
#SBATCH --nodes=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=36:00:00
#SBATCH --job-name=resnet-cifar10
#SBATCH --output=./slurm_output/%j.out
#SBATCH --mail-user=zmou1@jh.edu
#SBATCH --mail-type=END,FAIL


nvidia-smi

python3 /home/zmou1/scratchenalisn1/ziyao/l2d-cog/l2d_project/scripts/generate_cumulative_acc.py

