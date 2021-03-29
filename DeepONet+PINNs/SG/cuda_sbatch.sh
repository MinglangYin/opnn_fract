#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu 
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH --mem=12G
#SBATCH -t 10:00:00
#SBATCH --mem=8G
##SBATCH --constraint=quadrortx
#SBATCH -A gk-condo
#SBATCH -o MyCUDAJob-%j.out
#SBATCH --exclude=gpu1204

module load cuda/10.2
module load cudnn/7.6.5
module load python/3.7.4

# Compile CUDA program and run

source ~/apps/pyt_env/bin/activate

python3 -u main.py --epochs 100000 

