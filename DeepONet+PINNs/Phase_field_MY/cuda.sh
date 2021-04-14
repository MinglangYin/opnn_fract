#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1

###SBATCH -A gk-condo
###SBATCH -p gpu
###SBATCH --gres=gpu:1

# Request 1 CPU core
#SBATCH -n 1
#SBATCH --mem=12G
#SBATCH -t 10:30:00
#SBATCH -o MyCUDAJob-%j.out
#SBATCH --exclude=gpu1204

# Compile CUDA program and run
~/Environments/PyTorch-GPU/bin/python main.py --epochs 50000
