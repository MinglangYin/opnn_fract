#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1

###SBATCH -p gk-gpu-condo

# Request 1 CPU core
#SBATCH -n 1
#SBATCH --mem=12G
#SBATCH -t 100:30:00
#SBATCH --mem=8G
#SBATCH -o MyCUDAJob-%j.out
#SBATCH --exclude=gpu1204

# Compile CUDA program and run
~/Environments/PyTorch-GPU/bin/python main.py --epochs 2000 --batch-size 500000
