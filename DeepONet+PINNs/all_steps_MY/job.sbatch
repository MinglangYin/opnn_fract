#!/bin/bash
#SBATCH -N 1
#BATCH -n 4
#SBATCH -p gpu 
#SBATCH --gres=gpu:1
#SBATCH -t 10:00:00
#SBATCH --mem=32G
##SBATCH --constraint=quadrortx
#SBATCH -A gk-condo
#SBATCH -o pyt_Job-%j.out
#SBATCH -e pyt_Job-%j.err

module load python/3.7.4 cuda/11.1.1 cudnn/8.1.0
#export PATH=/users/sgoswam5/apps/pyt_env_r/bin:$PATH
#export PYTHONPATH=/users/sgoswam5/apps/pyt_env_r/lib/python3.7/site-packages:$PYTHONPATH

source /users/sgoswam5/apps/ccv_venv/bin/activate

python3 -u main.py --epochs 100
