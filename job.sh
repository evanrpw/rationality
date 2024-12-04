#!/bin/bash

#SBATCH --job-name=erpw
#SBATCH --partition=gpu
#SBATCH --time=0:10:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access

module purge
module add anaconda
module add cuda

conda activate env

python ~/rationality/main.py
