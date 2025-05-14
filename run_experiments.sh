#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=student
#SBATCH --cpus-per-task=4

cd $HOME/diffusion
source activate diffusion_density
python -u run_experiments.py