#!/bin/bash
#SBATCH --partition=Abaki
#SBATCH --qos=abaki
#SBATCH --job-name=ivs-data
#SBATCH --comment="Latent data generation"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jeongrim.kim@campus.lmu.de
#SBATCH --chdir=/home/k/kimje/IVS/ivs_diffusion
#SBATCH --output=/home/k/kimje/IVS/ivs_diffusion/slurm_logs/%x_%j_%N.out
#SBATCH --ntasks=1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ivs

python -m dataset.utils \
    --output_path "/home/k/kimje/IVS/ivs_diffusion/data/imagenet_latent"
