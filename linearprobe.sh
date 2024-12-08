#!/bin/bash
#SBATCH --job-name=ivs-probe
#SBATCH --comment="IVS Linear Probe"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jeongrim.kim@campus.lmu.de
#SBATCH --chdir=/home/k/kimje/IVS/ivs_diffusion
#SBATCH --output=/home/k/kimje/IVS/ivs_diffusion/slurm_logs/%x_%j_%N.out
#SBATCH --ntasks=1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ivs

python -u linearprobe.py \
    --data_dir "/home/k/kimje/IVS/data/imagenet" \
    --cfg_path "configs/model/rf_dit_unclip.yaml" \
    --ckpt_pth "diffusion/unclip.pt" \
    --timestep 0.75 \
    --layer_num 13 \
    --output_dir "output" \
    --model_name "unclip"\
    --caption False