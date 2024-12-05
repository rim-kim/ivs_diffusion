import argparse
from typing import Literal
import math 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cfg_path', type=str, required=True,
        help='path of model config file (.yaml)'
    )
    parser.add_argument(
        '--ckpt_pth', type=str, required=True,
        help='path of model checkpoint'
    )
    parser.add_argument(
        '--timestep', type=float, required=True,
        help='fixed timestep for linear probe (normalized)'
    )
    parser.add_argument(
        '--layer_num', type=int, required=True,
        help='transformer layer number for feature extraction'
    )
    parser.add_argument(
        '--layer_start', type=int, default=14,
        help='starting transformer layer for feature extraction (zero-based index, up to the last layer)'
    )
    parser.add_argument(
    '--feat_output_path', type=str, default="data/features/layer2features.pt",
    help='path where the internal representations will be stored'
    )   
    parser.add_argument(
        '--lr', type=float, default=1e-3,
    )
    parser.add_argument(
        '--epochs', type=int, default=30,
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
    )
    parser.add_argument(
        '--eval_interval', type=int, default=1,
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='path to save train checkpoints'
    )
    parser.add_argument(
        '--model_name', type=str, default="unclip",
        help='name of model for linear probing'
    )

    args = parser.parse_args()
    return args

def precompute_dataset_len(batch_size, split: Literal["train", "val"] = "train"):
    total_samples = 1281167 if split == "train" else 50000
    num_batches =  math.ceil(total_samples / batch_size)
    return num_batches