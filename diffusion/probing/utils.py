import argparse
from typing import Literal, Tuple, Optional
import math 
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor, Lambda
from torch.utils.data import DataLoader, Subset

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the script.

    :return: Parsed arguments as a Namespace object.
    """
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
        '--feat_output_dir', type=str, default="data/features/",
        help='dir where the internal representations will be stored'
    )   
    parser.add_argument(
        '--lr', type=float, default=1e-3,
    )
    parser.add_argument(
        '--epochs', type=int, default=30,
    )
    parser.add_argument(
        '--batch_size', type=int, default=64,
    )
    parser.add_argument(
        '--eval_interval', type=int, default=10,
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

def precompute_dataset_len(batch_size: int, split: Literal["train", "val"] = "train") -> int:
    """
    Calculates the number of batches for a dataset split.

    :param batch_size: The size of each batch.
    :param split: Dataset split, either "train" or "val". Defaults to "train".
    :return: The number of batches.
    """
    total_samples = 1281167 if split == "train" else 50000
    num_batches =  math.ceil(total_samples / batch_size)
    return num_batches

def get_toy_data(batch_size: int, samples: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
    """
    Loads a toy dataset (CIFAR-10) and applies specified transformations.

    :param batch_size: The size of each batch.
    :param samples: Number of samples to load for training and validation. Defaults to twice the batch size.
    :return: Data loaders for training and validation datasets.
    """
    if not samples:
        train_samples = batch_size * 4
        val_samples = batch_size * 2
    transformations = Compose([
        Resize((256, 256)),
        ToTensor(),
        Lambda(lambda x: x * 2 - 1)
    ])
    toy_train_data = datasets.CIFAR10(
        root="./data/toy_data", train=True, download=True, transform=transformations
    )
    toy_val_data = datasets.CIFAR10(
        root="./data/toy_data", train=False, download=True, transform=transformations
    )
    train_data = Subset(toy_train_data, list(range(train_samples)))
    val_data = Subset(toy_val_data, list(range(val_samples)))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader