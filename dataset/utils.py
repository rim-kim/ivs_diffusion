import math 
from typing import Literal, Tuple, Optional, List

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor, Lambda

from dataset.imagenet_classes import classes


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

def get_captions(targets: torch.Tensor) -> List[str]:
    """
    Generates captions based on target class indices.

    :param targets: A tensor containing target class indices.
    :return: A list of caption strings corresponding to the target classes.
    """
    class_strings = [preprocess_caption(classes[tgt]) for tgt in targets.tolist()]
    return class_strings 

def preprocess_caption(label: str) -> str:
    """
    Preprocesses a label to generate a descriptive caption.

    :param label: The input label string.
    :return: A caption string describing the label.
    """
    first_label = label.split(",")[0]
    caption = f"a photo of a {first_label}"
    return caption
