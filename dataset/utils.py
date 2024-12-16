import argparse
import io
import math
import os
from typing import Literal, Tuple, Optional, List
import zlib

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor, Lambda
from tqdm import tqdm
from webdataset import TarWriter

from dataset.imagenet_classes import classes
from dataset.dataset_preprocessing import DatasetLoader
from diffusion.model.modules.ae import AutoencoderKL
from configs.tokens.tokens import HF_TOKEN


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

def compute_latent_dataset(model, dataloader, output_path, samples_per_shard, device='cuda'):
    """
    Computes and collect latent from model.

    :param model: The model to get latent from.
    :param dataloder: The dataloader to iterate over.
    :param output_path: The path of directory to store latent.
    :param samples_per_shard: The number of samples per shard file.
    :param device: The device of the model and data. 'cuda' as default.
    """
    os.makedirs(output_path, exist_ok=True)
    sample_id, shard_id = 0, 0
    shard_writer = None
    
    for _, (imgs, targets) in enumerate(tqdm(iterable=dataloader,total=dataloader.nsamples)):
        imgs, targets = imgs.to(device), targets.to(device)
        with torch.no_grad():
            # convert to cpu to write .tar
            latents = model.encode(imgs).cpu()
            targets = targets.cpu().numpy()

        for latent, label in zip(latents, targets):
            # create new shard for every samples_per_shard
            if sample_id % samples_per_shard == 0:
                if shard_writer:
                    shard_writer.close()
                shard_writer = TarWriter(f"{output_path}/latent-{shard_id:04d}.tar")
                shard_id += 1

            # serialize numpy latent in buffer
            latent_buffer = io.BytesIO()
            np.save(latent_buffer, latent.numpy())
            latent_buffer.seek(0)
            # compress npy
            compressed_latent = zlib.compress(latent_buffer.read())

            # write serialized latent into the shard
            shard_writer.write({
                '__key__': f"{sample_id:07d}",
                'latent.npy.zlib': compressed_latent,
                'cls.txt': str(label)
            })
            sample_id += 1

    if shard_writer:
        shard_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--samples_per_shard", type=int, default=1000)
    args = parser.parse_args()
    handler = DatasetLoader(
        hf_token=HF_TOKEN,
        cache_dir=args.data_path,
        preprocessed_data_dir="",
        batch_size=64,
        epochs=1,
    )
    # train_dataloader = handler.make_dataloader(split="train")
    test_dataloader = handler.make_dataloader(split="val")
    model = AutoencoderKL()
    model.eval().to('cuda')
    # compute_latent_dataset(model, train_dataloader, f"{args.output_path}/train", args.samples_per_shard)
    compute_latent_dataset(model, test_dataloader, f"{args.output_path}/val", args.samples_per_shard)
    