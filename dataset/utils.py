import argparse
import io
import math
from typing import Literal, Tuple, Optional, List
import zlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor, Lambda
from tqdm import tqdm
from webdataset import TarWriter
from webdataset.compat import WebLoader

from utils.logging import logger
from dataset.imagenet_classes import classes
from dataset.dataset_loaders import DatasetLoader
from diffusion.model.modules.ae import AutoencoderKL
from diffusion.model.modules.clip import ClipImgEmbedder
from configs.tokens.tokens import HF_TOKEN
from configs.hyperparameters.hyperparameters import DEVICE
from configs.path_configs.path_configs import TOY_DATA_DIR, TRAIN_LATENT_AND_CLIP_DIR, VAL_LATENT_AND_CLIP_DIR

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
        root=TOY_DATA_DIR, train=True, download=True, transform=transformations
    )
    toy_val_data = datasets.CIFAR10(
        root=TOY_DATA_DIR, train=False, download=True, transform=transformations
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

def compute_latent_and_clip_representations(
    ae_model: AutoencoderKL,
    clip_model: ClipImgEmbedder,
    dataloader: WebLoader,
    output_path: Path,
    samples_per_shard: int,
    device: str = DEVICE
) -> None:
    """
    Compute and save latent representations and CLIP embeddings from the models for each image in the dataset.

    :param ae_model: The autoencoder model used to encode images into latent representations.
    :param clip_model: The CLIP image embedding model used to generate image embeddings.
    :param dataloader: The dataloader for iterating over the dataset images and labels.
    :param output_path: The directory path where latent shards are stored.
    :param samples_per_shard: The number of samples to store per shard file.
    :param device: The device on which the model and data will be processed (default: "cuda").
    """
    logger.info(f"Starting computation of latent and CLIP representations. Output path: '{output_path}', Samples per shard: {samples_per_shard}.")
    sample_id, shard_id = 0, 0
    shard_writer = None
    shard_sample_count = 0

    try:
        # Iterate over batches of images and targets
        for imgs, targets in tqdm(iterable=dataloader, total=dataloader.nsamples):
            logger.debug("Processing a new batch of images...")
            imgs, targets = imgs.to(device), targets.to(device)

            with torch.no_grad():
                # Encode images using the autoencoder and generate CLIP embeddings
                latents = ae_model.encode(imgs).cpu()
                targets = targets.cpu().numpy()
                clip_embs = clip_model(imgs).cpu()

            for latent, label, clip_emb in tqdm(iterable=zip(latents, targets, clip_embs), total=len(latents), leave=None):
                # Create a new shard after reaching `samples_per_shard` limit
                if sample_id % samples_per_shard == 0:
                    if shard_writer:
                        shard_writer.close()
                        logger.info(f"Closed shard {shard_id - 1} with {shard_sample_count} samples.")
                    
                    shard_writer = TarWriter(f"{output_path}/latent-{shard_id:04d}.tar")
                    logger.info(f"Started new shard {shard_id}.")
                    shard_id += 1
                    shard_sample_count = 0

                # Serialize numpy latent and CLIP embeddings into in-memory buffers
                latent_buffer = io.BytesIO()
                clip_emb_buffer = io.BytesIO()

                # Save arrays to buffer
                np.save(latent_buffer, latent.numpy())
                np.save(clip_emb_buffer, clip_emb.numpy())

                # Reset buffer positions
                latent_buffer.seek(0)
                clip_emb_buffer.seek(0)

                # Compress latent and embedding buffers
                compressed_latent = zlib.compress(latent_buffer.read())
                compressed_clip_emb = zlib.compress(clip_emb_buffer.read())

                # Write serialized data to the shard
                shard_writer.write({
                    '__key__': f"{sample_id:07d}",
                    'latent.npy.zlib': compressed_latent,
                    'cls.txt': str(label),
                    'clip_emb.npy.zlib': compressed_clip_emb,
                })
                logger.debug(f"Sample {sample_id} written to shard {shard_id - 1}.")
                sample_id += 1
                shard_sample_count += 1

    except Exception as e:
        logger.error(f"An error occurred during computation: {e}")
        raise

    finally:
        if shard_writer:
            shard_writer.close()
            logger.info(f"Closed final shard {shard_id - 1} with {shard_sample_count} samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--samples_per_shard", type=int, default=1000)
    parser.add_argument("--dataset", type=str, choices=["train", "val", "both"], default="both",
                        help="Choose the dataset to process: 'train', 'val', or 'both'.")
    args = parser.parse_args()
    
    handler = DatasetLoader(
        hf_token=HF_TOKEN,
        batch_size=args.batch_size,
        streaming=False,
    )
    ae_model = AutoencoderKL()
    ae_model.eval().to(DEVICE)
    clip_model = ClipImgEmbedder()
    clip_model.eval().to(DEVICE)

    # Process based on the selected dataset
    if args.dataset in ["train", "both"]:
        train_dataloader = handler.get_dataloader(split="train")
        logger.info("Computing latents and CLIP embeddings on training dataset...")
        compute_latent_and_clip_representations(ae_model, clip_model, train_dataloader, TRAIN_LATENT_AND_CLIP_DIR, args.samples_per_shard)
    
    if args.dataset in ["val", "both"]:
        test_dataloader = handler.get_dataloader(split="val")
        logger.info("Computing latents and CLIP embeddings on validation dataset...")
        compute_latent_and_clip_representations(ae_model, clip_model, test_dataloader, VAL_LATENT_AND_CLIP_DIR, args.samples_per_shard)
