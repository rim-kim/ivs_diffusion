import io
import os
import random

import numpy as np
import torch

import webdataset as wds


class LatentDatasetLoader:
    """
    A class to handle loading of latent datasets for training and validation.
    Provides methods to create dataloaders for training and validation workflows.
    """
    def __init__(self, train_shard_path, val_shard_path, batch_size, seed=42, verbose=False):
        self.train_shard_path = train_shard_path
        self.val_shard_path = val_shard_path
        self.batch_size = batch_size
        self.seed = seed

        # If verbose, prints shard info
        os.environ["GOPEN_VERBOSE"] = "1" if verbose else "0"

        # Set deterministic behavior for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Get the number of shard files from local path
        self.train_shards = len([f for f in os.listdir(train_shard_path) 
                                 if f.startswith("latent-") and f.endswith(".tar")])
        self.val_shards = len([f for f in os.listdir(val_shard_path) 
                                 if f.startswith("latent-") and f.endswith(".tar")])
        
    def worker_init_fn(self, worker_id):
        """
        Initialize each worker with a unique seed for reproducibility.
        """
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def _call_dataloader(self, is_training=True, is_reduced=False):
        """
        Create a dataloader for given split.
        Shuffle is enabled only in training mode.
        """
        # Deactivate shuffling if not in training mode
        if is_training:
            shardshuffle=True
            shuffle=1000
            dataset_url = f"{self.train_shard_path}/latent-{{0000..{self.train_shards-1}}}.tar"
        else:
            shardshuffle=False
            shuffle=0
            dataset_url = f"{self.val_shard_path}/latent-{{0000..00{self.val_shards-1}}}.tar"

        def make_sample(sample):
            latent = torch.load(io.BytesIO(sample['latent.pth']))
            label = int(sample['cls.txt'])
            return latent, label
        
        # Load the dataset
        dataset = wds.WebDataset(
            dataset_url, 
            resampled=True,
            shardshuffle=shardshuffle,
            nodesplitter=wds.split_by_node,
            detshuffle=True,
            seed=self.seed
        ).shuffle(shuffle).map(make_sample)

        # For training, rebatch after unbatching for more thorough mixing
        dataset = dataset.batched(self.batch_size)
        if is_training:
            dataloader = wds.WebLoader(dataset, batch_size=None, num_workers=4, worker_init_fn=self.worker_init_fn)
            dataloader = dataloader.unbatched().shuffle(1000).batched(self.batch_size)
            dataloader = dataloader.with_epoch(1281167 // self.batch_size)
        else:
            # Reduce validation set to 10k
            if is_reduced:
                dataloader = wds.WebLoader(dataset.unbatched().shuffle(10000).slice(10000), batch_size=None)
                dataloader = dataloader.batched(self.batch_size)
                dataloader = dataloader.with_epoch(10000 // self.batch_size)
            else:
                # For validation mode, no unbatching/rebatching or extra shuffle
                dataloader = wds.WebLoader(dataset, batch_size=None)
                dataloader = dataloader.with_epoch(50000 // self.batch_size)
        return dataloader

    def make_dataloader(self, split="train", is_reduced=False):
        """
        Make a dataloader for training or validation based on the split parameter.
        """
        if split == "train":
            return self._call_dataloader()
        elif split == "val":
            return self._call_dataloader(is_training=False, is_reduced=is_reduced)
        else:
            raise ValueError(f"Unknown split '{split}' specified. Use 'train' or 'val'.")
