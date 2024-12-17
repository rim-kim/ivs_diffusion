import os
import random
from typing import Optional

import numpy as np
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Lambda
import webdataset as wds


class DatasetLoader:
    """
    A class to handle preprocessing and loading of datasets for training and validation.
    Provides methods to create dataloaders for training and validation workflows.
    """

    def __init__(
        self, 
        hf_token: str, 
        batch_size: int, 
        streaming: bool = False, 
        seed: int = 42, 
        verbose: bool = False
    ) -> None:
        """
        Initialize the DatasetLoader class.
        
        :param hf_token: The HuggingFace token for dataset access.
        :param batch_size: The batch size for the dataloaders.
        :param streaming: Whether to stream data or cache it locally.
        :param seed: Random seed for reproducibility.
        :param verbose: If True, enables verbose logging for shard info.
        """
        self.hf_token: str = hf_token
        self.batch_size: int = batch_size
        self.seed: int = seed
        self.cache_dir: Optional[str] = None if streaming else "data/imagenet"
        self.preprocessed_data_dir: str = "data/preprocessed_data"
        self.train_shards: int = 1024
        self.val_shards: int = 64

        os.environ["GOPEN_VERBOSE"] = "1" if verbose else "0"

        # Set deterministic behavior for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.trainset_url: str = (
            "pipe:curl -s -L "
            "https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main/"
            "imagenet1k-train-{{0000..{shards}}}.tar -H 'Authorization:Bearer {hf_token}'"
        )
        self.valset_url: str = (
            "pipe:curl -s -L "
            "https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main/"
            "imagenet1k-validation-{{00..{shards}}}.tar -H 'Authorization:Bearer {hf_token}'"
        )

    @staticmethod
    def preprocessing_transform() -> Compose:
        """
        Return the preprocessing transform to apply to dataset images.
        Scales images to 256x256 and normalizes them to [-1, 1].
        
        :return: A torchvision Compose transform for image preprocessing.
        """
        return Compose([
            Resize((256, 256)),
            ToTensor(),
            Lambda(lambda x: x * 2 - 1)
        ])

    def worker_init_fn(self, worker_id: int) -> None:
        """
        Initialize each worker with a unique seed for reproducibility.
        
        :param worker_id: ID of the worker process.
        """
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def _call_dataloader(
        self, 
        dataset_url: str, 
        is_training: bool = True, 
        is_reduced: bool = False
    ) -> wds.WebLoader:
        """
        Create a dataloader for the given dataset URL.
        
        :param dataset_url: URL to the dataset shards.
        :param is_training: Whether to configure the dataloader for training.
        :param is_reduced: If True, reduces the size of the validation set to 10k.
        :return: A WebDataset-based dataloader for the specified split.
        """
        shardshuffle = is_training
        shuffle = 1000 if is_training else 0
        transform = self.preprocessing_transform()

        def make_sample(sample: dict) -> tuple:
            return transform(sample["jpg"]), sample["cls"]
        # Load the dataset
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
        dataset = wds.WebDataset(
            dataset_url, 
            resampled=True,
            shardshuffle=shardshuffle,
            cache_dir=self.cache_dir, 
            nodesplitter=wds.split_by_node,
            detshuffle=True,
            seed=self.seed
        ).shuffle(shuffle).decode("pil").map(make_sample)
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

    def make_dataloader(self, split: str = "train", is_reduced: bool = False) -> wds.WebLoader:
        """
        Make a dataloader for training or validation based on the split parameter.
        
        :param split: The dataset split to use ('train' or 'val').
        :param is_reduced: If True, reduces validation set size.
        :return: A WebDataset-based dataloader for the specified split.
        :raises ValueError: If an unknown split is specified.
        """
        if split == "train":
            return self._call_dataloader(self.trainset_url.format(shards=self.train_shards-1, hf_token=self.hf_token))
        elif split == "val":
            return self._call_dataloader(
                self.valset_url.format(shards=self.val_shards-1, hf_token=self.hf_token),
                is_training=False, 
                is_reduced=is_reduced
            )
        else:
            raise ValueError(f"Unknown split '{split}' specified. Use 'train' or 'val'.")

    def precompute_and_cache(self, dataloader: wds.WebLoader) -> None:
        """
        Precompute and cache the preprocessed tensors for the dataset.
        
        :param dataloader: A WebDataset dataloader to process and cache.
        """
        os.makedirs(self.preprocessed_data_dir, exist_ok=True)
        for batch_idx, (images, labels) in enumerate(dataloader):
            torch.save((images, labels), os.path.join(self.preprocessed_data_dir, f"batch_{batch_idx}.pt"))
            print(f"Saved batch {batch_idx} to {self.preprocessed_data_dir}")
