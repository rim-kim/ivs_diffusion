from abc import ABC, abstractmethod
import os
import random
import numpy as np
import torch
import webdataset as wds
from pathlib import Path
from torchvision.transforms import Compose, Resize, ToTensor, Lambda
import io
import zlib
from typing import Optional, Literal, Tuple, Dict

from configs.path_configs.path_configs import IMAGENET_SHARDS_DIR, PREPROCESSED_IMAGENET_DIR, TRAIN_LATENT_AND_CLIP_DIR, VAL_LATENT_AND_CLIP_DIR


class BaseDatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.
    """

    def __init__(self, batch_size: int, seed: int = 42, verbose: bool = False) -> None:
        """
        Initialize the BaseDatasetLoader class.

        :param batch_size: The batch size for the dataloaders.
        :param seed: Random seed for reproducibility.
        :param verbose: If True, enables verbose logging for shard info.
        """
        self.batch_size: int = batch_size
        self.seed: int = seed
        os.environ["GOPEN_VERBOSE"] = "1" if verbose else "0"

        # Set deterministic behavior for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def worker_init_fn(self, worker_id: int) -> None:
        """
        Initialize each worker with a unique seed for reproducibility.

        :param worker_id: ID of the worker process.
        """
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    @abstractmethod
    def process_sample(self, sample: Dict) -> Tuple[torch.Tensor, int]:
        """
        Abstract method for creating a data sample from the dataset.
        
        :param sample: A dictionary containing the data sample.
        :return: A tuple containing the transformed sample and label.
        """
        pass

    @abstractmethod
    def get_dataset_path(self, split: Literal["train", "val"]) -> str:
        """
        Abstract method for constructing the dataset URL or local path based on the split.

        :param split: The dataset split ("train" or "val").
        :return: The dataset URL or path.
        """
        pass

    def _construct_dataloader(
        self, 
        split: Literal["train", "val"],
        is_training: bool = True, 
        is_reduced: bool = False
    ) -> wds.WebLoader:
        """
        Construct a dataloader for the given dataset URL or path.

        :param split: The dataset split ("train" or "val").
        :param is_training: Whether to configure the dataloader for training.
        :param is_reduced: If True, reduces the size of the validation set to 10k samples.
        :return: A WebDataset-based dataloader for the specified split.
        """
        dataset_path = self.get_dataset_path(split)
        shardshuffle = is_training
        shuffle = 1000 if is_training else 0

        # Load and shuffle the dataset
        dataset_args = {
            "resampled": True,
            "shardshuffle": shardshuffle,
            "nodesplitter": wds.split_by_node,
            "detshuffle": True,
            "seed": self.seed,
        }
        if hasattr(self, "cache_dir"):
            dataset_args["cache_dir"] = self.cache_dir
        dataset = wds.WebDataset(dataset_path, **dataset_args).shuffle(shuffle)

        # Only apply decoding if decoding_mode is set
        if hasattr(self, "decoding_mode"):
            dataset = dataset.decode(self.decoding_mode)

        # Process and batch dataset samples
        dataset = dataset.map(self.process_sample).batched(self.batch_size)

        if is_training:
            # For training, rebatch after unbatching for more thorough mixing
            dataloader = wds.WebLoader(dataset, batch_size=None, num_workers=4, worker_init_fn=self.worker_init_fn)
            dataloader = dataloader.unbatched().shuffle(shuffle).batched(self.batch_size).with_epoch(1281167 // self.batch_size)
        else:
            # For validation, no unbatching/rebatching or extra shuffle
            if is_reduced:
                # Reduce validation set to 10k
                dataloader = wds.WebLoader(dataset.slice(10000), batch_size=None).with_epoch(10000 // self.batch_size)
            else:
                dataloader = wds.WebLoader(dataset, batch_size=None).with_epoch(50000 // self.batch_size)
        return dataloader

    def get_dataloader(self, split: Literal["train", "val"] = "train", is_reduced: bool = False) -> wds.WebLoader:
        """
        Get the dataloader for training or validation based on the split parameter.

        :param split: The dataset split to use ('train' or 'val').
        :param is_reduced: If True, reduces validation set size to 10k samples.
        :return: A WebDataset-based dataloader for the specified split.
        :raises ValueError: If an unknown split is specified.
        """
        if split not in {"train", "val"}:
            raise ValueError(f"Unknown split '{split}' specified. Use 'train' or 'val'.")
        return self._construct_dataloader(split, is_training=(split == "train"), is_reduced=is_reduced)


class DatasetLoader(BaseDatasetLoader):
    """
    Image dataset loader inheriting from BaseDatasetLoader.
    Handles ImageNet dataset loading from HuggingFace repository.
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

        Attributes:
            cache_dir (Path): Directory path for caching dataset shards if streaming is disabled.
            decoding_mode (str): The mode to decode image files into.
            preprocessed_data_dir (Path): Directory path for storing preprocessed ImageNet data.
            train_shards (int): Number of shards for the training dataset. Defaults to 1024.
            val_shards (int): Number of shards for the validation dataset. Defaults to 64.
        """
        super().__init__(batch_size, seed, verbose)
        self.hf_token: str = hf_token
        self.cache_dir: Optional[Path] = None if streaming else IMAGENET_SHARDS_DIR
        self.decoding_mode: str = "pil"
        self.preprocessed_data_dir: Path = PREPROCESSED_IMAGENET_DIR
        self.train_shards: int = 1024
        self.val_shards: int = 64

    def process_sample(self, sample: Dict) -> Tuple[torch.Tensor, int]:
        """
        Preprocess image samples: resize, normalize, and convert to tensor.

        :param sample: A dictionary containing image and label data.
        :return: A tuple containing the processed image tensor and its class label.
        """
        transform = Compose([
            Resize((256, 256)),
            ToTensor(),
            Lambda(lambda x: x * 2 - 1)
        ])
        return transform(sample["jpg"]), sample["cls"]

    def get_dataset_path(self, split: Literal["train", "val"]) -> str:
        """
        Construct dataset URL based on the split.

        :param split: The dataset split ("train" or "val").
        :return: The dataset URL.
        """
        if split == "train":
            return (
                "pipe:curl -s -L "
                f"https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main/"
                f"imagenet1k-train-{{0000..{self.train_shards - 1}}}.tar -H 'Authorization:Bearer {self.hf_token}'"
            )
        else:
            return (
                "pipe:curl -s -L "
                f"https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main/"
                f"imagenet1k-validation-{{00..{self.val_shards - 1}}}.tar -H 'Authorization:Bearer {self.hf_token}'"
            )
    
    def precompute_and_cache(self, dataloader: wds.WebLoader) -> None:
        """
        Precompute and cache the preprocessed tensors for the dataset.

        :param dataloader: A WebDataset dataloader to process and cache.
        """
        for batch_idx, (images, labels) in enumerate(dataloader):
            torch.save((images, labels), self.preprocessed_data_dir / f"batch_{batch_idx}.pt")
            print(f"Saved batch {batch_idx} to {self.preprocessed_data_dir}")


class LatentDatasetLoader(BaseDatasetLoader):
    """
    Latent dataset loader inheriting from BaseDatasetLoader.
    Handles loading of precomputed latent vectors.
    """

    def __init__(self, batch_size: int, seed: int = 42, verbose: bool = False) -> None:
        """
        Initialize the LatentDatasetLoader class.

        :param batch_size: The batch size for the dataloaders.
        :param seed: Random seed for reproducibility.
        :param verbose: If True, enables verbose logging for shard information.

        Attributes:
            train_shard_path (Path): Directory path containing training latent shards.
            val_shard_path (Path): Directory path containing validation latent shards.
            train_shards (int): Number of training latent shards available.
            val_shards (int): Number of validation latent shards available.
        """
        super().__init__(batch_size, seed, verbose)
        self.train_shard_path: Path = TRAIN_LATENT_AND_CLIP_DIR
        self.val_shard_path: Path = VAL_LATENT_AND_CLIP_DIR
        self.train_shards: int = len(list(self.train_shard_path.glob("latent-*.tar")))
        self.val_shards: int = len(list(self.val_shard_path.glob("latent-*.tar")))

    def process_sample(self, sample: Dict[str, bytes]) -> Tuple[torch.Tensor, int]:
        """
        Preprocess latent samples: decompress and convert to tensor.

        :param sample: A dictionary containing latent vector data and class label.
        :return: A tuple containing the latent tensor and its corresponding class label.
        """
        decompressed_latent = zlib.decompress(sample['latent.npy.zlib'])
        decompressed_clip_emb = zlib.decompress(sample['clip_emb.npy.zlib'])
        latent = torch.tensor(np.load(io.BytesIO(decompressed_latent)))
        clip_emb = torch.tensor(np.load(io.BytesIO(decompressed_clip_emb)))
        label = int(sample['cls.txt'])
        return latent, label, clip_emb

    def get_dataset_path(self, split: Literal["train", "val"]) -> str:
        """
        Construct dataset local path based on the split.

        :param split: The dataset split ("train" or "val").
        :return: The URL or file path to the latent shards for the specified split.
        """
        if split == "train":
            return f"{self.train_shard_path}/latent-{{0000..{self.train_shards - 1}}}.tar"
        else:
            return f"{self.val_shard_path}/latent-{{0000..{self.val_shards - 1}}}.tar"
