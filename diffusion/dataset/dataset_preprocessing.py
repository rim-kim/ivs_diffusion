from huggingface_hub import login, get_token
import webdataset as wds
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Lambda
import os

class DatasetLoader:
    """
    A class to handle preprocessing and loading of datasets for training and validation.
    Provides methods to create dataloaders for training and validation workflows.
    """

    def __init__(self, hf_token, cache_dir, preprocessed_data_dir, batch_size, seed=42):
        self.hf_token = hf_token
        self.cache_dir = cache_dir
        self.preprocessed_data_dir = preprocessed_data_dir
        self.batch_size = batch_size
        self.seed = seed

        # Set deterministic behavior for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Define URLs for datasets
        self.trainset_url = f"pipe:curl -s -L https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main/imagenet1k-train-{{0000..1023}}.tar -H 'Authorization:Bearer {self.hf_token}'"
        self.valset_url = f"pipe:curl -s -L https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main/imagenet1k-validation-{{00..63}}.tar -H 'Authorization:Bearer {self.hf_token}'"
    
    @staticmethod
    def preprocessing_transform():
        """
        Return the preprocessing transform to apply to dataset images.
        Scales images to 256x256 and normalizes them to [-1, 1].
        """
        return Compose([
            Resize((256, 256)),
            ToTensor(), 
            Lambda(lambda x: x * 2 - 1)
        ])

    def _call_dataloader(self, dataset_url, is_training=True):
        """
        Create a dataloader for the given dataset URL.
        Shuffle is enabled only if needed (e.g., for training).
        """
        # Activate shuffling by default
        shardshuffle=True
        shuffle=1000

        transform = self.preprocessing_transform()

        def make_sample(sample):
            return transform(sample["jpg"]), sample["cls"]
        
        # Deactivate shuffling if not in training mode
        if not is_training:
            shardshuffle=False
            shuffle=0

        # Load the dataset
        os.makedirs(self.cache_dir, exist_ok=True)
        dataset = wds.WebDataset(
            dataset_url, 
            resampled=True,
            shardshuffle=shardshuffle,
            cache_dir=self.cache_dir, 
            nodesplitter=wds.split_by_node
        ).shuffle(shuffle).decode("pil").map(make_sample)

        # For training, rebatch after unbatching for more thorough mixing
        if is_training:
            dataset = dataset.batched(64)
            dataloader = wds.WebLoader(dataset, batch_size=None, num_workers=4)
            dataloader = dataloader.unbatched().shuffle(1000).batched(self.batch_size)
            dataloader = dataloader.with_epoch(1024 * 100 // 64)  # Fixed epoch length
        else:
            # For validation mode, no unbatching/rebatching or extra shuffle
            dataset = dataset.batched(64)
            dataloader = wds.WebLoader(dataset, batch_size=None, num_workers=4)

        return dataloader

    def make_dataloader(self, split="train"):
        """
        Make a dataloader for training or validation based on the split parameter.
        """
        if split == "train":
            return self._call_dataloader(self.trainset_url)
        elif split == "val":
            return self._call_dataloader(self.valset_url, is_training=False)
        else:
            raise ValueError(f"Unknown split '{split}' specified. Use 'train' or 'val'.")

    def precompute_and_cache(self, dataloader):
        """
        Precompute and cache the preprocessed tensors for the dataset.
        """
        os.makedirs(self.preprocessed_data_dir, exist_ok=True)
        for batch_idx, (images, labels) in enumerate(dataloader):
            torch.save((images, labels), os.path.join(self.preprocessed_data_dir, f"batch_{batch_idx}.pt"))
            print(f"Saved batch {batch_idx} to {self.preprocessed_data_dir}")


if __name__ == "__main__":
    HF_TOKEN = "" # TODO: Add token as secret value
    login(token=HF_TOKEN)

    handler = DatasetLoader(
        hf_token=get_token(),
        cache_dir="../data/shards",
        preprocessed_data_dir="../data/preprocessed_data",
        batch_size=32
    )

    # Get training dataloader
    train_loader = handler.make_dataloader(split="train")

    # for batch_idx, (images, labels) in enumerate(train_loader):
    #     print(f"Train Batch {batch_idx}: Images shape {images.shape}, Labels shape {labels.shape}")

    # Precompute and cache preprocessed tensors for training data
    handler.precompute_and_cache(train_loader)