from collections import defaultdict
from typing import Literal, DefaultDict, Tuple, Union
from pathlib import Path
from jaxtyping import Float
import torch

from dataset.utils import get_captions
from diffusion.model.t2i import T2ILatentRF2d
from diffusion.model.unclip import UnclipLatentRF2d
from configs.path_configs.path_configs import FEATURES_DIR

from utils.logging import logger


def extract_features(
    model: torch.nn.Module,
    model_name: str,
    model_input: Tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "..."]],
    layer_start: int,
    timestep: Union[float, int],
    batch_idx: int,
    feat_output_dir: Path = FEATURES_DIR,
    mode: Literal["train", "val", "test"] = "train",
    save: bool = False,
) -> DefaultDict[int, Float[torch.Tensor, "..."]]:
    """
    Extracts features from specified layers of the model's `unet.mid_level` module.

    :param model: The model to extract features from.
    :param model_name: The name of the model configuration.
    :param model_input: Input tensor for the model.
    :param layer_start: Starting layer index for feature extraction.
    :param timestep: The amount of noise to diffuse.
    :param batch_idx: Batch index, used for naming saved feature files.
    :param feat_output_dir: Directory to save extracted features (if save is True).
    :param mode: Mode of operation, can be "train", "val", or "test". Defaults to "train".
    :param save: Whether to save the extracted features to a file. Defaults to False.
    :return: A dictionary mapping layer indices to feature tensors.
    """
    imgs, targets = model_input
    layer2features = defaultdict(lambda: torch.empty(0))
    hooks = []

    # Hook function capturing layer index
    def hook(layer_idx):
        def inner_hook(module, input, features):
            layer2features[layer_idx] = features
        return inner_hook
    
    # Register hooks and store hook handles
    for layer_idx in range(layer_start, len(model.unet.mid_level)):
        hook_handle = model.unet.mid_level[layer_idx].register_forward_hook(hook(layer_idx))
        hooks.append(hook_handle)

    # Provide correct input to pre-trained model
    with torch.no_grad():
        if isinstance(model, T2ILatentRF2d):
            if model_name == "t2i":
                captions = get_captions(targets)
            elif model_name == "t2i_uncond":
                captions = [""] * imgs.size(0)
            else:
                raise ValueError(f"Unexpected model_name: {model_name}")
            model.get_features(imgs, timestep, captions)
        elif isinstance(model, UnclipLatentRF2d):
            model.get_features(imgs, timestep)
        else:
            raise TypeError(f"Unsupported model type: {type(model).__name__}")

    # Remove hooks after feature extraction
    for hook_handle in hooks:
        hook_handle.remove()

    # Save the features dictionary of each batch
    if save:
        logger.info(f"Saving features from layer {layer_start} to {len(model.unet.mid_level)} for batch nr. {batch_idx}.")
        fname = f"{mode}_{batch_idx}_layer2features.pt"
        save_path = feat_output_dir / fname
        torch.save(dict(layer2features), save_path)

    return layer2features


class LinearProbeClassifier(torch.nn.Module):
    """
    A linear classifier for probing features extracted from a specific layer.

    :param feature_dim: The dimensionality of the input features.
    :param num_classes: The number of output classes for classification. Defaults to 1000.
    :param layer_idx: Layer index from which to extract features for classification.
    :param kwargs: Additional keyword arguments.
    """
    def __init__(
        self,
        feature_dim: int = 1152,
        num_classes: int = 1000,
        layer_idx: int = 14,
        **kwargs,
    ):
        super().__init__()
        self.classifier = torch.nn.Linear(feature_dim, num_classes)
        self.layer_idx = layer_idx

    def forward(self, x: defaultdict[int, Float[torch.Tensor, "b ..."]], **data_kwargs) -> Float[torch.Tensor, "b"]:
        """
        Performs forward pass for classification using features from a specific layer.

        :param x: Dictionary mapping layer indices to feature tensors.
        :param data_kwargs: Additional keyword arguments.
        :return: Predicted logits for each class.
        :raises KeyError: If the specified layer index is not found in the input features.
        """
        if self.layer_idx not in x:
            raise KeyError(f"Layer index {self.layer_idx} not found in the provided features.")
        layer_features = x[self.layer_idx]
        pooled_features = layer_features.mean(dim=1)
        return self.classifier(pooled_features)
