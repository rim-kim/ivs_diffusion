from typing import Tuple, Union

import torch
from jaxtyping import Float, Int

from dataset.utils import get_captions
from diffusion.model.rf import LatentRF2D
from diffusion.model.t2i import T2ILatentRF2d
from diffusion.model.unclip import UnclipLatentRF2d


def extract_features(
    pretrained_model: torch.nn.Module,
    model_name: str,
    layer_idx: int,
    data: Tuple[Float[torch.Tensor, "b ..."], Int[torch.Tensor, "b"], Float[torch.Tensor, "b ..."]],
    timestep: Union[float, int],
) -> Float[torch.Tensor, "b ..."]:
    """
    Extracts features from specified layers of the model's `unet.mid_level` module.

    :param pretrained_model: The pre-trained model to extract features from.
    :param model_name: The name of the pre-trained model configuration.
    :param layer_idx: The transformer layer index to extract features from.
    :param data: Input tensors for the pre-trained model.
    :param timestep: The amount of noise to diffuse.
    :return: A tensor containing features of specified transformer layer.
    """
    if not (14 <= layer_idx <= 27):
        raise ValueError(f"The layer index must be an integer in the range 14 to 27 (inclusive), not {layer_idx}.")
    imgs, targets, clip_embds = data
    features = []

    # Hook function storing the layer features
    def hook(module, input, output):
        features.append(output)

    # Register the hook handle
    hook_handle = pretrained_model.unet.mid_level[layer_idx].register_forward_hook(hook)

    # Provide correct input to pre-trained model
    with torch.no_grad():
        if isinstance(pretrained_model, T2ILatentRF2d):
            if model_name == "t2i":
                captions = get_captions(targets)
            elif model_name == "t2i_uncond":
                captions = [""] * imgs.size(0)
            else:
                raise ValueError(f"Unexpected model_name: {model_name}")
            pretrained_model.get_features(imgs, timestep, captions)
        elif isinstance(pretrained_model, UnclipLatentRF2d):
            pretrained_model.get_features(imgs, timestep, clip_embds)
        elif isinstance(pretrained_model, LatentRF2D):
            pretrained_model.get_features(imgs, timestep)
        else:
            raise TypeError(f"Unsupported model type: {type(pretrained_model).__name__}")

    # Remove hook after feature extraction
    hook_handle.remove()

    return features[-1]


class LinearProbeClassifier(torch.nn.Module):
    """
    A linear classifier for probing features extracted from a specific layer.

    :param layer_idx: Layer index from which to extract features for classification.
    :param feature_dim: The dimensionality of the input features. Defaults to 1152.
    :param num_classes: The number of output classes for classification. Defaults to 1000.
    """

    def __init__(
        self,
        layer_idx: int = 14,
        feature_dim: int = 1152,
        num_classes: int = 1000,
        pooling: bool = True,
    ):
        super().__init__()
        self.classifier = torch.nn.Linear(feature_dim, num_classes)
        self.layer_idx = layer_idx
        self.pooling = pooling

    def forward(self, layer_features: Float[torch.Tensor, "b ..."]) -> Float[torch.Tensor, "b ..."]:
        """
        Performs forward pass for classification using features from a specific layer.

        :param layer_features: A tensor containing features of specified transformer layer.
        :return: Predicted logits for each class.
        """
        if self.pooling:
            layer_features = layer_features.mean(dim=1)
        return self.classifier(layer_features)
