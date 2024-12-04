import torch
from torch import nn
from jaxtyping import Float
from collections import defaultdict


def feature_extractor(model, model_input, layer_start, feat_output_path):
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

    with torch.no_grad():
        _ = model(model_input)

    # Remove hooks after feature extraction
    for hook_handle in hooks:
        hook_handle.remove()

    # Save the features dictionary
    torch.save(dict(layer2features), feat_output_path)

    return layer2features


class LinearProbeClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int = 1152,
        num_classes: int = 1000,
        **kwargs,
    ):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: Float[torch.Tensor, "b ..."], **data_kwargs) -> Float[torch.Tensor, "b"]:
        pooled_features = x.mean(dim=1)
        return self.classifier(pooled_features)