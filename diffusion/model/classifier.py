import torch
from torch import nn
from jaxtyping import Float


class FeatureExtractor(nn.Module):
    def __init__(
        self, 
        model: nn.Module,
        layer_num: int,
        fixed_t: Float,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.layer_num = layer_num
        self.fixed_t = fixed_t

    def forward(self, x: Float[torch.Tensor, "b ..."], **data_kwargs):
        features = []

        def hook(module, input, output):
            features.append(output)
        
        self.model.unet.mid_level[self.layer_num].register_forward_hook(hook)
        _ = self.model.extract_features(x, self.fixed_t, **data_kwargs)
        return features[-1]


class LinearProbeClassifier(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        feature_dim: int = 1152,
        num_classes: int = 1000,
        **kwargs,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: Float[torch.Tensor, "b ..."], **data_kwargs) -> Float[torch.Tensor, "b"]:
        with torch.no_grad():
            features = self.feature_extractor(x, **data_kwargs)

        pooled_features = features.mean(dim=1)
        return self.classifier(pooled_features)

