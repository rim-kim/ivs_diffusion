from pydoc import locate
from diffusion.model.modular.layers import TokenMerge2D, TokenSplitLast2D
from diffusion.model.modular.layers.attention import CrossTransformerLayer, TransformerLayer
from torch import nn
import torch
from jaxtyping import Float
from einops import rearrange


class Level(nn.ModuleList):
    def forward(self, x, *args, **kwargs):
        for layer in self:
            x = layer(x, *args, **kwargs)
        return x


class Transformer(nn.Module):
    def __init__(self, timestep_width=1024):
        super().__init__()

        in_channels = 4  # VAE latent channels
        out_channels = 4  # VAE latent channels

        # transformer config
        transformer_width = 1152
        transformer_depth = 28
        d_heads = 72

        self.mid_level, self.mid_merge, self.mid_split = None, None, None
        self.mid_level = Level(
            [TransformerLayer(transformer_width, timestep_width, d_heads) for _ in range(transformer_depth)]
        )
        self.mid_merge = TokenMerge2D(in_features=in_channels, out_features=transformer_width, patch_size=(2, 2))
        self.mid_split = TokenSplitLast2D(
            in_features=transformer_width,
            out_features=out_channels,
            patch_size=(2, 2),
        )

    def forward(
        self, x: Float[torch.Tensor, "B C *DIMS"], pos: Float[torch.Tensor, "B cn *DIM"], cond_norm
    ):

        x = rearrange(x, "b c ... -> b ... c")
        pos = rearrange(pos, "b cn ... -> b ... cn")

        C_pos = pos.shape[-1]

        x = self.mid_merge(x)
        pos = self.mid_merge.downscale_pos(pos)
        B, *DIMS, C = x.shape
        x = x.reshape(B, -1, C)
        pos = pos.reshape(B, -1, C_pos)
        x = self.mid_level(x, pos=pos, cond_norm=cond_norm)
        x = x.reshape(B, *DIMS, C)
        pos = pos.reshape(B, *DIMS, C_pos)
        x = self.mid_split(x)

        x = rearrange(x, "b ... c -> b c ...")

        return x

class CATransformer(nn.Module):
    def __init__(self):
        super().__init__()

        in_channels = 4  # VAE latent channels
        out_channels = 4  # VAE latent channels

        # transformer config
        transformer_width = 1152
        transformer_depth = 28
        ca_width = 768
        d_heads = 72
        timestep_width = 1024

        self.mid_level, self.mid_merge, self.mid_split = None, None, None
        self.mid_level = Level(
            [CrossTransformerLayer(transformer_width, ca_width, timestep_width, d_heads) for _ in range(transformer_depth)]
        )
        self.mid_merge = TokenMerge2D(in_features=in_channels, out_features=transformer_width, patch_size=(2, 2))
        self.mid_split = TokenSplitLast2D(
            in_features=transformer_width,
            out_features=out_channels,
            patch_size=(2, 2),
        )

    def forward(
        self, x: Float[torch.Tensor, "B C *DIMS"], pos: Float[torch.Tensor, "B cn *DIM"], x_cross, cond_norm
    ):

        x = rearrange(x, "b c ... -> b ... c")
        pos = rearrange(pos, "b cn ... -> b ... cn")

        C_pos = pos.shape[-1]

        x = self.mid_merge(x)
        pos = self.mid_merge.downscale_pos(pos)
        B, *DIMS, C = x.shape
        x = x.reshape(B, -1, C)
        pos = pos.reshape(B, -1, C_pos)
        x = self.mid_level(x, pos=pos, x_cross=x_cross, cond_norm=cond_norm)
        x = x.reshape(B, *DIMS, C)
        pos = pos.reshape(B, *DIMS, C_pos)
        x = self.mid_split(x)

        x = rearrange(x, "b ... c -> b c ...")

        return x
