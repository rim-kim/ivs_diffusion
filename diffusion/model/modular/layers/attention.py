from diffusion.model.modular.layers.rope import AxialRoPE2D
import torch
from torch import nn
import torch.nn.functional as F
from diffusion.model.modular.layers import (
    AdaRMSNorm,
    FeedForwardBlock,
    RMSNorm,
    scale_for_cosine_sim,
)
from k_diffusion.models.image_transformer_v2 import (
    Linear,
    apply_wd,
    checkpoint,
    zero_init,
)
from einops import rearrange
from jaxtyping import Float


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_cond_norm=None, d_head=64, dropout=0.0, ff_expand=3):
        super().__init__()
        d_ff = d_model * ff_expand

        self.self_attn = CondAttentionBlock(d_model, d_cond_norm, d_head, dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, d_cond_norm, dropout)

    def forward(self, x, pos, cond_norm):
        x = checkpoint(self.self_attn, x, pos, cond_norm)
        x = checkpoint(self.ff, x, cond_norm)
        return x


class CrossTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_cross,
        d_cond_norm,
        d_head=64,
        dropout=0.0,
        ff_expand=3,
    ):
        super().__init__()
        d_ff = d_model * ff_expand

        self.self_attn = CondAttentionBlock(d_model, d_cond_norm, d_head, dropout)
        self.cross_attn = CondCrossAttentionBlock(d_model, d_cross, d_cond_norm, d_head, dropout)
        self.ff = FeedForwardBlock(d_model, d_ff, d_cond_norm, dropout)

    def forward(self, x: Float[torch.Tensor, "B L C"], pos, x_cross, cond_norm, cross_mask=None):
        x = checkpoint(self.self_attn, x, pos, cond_norm)
        x_skip = x
        x = checkpoint(self.cross_attn, x, x_cross, cond_norm)

        if cross_mask is not None:
            x = x_skip * (1 - cross_mask[:, None, None]) + x * cross_mask[:, None, None]

        x = checkpoint(self.ff, x, cond_norm)
        return x


class CondCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_cross: int,
        d_cond_norm: int,
        d_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.norm = AdaRMSNorm(d_model, d_cond_norm)
        self.norm_cross = RMSNorm(d_cross)
        self.q_proj = apply_wd(Linear(d_model, d_model, bias=False))
        self.kv_proj = apply_wd(Linear(d_cross, d_model * 2, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head},"

    def forward(
        self,
        x: Float[torch.Tensor, "b l d"],
        x_cross: Float[torch.Tensor, "b l' d'"],
        cond_norm: Float[torch.Tensor, "b d"],
    ) -> Float[torch.Tensor, "b l d"]:

        skip = x
        x = self.norm(x, cond_norm)
        x_cross = self.norm_cross(x_cross)
        q = self.q_proj(x)
        kv = self.kv_proj(x_cross)

        q = rearrange(q, "n l (nh e) -> n nh l e", e=self.d_head)
        k, v = rearrange(kv, "n l (t nh e) -> t n nh l e", t=2, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)
        x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        x = rearrange(x, "n nh l e -> n l (nh e)")

        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class CondAttentionBlock(nn.Module):
    def __init__(self, d_model: int, d_cond_norm: int, d_head: int = 64, dropout: float = 0.0):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.norm = AdaRMSNorm(d_model, d_cond_norm)
        self.qkv_proj = apply_wd(Linear(d_model, d_model * 3, bias=False))
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.pos_emb = AxialRoPE2D(d_head // 2, self.n_heads)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = apply_wd(zero_init(Linear(d_model, d_model, bias=False)))

    def extra_repr(self):
        return f"d_head={self.d_head},"

    def forward(self, x, pos, cond_norm):
        skip = x
        x = self.norm(x, cond_norm)
        qkv = self.qkv_proj(x)
        pos = pos.to(qkv.dtype)
        theta = self.pos_emb(pos)

        q, k, v = rearrange(qkv, "n l (t nh e) -> t n nh l e", t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)
        theta = theta.movedim(-2, -3)
        q = self.pos_emb.apply_emb(q, theta)
        k = self.pos_emb.apply_emb(k, theta)

        x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        x = rearrange(x, "n nh l e -> n l (nh e)")

        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip
