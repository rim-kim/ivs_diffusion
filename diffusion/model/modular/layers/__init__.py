import torch.nn as nn
import torch.nn.functional as F
from k_diffusion.models.image_transformer_v2 import Linear, apply_wd, zero_init, tag_module, LinearGEGLU
import torch
from einops import rearrange
from typing import Literal
from functools import reduce


def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype) ** 2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


def scale_for_cosine_sim(q, k, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype) ** 2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype) ** 2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.to(q.dtype), k * scale_k.to(k.dtype)


def scale_for_cosine_sim_qkv(qkv, scale, eps):
    q, k, v = qkv.unbind(2)
    q, k = scale_for_cosine_sim(q, k, scale[:, None], eps)
    return torch.stack((q, k, v), dim=2)


def scale_for_cosine_sim_kv(q, kv, scale, eps):
    k, v = kv.unbind(2)
    q, k = scale_for_cosine_sim(q, k, scale[:, None], eps)
    return q, torch.stack((k, v), dim=2)


def linear_swiglu(x, weight, bias=None):
    x = x @ weight.mT
    if bias is not None:
        x = x + bias
    x, gate = x.chunk(2, dim=-1)
    return x * F.silu(gate)


class LinearSwiGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x):
        return linear_swiglu(x, self.weight, self.bias)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, d_cond_norm, dropout=0.0):
        super().__init__()
        self.norm = AdaRMSNorm(d_model, d_cond_norm)
        self.up_proj = apply_wd(LinearSwiGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(Linear(d_ff, d_model, bias=False)))

    def forward(self, x, cond_norm):
        skip = x
        x = self.norm(x, cond_norm)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class AdaRMSNorm(nn.Module):
    def __init__(self, features, cond_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.linear = apply_wd(zero_init(Linear(cond_features, features, bias=False)))
        tag_module(self.linear, "mapping")

    def extra_repr(self):
        return f"eps={self.eps},"

    def forward(self, x, cond):
        # removed one additional expansion here
        return rms_norm(x, self.linear(cond)[:, None, :] + 1, self.eps)


class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)


def Patch2D(type: Literal["split", "split_last" "merge"], in_features, out_features, patch_size=(2, 2)):
    if type == "split":
        return TokenSplit2D(in_features, out_features, patch_size)
    if type == "split_last":
        return TokenSplitLast2D(in_features, out_features, patch_size)
    elif type == "merge":
        return TokenMerge2D(in_features, out_features, patch_size)
    else:
        raise ValueError(f"Unknown type: {type}")


class TokenMerge2D(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(Linear(in_features * self.h * self.w, out_features, bias=False))

    def downscale_pos(self, pos):
        pos = rearrange(pos, "... (h nh) (w nw) e -> ... h w (nh nw) e", nh=self.h, nw=self.w)
        return torch.mean(pos, dim=-2)

    def forward(self, x):
        x = rearrange(x, "... (h nh) (w nw) e -> ... h w (nh nw e)", nh=self.h, nw=self.w)
        return self.proj(x)


class TokenSplit2D(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(Linear(in_features, out_features * self.h * self.w, bias=False))
        self.fac = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, skip=None):
        x = self.proj(x)
        x = rearrange(x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w)
        if skip is None:
            return x
        return torch.lerp(skip, x, self.fac.to(x.dtype))


# Doens't have skip but norm
class TokenSplitLast2D(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.norm = RMSNorm(in_features)
        self.proj = apply_wd(Linear(in_features, out_features * self.h * self.w, bias=False))
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        x = self.norm(x)
        x = self.proj(x)
        x = rearrange(x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w)
        return x
