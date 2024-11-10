import torch
from torch import nn
import math
from k_diffusion.models.axial_rope import bounding_box, centers, make_grid
from k_diffusion.models.image_transformer_v2 import apply_rotary_emb, apply_rotary_emb_
from abc import ABC, abstractmethod


class AbstractPosEnc(nn.Module, ABC):
    def __init__(self, d_head, n_heads):
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads

    @abstractmethod
    def forward(self, pos):
        pass

    @abstractmethod
    def apply_emb(self, x, theta):
        pass


class NoPosEnc(AbstractPosEnc):

    def __init__(self, d_head, n_heads):
        super().__init__(d_head, n_heads)

    def forward(self, pos):
        return pos

    def apply_emb(self, x, theta):
        return x


class AxialRoPEBase(AbstractPosEnc):

    def __init__(self, d_head, n_heads, in_place=False):
        super().__init__(d_head, n_heads)
        self.in_place = in_place

    def apply_emb(self, x, theta):
        if self.in_place:
            return apply_rotary_emb_(x, theta)
        else:
            return apply_rotary_emb(x, theta)

    @abstractmethod
    def forward(self, pos):
        pass


class AxialRoPE2D(AxialRoPEBase):
    def __init__(self, dim, n_heads, learnable_freqs=False, relative_canvas=True):
        super().__init__(dim, n_heads, in_place=not learnable_freqs)
        self.learnable_freqs = learnable_freqs

        if relative_canvas:
            min_freq = math.pi
            max_freq = 10.0 * math.pi
        else:
            min_freq = 1 / 10_000
            max_freq = 1.0

        log_min = math.log(min_freq)
        log_max = math.log(max_freq)
        freqs = torch.stack([torch.linspace(log_min, log_max, n_heads * dim // 4 + 1)[:-1].exp()] * 2)
        self.freqs = nn.Parameter(freqs.view(2, dim // 4, n_heads).mT.contiguous(), requires_grad=learnable_freqs)

    def extra_repr(self):
        return f"dim={self.freqs.shape[-1] * 4}, n_heads={self.freqs.shape[-2]}"

    def forward(self, pos):
        theta_h = pos[..., None, 0:1] * self.freqs[0].to(pos.dtype)
        theta_w = pos[..., None, 1:2] * self.freqs[1].to(pos.dtype)
        return torch.cat((theta_h, theta_w), dim=-1)


def make_axial_pos_2d(h, w, pixel_aspect_ratio=1.0, align_corners=False, dtype=None, device=None, relative_pos=True):
    if relative_pos:
        y_min, y_max, x_min, x_max = bounding_box(h, w, pixel_aspect_ratio)
    else:
        y_min, y_max, x_min, x_max = -h / 2, h / 2, -w / 2, w / 2

    if align_corners:
        h_pos = torch.linspace(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = torch.linspace(x_min, x_max, w, dtype=dtype, device=device)
    else:
        h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)
    return make_grid(h_pos, w_pos)
