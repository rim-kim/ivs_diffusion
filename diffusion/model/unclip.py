from typing import List, Optional, Union

import einops
import numpy as np
import torch
from jaxtyping import Float
from k_diffusion.models.image_transformer_v2 import Linear
from torch import nn
from tqdm.auto import tqdm

from diffusion.model.rf import LatentRF2D


class UnclipLatentRF2d(LatentRF2D):
    def __init__(self, img_embedder: nn.Module, d_img: int, d_t: int, c_dropout: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_embedder = img_embedder
        self.c_dropout = c_dropout
        self.img_proj = Linear(d_img, d_t, bias=False)
        self.norm = nn.LayerNorm(d_t)

    def get_conditioning(self, t: Float[torch.Tensor, "b"], c_img: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        if self.time_cond_type == "sigma":
            c_noise = torch.log(t) / 4
        elif self.time_cond_type == "rf_t":
            c_noise = t
        else:
            raise NotImplementedError(f'Unknown time conditioning type "{self.time_cond_type}".')

        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))

        keep_idx = (torch.rand(time_emb.shape[0]) >= self.c_dropout).nonzero().flatten()
        c_img = c_img[keep_idx]
        c_img = kwargs["clip_embds"]
        c_img = self.img_proj(c_img)
        c_img = self.norm(c_img)

        img_emb = torch.zeros_like(time_emb)
        img_emb[keep_idx] = c_img

        cond_time = self.mapping(time_emb + img_emb)

        return {"cond_norm": cond_time}

    def get_unconditional_conditioning(self, t: Float[torch.Tensor, "b"], **kwargs) -> dict[str, torch.Tensor]:
        if self.time_cond_type == "sigma":
            c_noise = torch.log(t) / 4
        elif self.time_cond_type == "rf_t":
            c_noise = t
        else:
            raise NotImplementedError(f'Unknown time conditioning type "{self.time_cond_type}".')

        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))
        cond_time = self.mapping(time_emb)

        return {"cond_norm": cond_time}

    def forward(self, x: Float[torch.Tensor, "b ..."], **data_kwargs) -> Float[torch.Tensor, "b"]:
        return super().forward(x=x, c_img=x, **data_kwargs)

    def get_features(self, x: Float[torch.Tensor, "b ..."], t: int, clip_embds: Float[torch.Tensor, "b ..."]):
        return super().get_features(x, c_img=x, t=t, clip_embds=clip_embds)

    def sample(
        self,
        z: Float[torch.Tensor, "b c ..."],
        c_img: Float[torch.Tensor, "b c ..."],
        sample_steps=50,
        return_list: bool = False,
        **data_kwargs,
    ) -> Union[Float[torch.Tensor, "b ..."], List[Float[torch.Tensor, "b ..."]]]:
        return super().sample(z, c_img=c_img, sample_steps=sample_steps, return_list=return_list, **data_kwargs)

    @torch.no_grad()
    def validate(
        self,
        dataloader_val: "torch.utils.data.DataLoader",
        global_rank: int,
        global_samples: int,
        max_steps: Optional[int],
        device,
        dtype,
        wandb,
        monitor,
    ) -> None:
        c_dropout = self.c_dropout
        self.c_dropout = 0.0
        if self.val_shape is None:
            if global_rank == 0:
                print("Skipping validation for LatentRF model as val_shape is not provided.")
            return
        if global_rank == 0:
            for i, val_batch in enumerate(
                tqdm(dataloader_val, desc=f"Validating", disable=(global_rank != 0), total=max_steps)
            ):
                sample_tensor = torch.randn((1, *self.val_shape), dtype=dtype, generator=torch.manual_seed(i)).to(
                    device
                )
                c_img = val_batch["x"][:1].to(device=device, dtype=dtype)
                res = einops.rearrange(
                    self.sample(z=sample_tensor, c_img=c_img)[0],
                    "c h w -> h w c",
                )
                c_img = self.ae.decode(self.ae.encode(c_img))
                img = einops.rearrange(c_img[0], "c h w -> h w c")
                res = torch.cat([img, res], dim=0)
                wandb.log(
                    {
                        f"Val/Vis/sample_{i}": wandb.Image(
                            ((res.clip(-1, 1) / 2 + 0.5) * 255).round().float().cpu().numpy().astype(np.uint8)
                        )
                    },
                    step=global_samples,
                )

                if max_steps is not None and i + 1 >= max_steps:
                    break

        self.c_dropout = c_dropout
