from typing import Union, List, Literal, Optional

import torch
from torch import nn
from jaxtyping import Float
from tqdm.auto import trange
import einops
import numpy as np  

from k_diffusion.models.image_transformer_v2 import MappingNetwork, tag_module, Linear, layers
from abc import ABC, abstractmethod

from diffusion.model.modular.layers.rope import make_axial_pos_2d

class RF(nn.Module, ABC):
    def __init__(
        self,
        unet: nn.Module,
        mapping,
        train_timestep_sampling: Literal["logit_sigmoid", "uniform", "fixed"] = "fixed",
        time_cond_type: Literal["sigma", "rf_t"] = "rf_t",
        cfg_scale: float = 1.0,
        fixed_t: float = 0.5,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.train_timestep_sampling = train_timestep_sampling
        self.mapping = mapping
        self.cfg_scale = cfg_scale

        self.time_emb = layers.FourierFeatures(1, mapping.width)
        self.time_in_proj = Linear(mapping.width, mapping.width, bias=False)
        self.time_cond_type = time_cond_type
        self.mapping = tag_module(
            MappingNetwork(mapping.depth, mapping.width, mapping.d_ff, dropout=mapping.dropout), "mapping"
        )
        self.fixed_t = fixed_t

    @abstractmethod
    def get_pos(self, x: Float[torch.Tensor, "B C *DIM"]) -> Float[torch.Tensor, "B *DIM c"]:
        pass

    def get_conditioning(self, t: Float[torch.Tensor, "b"], **kwargs) -> dict[str, torch.Tensor]:
        if self.time_cond_type == "sigma":
            c_noise = torch.log(t) / 4
        elif self.time_cond_type == "rf_t":
            c_noise = t
        else:
            raise NotImplementedError(f'Unknown time conditioning type "{self.time_cond_type}".')

        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))
        cond_time = self.mapping(time_emb)

        return {"cond_norm": cond_time}
    
    # Used for the unconditional branch of CFG. If you want custom behaviour, 
    # e.g. because you have multiple conditionings, overwrite this in your subclass
    def get_unconditional_conditioning(self, t: Float[torch.Tensor, "b"], **kwargs) -> dict[str, torch.Tensor]:
        # By default, the unconditional path for CFG will only use time conditioning
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
        B = x.size(0)
        if self.train_timestep_sampling == "logit_sigmoid":
            t = torch.sigmoid(torch.randn((B,), device=x.device))
        elif self.train_timestep_sampling == "uniform":
            t = torch.rand((B,), device=x.device)
        elif self.train_timestep_sampling == "fixed":
            t = torch.full((B,), self.fixed_t, device=x.device)
        else:
            raise ValueError(f'Unknown train timestep sampling method "{self.train_timestep_sampling}".')
        texp = t.view([B, *([1] * len(x.shape[1:]))])

        z1 = torch.randn_like(x)

        zt = (1 - texp) * x + texp * z1

        # make t, zt into same dtype as x
        dtype = x.dtype
        zt, t = zt.to(dtype), t.to(dtype)

        cond_dict = self.get_conditioning(t, **data_kwargs)
        pos = self.get_pos(zt)

        vtheta = self.unet(zt, pos=pos, **cond_dict)
        return ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))

    @torch.no_grad()
    def sample(
        self, z: Float[torch.Tensor, "b c ..."], sample_steps=50, return_list: bool = False, **data_kwargs
    ) -> Union[Float[torch.Tensor, "b ..."], List[Float[torch.Tensor, "b ..."]]]:
        B = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * B, device=z.device, dtype=z.dtype).view([B, *([1] * len(z.shape[1:]))])
        if return_list:
            images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * B, device=z.device, dtype=z.dtype)
            pos = self.get_pos(z)

            cond_dict = self.get_conditioning(t, **data_kwargs)
            vc_cond = self.unet(z, pos=pos, **cond_dict)
            
            if self.cfg_scale > 1.0:
                uncond_dict = self.get_unconditional_conditioning(t, **data_kwargs)
                vc_uncond = self.unet(z, pos=pos, **uncond_dict)
                z = z - dt * (vc_uncond + self.cfg_scale * (vc_cond - vc_uncond))
            else:
                z = z - dt * vc_cond
            
            if return_list:
                images.append(z)
        if return_list:
            return images
        else:
            return z



class LatentRF2D(RF):
    def __init__(
        self,
        ae: nn.Module,
        val_shape: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ae = ae
        self.val_shape = val_shape

    def forward(self, x: Float[torch.Tensor, "b ..."], **data_kwargs) -> Float[torch.Tensor, "b"]:
        latent = self.ae.encode(x)
        return super().forward(latent, **data_kwargs)

    def get_pos(self, x: Float[torch.Tensor, "B C *DIM"]) -> Float[torch.Tensor, "B *DIM c"]:
        B, _, *DIMS = x.shape
        pos = make_axial_pos_2d(*DIMS, device=x.device).view(1, *DIMS, -1).expand(B, -1, -1, -1)
        return pos.movedim(-1, 1)

    def sample(
        self, z: Float[torch.Tensor, "b c ..."], sample_steps=50, return_list: bool = False, **data_kwargs
    ) -> Union[Float[torch.Tensor, "b ..."], List[Float[torch.Tensor, "b ..."]]]:
        latent = super().sample(z, sample_steps=sample_steps, return_list=return_list, **data_kwargs)
        return self.ae.decode(latent)

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
        if self.val_shape is None:
            if global_rank == 0:
                print("Skipping validation for LatentRF model as val_shape is not provided.")
            return
        if global_rank == 0:
            for i in trange(
                (max_steps if max_steps is not None else 1),
                desc=f"Validating",
                disable=(global_rank != 0),
                total=max_steps,
            ):
                sample_tensor = torch.randn((1, *self.val_shape), dtype=dtype, generator=torch.manual_seed(i)).to(
                    device
                )
                res = einops.rearrange(
                    self.sample(sample_tensor)[0],
                    "c h w -> h w c",
                )
                wandb.log(
                    {
                        f"Val/Vis/sample_{i}": wandb.Image(
                            ((res.clip(-1, 1) / 2 + 0.5) * 255).round().float().cpu().numpy().astype(np.uint8)
                        )
                    },
                    step=global_samples,
                )
