from typing import Union, List, Optional

import torch
from torch import nn
from jaxtyping import Float
import einops
import numpy as np
from diffusion.model.rf import LatentRF2D
from k_diffusion.models.image_transformer_v2 import Linear
from tqdm.auto import tqdm
from PIL import Image, ImageDraw


# from CompVis/stable-diffusion
def log_txt_as_img(wh: tuple[int, int], xc: list[str], size=10, device=None, dtype=None):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start : start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black")
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts, device=device, dtype=dtype)
    return txts


class T2ILatentRF2d(LatentRF2D):
    def __init__(self, txt_embedder: nn.Module, c_dropout: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.txt_embedder = txt_embedder
        self.c_dropout = c_dropout

    def get_conditioning(self, t: Float[torch.Tensor, "b"], txt: list[str], **kwargs) -> dict[str, torch.Tensor]:
        if self.time_cond_type == "sigma":
            c_noise = torch.log(t) / 4
        elif self.time_cond_type == "rf_t":
            c_noise = t
        else:
            raise NotImplementedError(f'Unknown time conditioning type "{self.time_cond_type}".')

        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))
        cond_time = self.mapping(time_emb)

        B = time_emb.shape[0]

        with torch.no_grad():
            prompt_embs = self.txt_embedder(txt)

        return {"cond_norm": cond_time, "x_cross": prompt_embs}

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

        B = time_emb.shape[0]

        with torch.no_grad():
            prompt_embs = self.txt_embedder([""] * B)

        return {"cond_norm": cond_time, "x_cross": prompt_embs}
    
    def forward(self, x: Float[torch.Tensor, "b ..."], caption: list[str], **data_kwargs) -> Float[torch.Tensor, "b"]:
        return super().forward(x=x, txt=caption, **data_kwargs)
    
    def get_features(self, x: Float[torch.Tensor, "b ..."], t: int, caption: list[str]):
        return super().get_features(x=x, txt=caption, t=t)

    def sample(
        self,
        z: Float[torch.Tensor, "b c ..."],
        txt: Float[torch.Tensor, "b c ..."],
        sample_steps=50,
        return_list: bool = False,
        **data_kwargs,
    ) -> Union[Float[torch.Tensor, "b ..."], List[Float[torch.Tensor, "b ..."]]]:
        return super().sample(z, txt=txt, sample_steps=sample_steps, return_list=return_list, **data_kwargs)

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
                txt = val_batch["caption"][:1]
                res = self.sample(z=sample_tensor, txt=txt)
                txt_img = log_txt_as_img(res.shape[2:], txt, device=device, dtype=dtype)
                res = einops.rearrange(
                    torch.cat([txt_img[0], res[0]], dim=1),
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

                if max_steps is not None and i + 1 >= max_steps:
                    break

        self.c_dropout = c_dropout
