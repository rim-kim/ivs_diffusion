import torch
from torch import nn
from transformers import CLIPVisionModel, CLIPVisionModelWithProjection
from torchvision.transforms.functional import normalize
from jaxtyping import Float
import torch
from torch import nn
from torch.nn.functional import avg_pool2d, interpolate
from torchvision.transforms.functional import center_crop
from transformers import AutoTokenizer, CLIPTextModel
import os


class ClipTextEmbedder(nn.Module):

    def __init__(
        self,
        repo: str = "openai/clip-vit-large-patch14",
    ) -> None:
        super().__init__()
        self.repo = repo

        # otherwise we get weird warnings from HF
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.model = CLIPTextModel.from_pretrained(repo)
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        self.max_length = 77

        torch.compile(self.model)

        self.model.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def forward(self, prompts: list[str]) -> torch.Tensor:

        tokens = self.tokenizer(prompts, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        for k, v in tokens.items():
            if isinstance(v, torch.Tensor):
                tokens[k] = v.to(device=self.model.device)
        outputs = self.model(**tokens)

        return outputs.last_hidden_state


class ClipImgEmbedder(nn.Module):

    def __init__(
        self,
        clip_model: str = "openai/clip-vit-large-patch14",
        with_projection: bool = False,
    ) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.with_projection = with_projection
        if with_projection:
            self.model = CLIPVisionModelWithProjection.from_pretrained(clip_model)
        else:
            self.model = CLIPVisionModel.from_pretrained(clip_model)

        torch.compile(self.model)

        self.model.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def forward(self, imgs: Float[torch.Tensor, "B C H W"]) -> Float[torch.Tensor, "B D"]:
        assert imgs.min() >= -1.01
        assert imgs.max() <= 1.01
        assert len(imgs.shape) == 4

        imgs = (imgs + 1.0) / 2.0
        imgs = interpolate(imgs, [224, 224], mode="bilinear")
        imgs = normalize(
            imgs,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        vision_outputs = self.model(imgs)
        if self.with_projection:
            out = vision_outputs.image_embeds
        else:
            out = vision_outputs.pooler_output

        return out


# this resize is preferable when differentiating through it
def better_resize(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    ss = imgs.shape
    assert ss[-3] == 3

    H, W = ss[-2:]

    if len(ss) == 3:
        imgs = imgs.unsqueeze(0)

    side = min(H, W)
    factor = side // image_size

    imgs = center_crop(imgs, [side, side])
    if factor > 1:
        imgs = avg_pool2d(imgs, factor)
    imgs = interpolate(imgs, [image_size, image_size], mode="bilinear")

    if len(ss) == 3:
        imgs = imgs[0]
    return imgs


class OpenClipImgEmbedder(nn.Module):

    def __init__(
        self,
        clip_model: str,
    ) -> None:
        import open_clip

        super().__init__()
        self.clip_model = clip_model
        self.model = open_clip.create_model_from_pretrained("hf-hub:" + clip_model, return_transform=False)
        self.image_size = 224

        torch.compile(self.model)

        self.model.requires_grad_(False)
        self.model.eval()

    # @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        assert imgs.min() >= -1.01
        assert imgs.max() <= 1.01
        assert len(imgs.shape) == 4

        imgs = (imgs + 1.0) / 2.0
        imgs = better_resize(imgs, self.image_size)
        imgs = normalize(
            imgs,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        image_features = self.model.encode_image(imgs)

        return image_features
