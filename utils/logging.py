import numpy as np
import torch
from torchvision.utils import make_grid

from einops import rearrange


# theoretically we do not need this function as wandb.Image directly calls make_grid under the hood, however with normalization!!!
def image_batch_to_grid_image(images, nrow=None, padding=2):
    """
    Input: Images BxCxHxW
    Output: Image CxnHxmW, n can be given or be determined, m is determined automatically
    """
    # calculate perfect n
    if nrow is None:
        nrow = int(np.ceil(np.sqrt(images.size(0))))

    grid = make_grid(images, nrow=nrow, padding=padding)
    return grid


def video_batch_to_grid_image(videos, padding=2):
    """
    Input: Videos BxCxTxHxW
    Output:: Image CxBHxTW
    """

    nrow = videos.size(2)
    videos = rearrange(videos, "b c t h w -> (b t) c h w")

    grid = make_grid(videos, nrow=nrow, padding=padding)

    return grid


def log_image(wandb, image, key, image_args={}):
    img = (img.cpu().numpy() + 1.0) * 0.5 * 255.0
    img = rearrange(img, "c h w -> h w c")

    img = wandb.Image(img, **image_args)
    wandb.log({key: [img]})
