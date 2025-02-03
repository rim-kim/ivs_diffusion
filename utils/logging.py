import logging

import numpy as np
from colorlog import ColoredFormatter
from einops import rearrange
from torchvision.utils import make_grid

from configs.path_configs.path_configs import LOG_FILE_PATH


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


def setup_logger():
    """Sets up a logging system with both file and console handlers
    at different log levels."""
    logger = logging.getLogger("linear_probing")

    # Prevent configuring the logger multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    # File handler for detailed debug logs
    fh = logging.FileHandler(LOG_FILE_PATH)
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)-8s - %(message)s")
    fh.setFormatter(file_formatter)

    # Console handler for user-friendly colored logs
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    LOGFORMAT = "%(asctime)s - %(log_color)s%(levelname)-8s%(reset)s - " "%(log_color)s%(message)s%(reset)s"
    color_formatter = ColoredFormatter(LOGFORMAT)
    ch.setFormatter(color_formatter)

    # Adding handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


logger = setup_logger()
