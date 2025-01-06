from pathlib import Path
from datetime import datetime
from typing import List

# Absolute path to the root dir
ROOT_ABSPATH = Path(__file__).resolve().parent.parent.parent

# First-level dirs (ivs_diffusion)
CONFIGS_DIR = ROOT_ABSPATH / "configs"
DATA_DIR = ROOT_ABSPATH / "data"
MODEL_CKPT_DIR = ROOT_ABSPATH / "model_checkpoints"
LOGS_DIR = ROOT_ABSPATH / "logs"

# Second-level dirs (configs)
HYPERPARAMS_DIR = CONFIGS_DIR / "hyperparameters"
MODEL_CONFIGS_DIR = CONFIGS_DIR / "model"
TOKENS_DIR = CONFIGS_DIR / "tokens"

# Second-level dirs (data)
FEATURES_DIR = DATA_DIR / "features"
IMAGENET_SHARDS_DIR = DATA_DIR / "imagenet_shards"
IMAGENET_LATENT_AND_CLIP_DIR = DATA_DIR / "imagenet_latent_and_clip"
TOY_DATA_DIR = DATA_DIR / "toy_data"
PREPROCESSED_IMAGENET_DIR = DATA_DIR / "preprocessed_data"

# Second-level dirs (model_checkpoints)
MODEL_CKPT_PROBING_DIR = MODEL_CKPT_DIR / "probing"

# Second-level dirs (logs)
LOG_DIR_PATH = LOGS_DIR / datetime.now().strftime("%y%m%d-%H%M%S")

# Third-level files (logs)
LOG_FILE_PATH = LOG_DIR_PATH / "linear_probing.log"

# Third-level dirs (data)
TRAIN_LATENT_AND_CLIP_DIR = IMAGENET_LATENT_AND_CLIP_DIR / "train"
VAL_LATENT_AND_CLIP_DIR = IMAGENET_LATENT_AND_CLIP_DIR / "val"

dirs = [
    CONFIGS_DIR,
    DATA_DIR,
    MODEL_CKPT_DIR,
    LOGS_DIR,
    HYPERPARAMS_DIR,
    MODEL_CONFIGS_DIR,
    TOKENS_DIR,
    FEATURES_DIR,
    IMAGENET_SHARDS_DIR,
    IMAGENET_LATENT_AND_CLIP_DIR,
    TOY_DATA_DIR,
    PREPROCESSED_IMAGENET_DIR,
    MODEL_CKPT_PROBING_DIR,
    LOG_DIR_PATH,
    TRAIN_LATENT_AND_CLIP_DIR,
    VAL_LATENT_AND_CLIP_DIR,
]

def create_dirs(dirs: List[Path]) -> None:
    """
    Creates the specified directories if they do not already exist.

    :param dirs: One or more `Path` objects representing the directories to be created.
    :raises TypeError: If any element in `dirs` is not a `Path` object.
    :return: None
    """
    for dir in dirs:
        if isinstance(dir, Path):
            dir.mkdir(parents=True, exist_ok=True)
        else: 
            raise TypeError("Expected a Path instance.")
        
create_dirs(dirs)
