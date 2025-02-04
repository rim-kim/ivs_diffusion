from argparse import ArgumentParser, Namespace
from collections import defaultdict
from typing import Any, Dict

import torch
from omegaconf import OmegaConf

from configs.hyperparameters.hyperparameters import DEVICE, default_models
from configs.warnings.suppress_warnings import suppress_warnings
from dataset.dataset_loaders import LatentDatasetLoader
from diffusion.model.modules.clip import ClipImgEmbedder, ClipTextEmbedder
from probing.classifier import LinearProbeClassifier
from probing.linear_probe import init_model, train
from utils.logging import logger

suppress_warnings()


def get_args() -> Namespace:
    """
    Parse command-line arguments to select models and hyperparameters.

    :return: A namespace containing selected models, timestep, and layer index.
    """
    parser = ArgumentParser(description="Select models and hyperparameters")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(default_models.keys()) + ["all"],
        default=["all"],
        help="Choose model(s) to run (or 'all' for all models).",
    )
    parser.add_argument("--timestep", type=float, default=0.25, help="Timestep value for all models (default: 0.25)")
    parser.add_argument("--layer_idx", type=int, default=14, help="Layer index value for all models (default: 14)")
    args = parser.parse_args()
    return args


def define_models(args: Namespace) -> Dict[str, Dict[str, Any]]:
    """
    Define and configure models based on user-specified arguments.

    :param args: Parsed command-line arguments containing model selection, timestep, and layer index.
    :return: A dictionary mapping model names to their respective hyperparameter configurations.
    """
    selected_models = default_models.keys() if "all" in args.models else args.models
    models = defaultdict(lambda: dict())
    for selected_model in selected_models:
        if "timestep" in default_models[selected_model]:
            default_models[selected_model]["timestep"] = args.timestep
        if "layer_idx" in default_models[selected_model]:
            default_models[selected_model]["layer_idx"] = args.layer_idx
        models[selected_model].update(default_models[selected_model])
    return models


if __name__ == "__main__":
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Exiting.")
        raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and drivers installed.")
    torch.set_float32_matmul_precision("high")

    # Iterate over the user-selected model configs and hyperparams
    models = define_models(get_args())
    for model_name, model_config in models.items():

        # Initialize the pre-trained model and the Linear Clasisfier
        if model_name == "txt_emb":
            pretrained_model = ClipTextEmbedder()
            pretrained_model.eval()
            classifier = LinearProbeClassifier(feature_dim=768, pooling=False)
            classifier.to(DEVICE)
        elif model_name == "img_emb":
            pretrained_model = ClipImgEmbedder()
            pretrained_model.eval()
            classifier = LinearProbeClassifier(feature_dim=1024, pooling=False)
            classifier.to(DEVICE)
        else:
            cfg = OmegaConf.load(model_config["cfg_path"])
            pretrained_model = init_model(cfg, model_config["ckpt_path"], DEVICE)
            classifier = LinearProbeClassifier(layer_idx=model_config["layer_idx"])
            classifier.to(DEVICE)

        # Call the DataLoader handler and DataLoaders
        handler = LatentDatasetLoader(batch_size=model_config["batch_size"])
        train_dataloader = handler.get_dataloader(split="train")
        test_dataloader = handler.get_dataloader(split="val")
        val_dataloader = handler.get_dataloader(split="val", is_reduced=True)

        # Train the classifier
        try:
            train(
                pretrained_model,
                classifier,
                train_dataloader,
                val_dataloader,
                test_dataloader,
                (model_name, model_config),
                DEVICE,
            )
        except Exception as e:
            logger.exception(f"An error occurred during training: {e}. Skipping model {model_name}...")
