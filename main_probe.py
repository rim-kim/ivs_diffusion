import torch
from omegaconf import OmegaConf

from configs.hyperparameters.hyperparameters import DEVICE, models
from configs.warnings.suppress_warnings import suppress_warnings
from dataset.dataset_loaders import LatentDatasetLoader
from diffusion.model.modules.clip import ClipImgEmbedder, ClipTextEmbedder
from probing.classifier import LinearProbeClassifier
from probing.linear_probe import init_model, train
from utils.logging import logger

suppress_warnings()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Exiting.")
        raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and drivers installed.")
    torch.set_float32_matmul_precision("high")

    # Iterate over the model configs and hyperparams
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
