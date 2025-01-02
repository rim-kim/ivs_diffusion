from huggingface_hub import login, get_token
from omegaconf import OmegaConf
import torch

from dataset.dataset_preprocessing import DatasetLoader
from dataset.latent_dataset import LatentDatasetLoader
from configs.path_configs.path_configs import TRAIN_LATENT_DIR, VAL_LATENT_DIR, MODEL_CKPT_PROBING_DIR
from configs.tokens.tokens import HF_TOKEN
from configs.hyperparameters.hyperparameters import models, DEVICE
from probing.classifier import LinearProbeClassifier
from probing.linear_probe import init_model, train
from utils.logging import logger


login(token=HF_TOKEN)

if __name__ == '__main__':
    if not torch.cuda.is_available():
            logger.error("CUDA is not available. Exiting.")
            raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and drivers installed.")

    # Iterate over the model configs and hyperparams
    for model_name, model_config in models.items():

        # Initialize the pre-trained model  
        cfg = OmegaConf.load(model_config["cfg_path"])
        pretrained_model = init_model(cfg, model_config["ckpt_path"], DEVICE)
            
        # Call the DataLoader handler and DataLoaders
        if model_name == "unclip":
            handler = DatasetLoader(
                hf_token=get_token(),
                batch_size=model_config["batch_size"],
                streaming=False,
            )
        else:
            handler = LatentDatasetLoader(
                train_shard_path=TRAIN_LATENT_DIR,
                val_shard_path=VAL_LATENT_DIR,
                batch_size=model_config["batch_size"],
            )
        train_dataloader = handler.make_dataloader(split="train")
        test_dataloader = handler.make_dataloader(split="val")
        val_dataloader = handler.make_dataloader(split="val", is_reduced=True)

        # Instantiate the Linear Classifier
        classifier = LinearProbeClassifier(layer_idx=model_config["layer_num"])
        classifier.to(DEVICE)

        # Train the classifier
        try:
            train(pretrained_model, classifier, train_dataloader, val_dataloader, test_dataloader, (model_name, model_config), DEVICE)
        except Exception as e:
            logger.exception(f"An error occurred during training: {e}.")
            save_file = MODEL_CKPT_PROBING_DIR / f"interrupted_{model_name}.pth"
            torch.save(classifier.classifier.state_dict(), save_file)
            logger.info(f"Checkpoint saved at {save_file} due to training interruption.")
