import os
from omegaconf import OmegaConf
from diffusion.probing.linear_probe import init_model, train
from utils.logging import logger
import torch
from huggingface_hub import login, get_token
from diffusion.dataset.dataset_preprocessing import DatasetLoader
from configs.tokens.tokens import HF_TOKEN
from configs.hyperparameters.hyperparameters import models
from diffusion.probing.classifier import LinearProbeClassifier
from diffusion.probing.utils import get_toy_data

DEBUG_MODE = False

if __name__ == '__main__':
    # Iterate over the model configs and hyperparams
    for model_name, model_config in models.items():

        # Define args and initialize the model  
        cfg = OmegaConf.load(model_config["cfg_path"])
        # cfg.fixed_t = config.timestep # TODO
        pretrained_model = init_model(cfg, model_config["ckpt_path"])
        
        # Define device
        if not torch.cuda.is_available():
            logger.error("CUDA is not available. Exiting.")
            raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and drivers installed.")

        # Retrieve the HF credentials
        login(token=HF_TOKEN)

        # Call the DataLoader handler and DataLoaders
        handler = DatasetLoader(
            hf_token=get_token(),
            cache_dir="../../data/shards",
            preprocessed_data_dir="../../data/preprocessed_data",
            batch_size=model_config["batch_size"],
            epochs=model_config["epochs"],
        )
        # If True, use toy data to expedite script validation
        if DEBUG_MODE:
            train_dataloader, test_dataloader = get_toy_data(model_config["batch_size"])
        else:
            train_dataloader = handler.make_dataloader(split="train")
            test_dataloader = handler.make_dataloader(split="val")

        # Instantiate the Linear Classifier
        classifier = LinearProbeClassifier(layer_idx=model_config["layer_num"])
        classifier.to(model_config["device"])

        # Train the classifier
        try:
            train(pretrained_model, classifier, train_dataloader, test_dataloader, (model_name, model_config))
        except Exception as e:
            logger.exception(f"An error occurred during training: {e}.")
            save_file = os.path.join(model_config["output_dir"], f"interrupted_{model_name}.pth")
            torch.save(classifier.classifier.state_dict(), save_file)
            logger.info(f"Checkpoint saved at {save_file} due to training interruption.")
