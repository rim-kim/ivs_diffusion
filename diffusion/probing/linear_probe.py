import os
import sys
sys.path.append(os.path.abspath("/home/z004x5av/repos/ivs_diffusion/utils")) # TODO: find alternative
from utils.logging import logger
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
import hydra
from diffusion.probing.utils import get_toy_data
from diffusion.probing.classifier import extract_features, LinearProbeClassifier
from diffusion.dataset.dataset_preprocessing import DatasetLoader
from huggingface_hub import login, get_token
from configs.tokens.tokens import HF_TOKEN
from typing import Literal

import torch
import wandb
from wandb.sdk.wandb_config import Config


hyperparams = {
    "cfg_path": "/home/z004x5av/repos/ivs_diffusion/configs/model/rf_dit_unclip.yaml",
    "ckpt_pth": "/home/z004x5av/repos/ivs_diffusion/model_checkpoints/unclip.pt",
    "timestep": 0.75,
    "layer_num": 14,
    "layer_start": 14,
    "feat_output_dir": "/home/z004x5av/repos/ivs_diffusion/data/features/",
    "lr": 1e-3,
    "epochs": 30,
    "batch_size": 64,
    "eval_interval": 2,
    "output_dir": "/home/z004x5av/repos/ivs_diffusion/model_checkpoints/probing/",
    "model_name": "unclip",
    "device": "cuda"
}

wandb.login()
DEBUG_MODE = False

def init_model(cfg: DictConfig, ckpt_pth: str) -> torch.nn.Module:
    """
    Initializes the model from a given configuration and checkpoint file.

    :param cfg: The configuration object containing model setup details.
    :param ckpt_pth: Path to the model checkpoint file.
    :return: The initialized model with the checkpoint loaded.
    """
    logger.info("Initializing model...")
    model = hydra.utils.instantiate(cfg)
    model = model.cuda()
    model.load_state_dict(
        {k: v for k, v in torch.load(ckpt_pth, map_location='cuda').items() if not 'ae' in k},
        strict=False)
    model.eval()
    logger.info("Model initialized and checkpoint loaded successfully.")
    return model

def train(
    classifier: LinearProbeClassifier, 
    train_dataloader: torch.utils.data.DataLoader, 
    test_dataloader: torch.utils.data.DataLoader
    ) -> None:
    """
    Trains the linear probe classifier using extracted features from the model.

    :param classifier: The linear probe classifier to be trained.
    :param train_dataloader: DataLoader for the training dataset.
    :param test_dataloader: DataLoader for the validation dataset.
    """
    run_name = f"{hyperparams['model_name']}_{hyperparams['layer_num']}"
    os.makedirs(os.path.join(hyperparams['output_dir'], run_name), exist_ok=True)
    wandb.init(project='linear_probe', name=run_name, config=hyperparams)
    config = wandb.config
    wandb.define_metric(name="train_batch_loss", step_metric="train_step")
    wandb.define_metric(name="top1_accuracy", step_metric="val1_step")
    wandb.define_metric(name="top5_accuracy", step_metric="val2_step")
    wandb.define_metric(name="epoch", step_metric="epoch_step")
    wandb.watch(classifier, log='all', log_freq=config.batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.classifier.parameters(), lr=config.lr)
    classifier.train()

    # Initialize best validation accuracy
    best_weighted_accuracy = 0.0  
    best_model_path = os.path.join(config.output_dir, f"best_model_{config.model_name}.pth")

    logger.info(f"Starting training for {config.epochs} epochs...")
    for epoch in range(1, config.epochs+1):
        batch_num = 0
        logger.info(f"Starting epoch {epoch}/{config.epochs}...")
        for (imgs, targets) in tqdm(iterable=train_dataloader,
                                    total=train_dataloader.nsamples,
                                    desc="Batches in training",
                                    unit=" Batch",
                                    colour="blue",
                                    leave=False):
            caption = [""] * imgs.size(0) # NOTE: Is this necessary?
            imgs, targets = imgs.to(config.device), targets.to(config.device)
            batch_num += 1
            # Extract features
            features = extract_features(model, imgs, config.layer_start, config.feat_output_dir, batch_num)
            # Make predictions
            output = classifier(features)
            # Compute loss, gradients and update weights
            loss = loss_fn(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log batch metrics to wandb
            wandb.log({
                "train_step": batch_num,
                "train_batch_loss": loss.item(),
            })
            wandb.log({
                "epoch_step": batch_num,
                "epoch": epoch})

            # Evaluate model at specified batch interval
            if (batch_num) % config.eval_interval == 0:
                logger.info(f"Evaluating model at batch number {batch_num}...")
                top1_accuracy, top5_accuracy = test(classifier, test_dataloader, config, batch_num, "val")

                # Save if best performing model until now
                weighted_accuracy = (top1_accuracy * 0.7) + (top5_accuracy * 0.3)
                if weighted_accuracy > best_weighted_accuracy:
                    best_weighted_accuracy = weighted_accuracy
                    save_dict = {
                        "classifier": classifier.classifier.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "batch": batch_num,
                        "best_weighted_accuracy": best_weighted_accuracy,
                    }
                    torch.save(save_dict, best_model_path)
                    logger.info(f"New best model saved at {best_model_path} with weighted accuracy {best_weighted_accuracy:.4f} at epoch {epoch} and batch idx {batch_num}.")


        logger.info(f"Epoch {epoch} completed with {batch_num} batches.")

    logger.info("Training completed.")
    wandb.finish()

def test(
    classifier: LinearProbeClassifier, 
    dataloader: torch.utils.data.DataLoader, 
    config: Config, 
    batch_num: int, 
    split: Literal["val", "test"] = "test"
    ) -> tuple[float, float]:
    """
    Evaluates the classifier on the specified dataset split.

    :param classifier: The linear probe classifier to be evaluated.
    :param dataloader: DataLoader for the evaluation dataset.
    :param config: The configuration object containing hyperparameters.
    :param batch_num: The current batch number during evaluation.
    :param split: Specifies whether evaluation is on "val" or "test" data. Defaults to "test".
    :return: Top-1 and Top-5 accuracy scores.
    """
    logger.info(f"Starting evaluation on {split} set...")
    classifier.eval()
    total_top1 = 0
    total_top5 = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(tqdm(iterable=dataloader,
                                                         total=dataloader.nsamples,
                                                         desc="Batches in validation", 
                                                         unit=" Batch",
                                                         colour="yellow"),
                                                         start=1):
            imgs, targets = imgs.to(config.device), targets.to(config.device)
            caption = [""] * imgs.size(0) # NOTE: is caption necessary?
             # Extract features
            features = extract_features(model, imgs, config.layer_start, config.feat_output_dir, batch_idx, mode=split)
            # Make predictions
            output = classifier(features)
            # Compute metrics
            _, top1_preds = torch.max(output, dim=-1)
            total_top1 += (top1_preds == targets).sum().item()
            _, top5_preds = torch.topk(output, 5, dim=-1)
            for i in range(targets.size(0)):
                if targets[i].item() in top5_preds[i].tolist():
                    total_top5 += 1
            total += targets.size(0)

        top1_accuracy = total_top1 / total
        top5_accuracy = total_top5 / total

        if split == "val":
            wandb.log({
                "val1_step": batch_num // config.eval_interval,
                "top1_accuracy": top1_accuracy,
            })
            wandb.log({
                "val2_step": batch_num // config.eval_interval,
                "top5_accuracy": top5_accuracy,
            })

        logger.info(f'{split}: top1 accuracy {top1_accuracy:.4f} and top5 accuracy {top5_accuracy:.4f}')
        return top1_accuracy, top5_accuracy

if __name__ == '__main__':
    # Define args and initialize the model  
    cfg = OmegaConf.load(hyperparams["cfg_path"])
    # cfg.fixed_t = config.timestep # TODO
    model = init_model(cfg, hyperparams["ckpt_pth"])
    
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
        batch_size=hyperparams["batch_size"],
        epochs=hyperparams["epochs"],
    )
    # If True, use toy data to expedite script validation
    if DEBUG_MODE:
        train_dataloader, test_dataloader = get_toy_data(hyperparams["batch_size"])
    else:
        train_dataloader = handler.make_dataloader(split="train")
        test_dataloader = handler.make_dataloader(split="val")

    # Instantiate the Linear Classifier
    classifier = LinearProbeClassifier(layer_idx=hyperparams["layer_num"])
    classifier.to(hyperparams["device"])

    # Train the classifier
    try:
        train(classifier, train_dataloader, test_dataloader)
    except Exception as e:
        logger.exception(f"An error occurred during training: {e}.")
        save_file = os.path.join(hyperparams["output_dir"], f"interrupted_{hyperparams['model_name']}.pth")
        torch.save(classifier.classifier.state_dict(), save_file)
        logger.info(f"Checkpoint saved at {save_file} due to training interruption.")
