from typing import Literal, Tuple, Union
from omegaconf import DictConfig
import hydra
import torch
from torch.nn.functional import cross_entropy
from tqdm import tqdm
import wandb
from wandb.sdk.wandb_config import Config

from configs.path_configs.path_configs import MODEL_CKPT_PROBING_DIR
from diffusion.model.t2i import T2ILatentRF2d
from diffusion.model.unclip import UnclipLatentRF2d
from probing.classifier import extract_features, LinearProbeClassifier
from utils.logging import logger


wandb.login()

def init_model(cfg: DictConfig, ckpt_path: str, device: str) -> torch.nn.Module:
    """
    Initializes the pre-trained model from a given configuration and checkpoint file.

    :param cfg: The configuration object containing the pre-trained model setup details.
    :param ckpt_path: Path to the pre-trained model checkpoint file.
    :param device: The device on which the model will be initialized and run (e.g., "cuda").
    :return: The initialized pre-trained model with the checkpoint loaded.
    """
    logger.info("Initializing the pre-trained model...")
    pretrained_model = hydra.utils.instantiate(cfg)
    pretrained_model = pretrained_model.to(device)
    pretrained_model.load_state_dict(
        {k: v for k, v in torch.load(ckpt_path, map_location=device).items() if not 'ae' in k},
        strict=False)
    pretrained_model.eval()
    logger.info(f"Model {pretrained_model.__class__.__name__} initialized and checkpoint loaded successfully.")
    return pretrained_model


def train(
    pretrained_model: Union[T2ILatentRF2d, UnclipLatentRF2d],
    classifier: LinearProbeClassifier, 
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    model_data: Tuple[str, dict],
    device: str,
    ) -> None:
    """
    Trains the linear probe classifier using extracted features from the pre-trained model.

    :param pretrained_model: The pre-trained model from which features are extracted.
    :param classifier: The linear probe classifier to be trained.
    :param train_dataloader: DataLoader for the training dataset.
    :param val_dataloader: DataLoader for the validation dataset.
    :param test_dataloader: DataLoader for final full validation dataset.
    :param model_data: A tuple containing the model name and its configuration dictionary.
    :param device: The device on which computations will be performed (e.g., "cuda").
    """
    model_name, model_config = model_data
    run_name = f"{model_name}_{model_config['layer_num']}_{model_config['timestep']}"
    full_run_name = MODEL_CKPT_PROBING_DIR / run_name
    full_run_name.mkdir(exist_ok=True)
    wandb.init(project="linear_probe", name=run_name, config=model_config)
    wandb.config["device"] = device
    config = wandb.config
    wandb.define_metric(name="train_batch_loss", step_metric="train_step")
    wandb.define_metric(name="top1_accuracy", step_metric="val1_step")
    wandb.define_metric(name="top5_accuracy", step_metric="val2_step")
    wandb.define_metric(name="val_avg_loss", step_metric="val3_step")
    wandb.define_metric(name="epoch", step_metric="epoch_step")
    wandb.watch(classifier, log='all', log_freq=config.batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.classifier.parameters(), lr=config.lr)
    classifier.train()

    # Initialize best validation accuracy
    best_weighted_accuracy = 0.0
    best_model_path = full_run_name / f"best_model_{model_name}.pth"

    logger.info(f"Starting training for {config.epochs} epochs...")
    for epoch in range(1, config.epochs+1):
        batch_num = 0
        logger.info(f"Starting epoch {epoch}/{config.epochs}...")
        for batch_idx, data in enumerate(tqdm(iterable=train_dataloader,
                                    total=train_dataloader.nsamples,
                                    desc="Batches in training",
                                    unit=" Batch",
                                    colour="blue",
                                    leave=False)):
            imgs, targets, clip_embds = data
            imgs, targets, clip_embds = imgs.to(config.device), targets.to(config.device), clip_embds.to(config.device)
            # Extract features
            features = extract_features(pretrained_model, model_name, data, config.layer_start, config.timestep, batch_idx)
            # Make predictions
            output = classifier(features)
            # Compute loss, gradients and update weights
            loss = loss_fn(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log batch metrics to wandb
            wandb.log({
                "train_step": batch_idx,
                "train_batch_loss": loss.item(),
            })
            wandb.log({
                "epoch_step": batch_idx,
                "epoch": epoch})

            # Evaluate model at specified batch interval
            if (batch_idx + 1) % config.eval_interval == 0:
                logger.info(f"Evaluating model at batch number {batch_idx + 1}...")
                top1_accuracy, top5_accuracy, _ = test(pretrained_model, classifier, val_dataloader, config, batch_idx, model_name, "val")

                # Save if best performing model until now
                weighted_accuracy = (top1_accuracy * 0.7) + (top5_accuracy * 0.3)
                if weighted_accuracy > best_weighted_accuracy:
                    best_weighted_accuracy = weighted_accuracy
                    save_dict = {
                        "classifier": classifier.classifier.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "batch": batch_idx,
                        "best_weighted_accuracy": best_weighted_accuracy,
                    }
                    torch.save(save_dict, best_model_path)
                    logger.info(f"New best model saved at {best_model_path} with weighted accuracy {best_weighted_accuracy:.4f} at epoch {epoch} and batch idx {batch_idx+1}.")

            batch_num += 1
        logger.info(f"Epoch {epoch} completed with {batch_num} batches.")

    logger.info("Training completed.")

    # Log final full validation
    logger.info("Starting full validation...")
    top1_accuracy, top5_accuracy, avg_loss = test(pretrained_model, classifier, test_dataloader, config, -1, model_name, "test")
    wandb.log({
        "full_top1_accuracy": top1_accuracy,
        "full_top5_accuracy": top5_accuracy,
        "full_avg_loss": avg_loss
    })
    wandb.finish()

def test(
    pretrained_model: Union[T2ILatentRF2d, UnclipLatentRF2d],
    classifier: LinearProbeClassifier, 
    dataloader: torch.utils.data.DataLoader, 
    config: Config, 
    batch_idx: int,
    model_name: str,
    split: Literal["val", "test"] = "test"
    ) -> Tuple[float, float]:
    """
    Evaluates the classifier on the specified dataset split.

    :param pretrained_model: The pre-trained model from which features are extracted.
    :param classifier: The linear probe classifier to be evaluated.
    :param dataloader: DataLoader for the evaluation dataset.
    :param config: The configuration object containing hyperparameters.
    :param batch_idx: The current batch index during evaluation.
    :param model_name: The name of the model configuration.
    :param split: Specifies whether evaluation is on "val" or "test" data. Defaults to "test".
    :return: Top-1 and Top-5 accuracy scores and average loss.
    """
    logger.info(f"Starting evaluation on {split} set...")
    classifier.eval()
    total_top1 = 0
    total_top5 = 0
    total_loss = 0
    total = 0

    with torch.no_grad():
        for (imgs, targets) in tqdm(iterable=dataloader,
                                                         total=dataloader.nsamples,
                                                         desc="Batches in validation", 
                                                         unit=" Batch",
                                                         colour="yellow"):
            imgs, targets = imgs.to(config.device), targets.to(config.device)
             # Extract features
            features = extract_features(pretrained_model, model_name, (imgs, targets), config.layer_start, config.timestep, batch_idx, mode=split)
            # Make predictions
            output = classifier(features)
            # Compute metrics
            _, top1_preds = torch.max(output, dim=-1)
            total_top1 += (top1_preds == targets).sum().item()
            _, top5_preds = torch.topk(output, 5, dim=-1)
            for i in range(targets.size(0)):
                if targets[i].item() in top5_preds[i].tolist():
                    total_top5 += 1
            loss = cross_entropy(output, targets)
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)

        top1_accuracy = total_top1 / total
        top5_accuracy = total_top5 / total
        avg_loss = total_loss / total

        if split == "val":
            wandb.log({
                "val1_step": (batch_idx + 1) // config.eval_interval,
                "top1_accuracy": top1_accuracy,
            })
            wandb.log({
                "val2_step": (batch_idx + 1) // config.eval_interval,
                "top5_accuracy": top5_accuracy,
            })
            wandb.log({
                "val3_step": (batch_idx + 1) // config.eval_interval,
                "avg_loss": avg_loss,
            })

        logger.info(f'{split}: top1 accuracy {top1_accuracy:.4f} and top5 accuracy {top5_accuracy:.4f}')
        logger.info(f'{split}: average loss {avg_loss:.4f}')
        return top1_accuracy, top5_accuracy, avg_loss
