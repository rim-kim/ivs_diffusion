from typing import Literal, Tuple, Union

import hydra
import torch
from omegaconf import DictConfig
from torch.nn.functional import cross_entropy
from tqdm import tqdm
from wandb.sdk.wandb_config import Config
from webdataset import WebLoader

import wandb
from dataset.utils import get_captions
from diffusion.model.rf import LatentRF2D
from diffusion.model.t2i import T2ILatentRF2d
from diffusion.model.unclip import UnclipLatentRF2d
from probing.classifier import LinearProbeClassifier, extract_features
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
    pretrained_model = hydra.utils.instantiate(cfg).to(device)
    pretrained_model.load_state_dict(
        {k: v for k, v in torch.load(ckpt_path, map_location=device).items() if not "ae" in k}, strict=False
    )
    pretrained_model.eval()
    logger.info(f"Model {pretrained_model.__class__.__name__} initialized and checkpoint loaded successfully.")
    return pretrained_model


def init_training_setup(classifier: LinearProbeClassifier, model_name: str, model_config: dict, device: str) -> Config:
    """
    Initializes wandb logging and defines training-related metrics.

    :param classifier: The linear probe classifier to be trained.
    :param model_name: The name of the pre-trained model.
    :param model_config: Dictionary containing model hyperparameters.
    :param device: The device for training (e.g., "cuda").
    :return: The wandb configuration object.
    """
    run_name = (
        f"{model_name}"
        if model_name in ["txt_emb", "img_emb"]
        else f"{model_name}_{model_config['layer_idx']}_{model_config['timestep']}"
    )

    wandb.init(project="trial_linear_probe", name=run_name, config=model_config)
    wandb.config["device"] = device
    wandb.define_metric(name="train_batch_loss", step_metric="train_step")
    wandb.define_metric(name="top1_accuracy", step_metric="val1_step")
    wandb.define_metric(name="top5_accuracy", step_metric="val2_step")
    wandb.define_metric(name="val_avg_loss", step_metric="val3_step")
    wandb.define_metric(name="epoch", step_metric="epoch_step")
    wandb.watch(classifier, log="all", log_freq=wandb.config.batch_size)

    return wandb.config


def extract_features_wrapper(
    pretrained_model: Union[T2ILatentRF2d, UnclipLatentRF2d, LatentRF2D],
    model_name: str,
    inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    config: Config,
) -> torch.Tensor:
    """
    Extracts features from the pre-trained model based on the model type.

    :param pretrained_model: The pre-trained model from which features are extracted.
    :param model_name: The name of the model.
    :param inputs: Tuple containing (images, targets, clip embeddings).
    :param config: The configuration object containing hyperparameters.
    :return: Extracted features as a tensor.
    """
    imgs, targets, clip_embds = inputs
    if model_name == "txt_emb":
        return pretrained_model(get_captions(targets))[:, -1, :].to(config.device)
    elif model_name == "img_emb":
        return clip_embds
    return extract_features(
        pretrained_model, model_name, config.layer_idx, (imgs, targets, clip_embds), config.timestep
    )


def train_one_batch(
    pretrained_model: Union[T2ILatentRF2d, UnclipLatentRF2d, LatentRF2D],
    classifier: LinearProbeClassifier,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_name: str,
    config: Config,
    batch_idx: int,
) -> float:
    """
    Processes a single training batch, computes loss, and updates the classifier.

    :param pretrained_model: The pre-trained model from which features are extracted.
    :param classifier: The linear probe classifier being trained.
    :param batch: Tuple containing (images, targets, clip embeddings).
    :param loss_fn: The loss function used for training.
    :param optimizer: Optimizer for updating model parameters.
    :param model_name: The name of the model.
    :param config: The configuration object containing hyperparameters.
    :param batch_idx: The current batch index.
    """
    imgs, targets, clip_embds = batch
    imgs, targets, clip_embds = imgs.to(config.device), targets.to(config.device), clip_embds.to(config.device)

    features = extract_features_wrapper(pretrained_model, model_name, (imgs, targets, clip_embds), config)
    output = classifier(features)

    loss = loss_fn(output, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    wandb.log({"train_step": batch_idx, "train_batch_loss": loss.item()})


def train(
    pretrained_model: Union[T2ILatentRF2d, UnclipLatentRF2d, LatentRF2D],
    classifier: LinearProbeClassifier,
    train_dataloader: WebLoader,
    val_dataloader: WebLoader,
    test_dataloader: WebLoader,
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
    config = init_training_setup(classifier, model_name, model_config, device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.classifier.parameters(), lr=config.lr)
    classifier.train()

    logger.info(f"Starting training for {config.epochs} epochs...")
    for epoch in range(1, config.epochs + 1):
        batch_num = 0
        for batch_idx, batch in enumerate(
            tqdm(
                iterable=train_dataloader,
                total=train_dataloader.nsamples,
                desc="Batches in training",
                unit=" Batch",
                colour="blue",
                leave=False,
            )
        ):
            train_one_batch(pretrained_model, classifier, batch, loss_fn, optimizer, model_name, config, batch_idx)
            wandb.log({"epoch_step": batch_idx, "epoch": epoch})

            if (batch_idx + 1) % config.eval_interval == 0:
                logger.info(f"Evaluating model at batch number {batch_idx + 1}...")
                test(pretrained_model, classifier, val_dataloader, config, batch_idx, model_name, "val")

            batch_num += 1
        logger.info(f"Epoch {epoch} completed with {batch_num} batches.")

    logger.info("Training completed. Starting full validation...")
    top1_accuracy, top5_accuracy, avg_loss = test(
        pretrained_model, classifier, test_dataloader, config, -1, model_name, "test"
    )
    wandb.log({"full_top1_accuracy": top1_accuracy, "full_top5_accuracy": top5_accuracy, "full_avg_loss": avg_loss})
    wandb.finish()


def compute_metrics(output: torch.Tensor, targets: torch.Tensor) -> Tuple[int, int, float]:
    """
    Computes Top-1 and Top-5 accuracy and loss.

    :param output: The classifier's output predictions.
    :param targets: Ground truth labels.
    :return: Tuple containing (Top-1 accuracy, Top-5 accuracy, loss).
    """
    _, top1_preds = torch.max(output, dim=-1)
    top1_correct = (top1_preds == targets).sum().item()

    _, top5_preds = torch.topk(output, 5, dim=-1)
    top5_correct = sum(targets[i].item() in top5_preds[i].tolist() for i in range(targets.size(0)))

    loss = cross_entropy(output, targets)
    return top1_correct, top5_correct, loss.item()


def test(
    pretrained_model: Union[T2ILatentRF2d, UnclipLatentRF2d, LatentRF2D],
    classifier: LinearProbeClassifier,
    dataloader: WebLoader,
    config: Config,
    batch_idx: int,
    model_name: str,
    split: Literal["val", "test"] = "test",
) -> Tuple[float, float, float]:
    """
    Evaluates the classifier on the specified dataset split.

    :param pretrained_model: The pre-trained model from which features are extracted.
    :param classifier: The linear probe classifier to be evaluated.
    :param dataloader: DataLoader for the evaluation dataset.
    :param config: The configuration object containing hyperparameters.
    :param batch_idx: The current batch index during evaluation.
    :param model_name: The name of the model configuration.
    :param split: Specifies whether evaluation is on "val" or "test" data.
    :return: Tuple containing (Top-1 accuracy, Top-5 accuracy, average loss).
    """
    logger.info(f"Starting evaluation on {split} set...")
    classifier.eval()

    total_top1, total_top5, total_loss, total_samples = 0, 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(
            iterable=dataloader, total=dataloader.nsamples, desc="Batches in validation", unit=" Batch", colour="yellow"
        ):
            imgs, targets, clip_embds = batch
            imgs, targets, clip_embds = imgs.to(config.device), targets.to(config.device), clip_embds.to(config.device)

            features = extract_features_wrapper(pretrained_model, model_name, (imgs, targets, clip_embds), config)
            output = classifier(features)

            top1_correct, top5_correct, loss = compute_metrics(output, targets)
            total_top1 += top1_correct
            total_top5 += top5_correct
            total_loss += loss * targets.size(0)
            total_samples += targets.size(0)

    top1_accuracy = total_top1 / total_samples
    top5_accuracy = total_top5 / total_samples
    avg_loss = total_loss / total_samples

    if split == "val":
        wandb.log(
            {
                "val1_step": (batch_idx + 1) // config.eval_interval,
                "top1_accuracy": top1_accuracy,
            }
        )
        wandb.log(
            {
                "val2_step": (batch_idx + 1) // config.eval_interval,
                "top5_accuracy": top5_accuracy,
            }
        )
        wandb.log(
            {
                "val3_step": (batch_idx + 1) // config.eval_interval,
                "avg_loss": avg_loss,
            }
        )

    logger.info(
        f"{split}: top1 accuracy {top1_accuracy:.4f}, top5 accuracy {top5_accuracy:.4f}, avg loss {avg_loss:.4f}"
    )
    return top1_accuracy, top5_accuracy, avg_loss
