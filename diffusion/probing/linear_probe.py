import os
import sys
sys.path.append(os.path.abspath("/home/z004x5av/repos/ivs_diffusion/utils")) # TODO: find alternative
from utils.logging import logger
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
import hydra
from diffusion.probing.utils import parse_args, precompute_dataset_len, get_toy_data
from diffusion.probing.classifier import feature_extractor, LinearProbeClassifier
from diffusion.dataset.dataset_preprocessing import DatasetLoader
from huggingface_hub import login, get_token

import torch
from torch import nn
import wandb

wandb.login()
DEBUG_MODE = False

def init_model(cfg: DictConfig, ckpt_pth: str):
    logger.info("Initializing model...")
    model = hydra.utils.instantiate(cfg)
    model = model.cuda()
    model.load_state_dict(
        {k: v for k, v in torch.load(ckpt_pth, map_location='cuda').items() if not 'ae' in k},
        strict=False)
    model.eval()
    logger.info("Model initialized and checkpoint loaded successfully.")
    return model

def train(classifier, train_dataloader, test_dataloader, args):
    run_name = f'{args.model_name}_{args.layer_num}'
    os.makedirs(os.path.join(args.output_dir, run_name), exist_ok=True)
    wandb.init(project='linear_probe', name=run_name)
    wandb.watch(classifier, log='all', log_freq=args.batch_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.classifier.parameters(), lr=args.lr)
    classifier.train()

    # Precompute dataloader len, since Webloader streams data
    if DEBUG_MODE:
        trainloader_len = len(train_dataloader)
    else:
        trainloader_len = precompute_dataset_len(args.batch_size)

    # Initialize best validation accuracy
    best_val_accuracy = 0.0  
    best_model_path = os.path.join(args.output_dir, f"best_model_{args.model_name}.pth")

    logger.info(f"Starting training for {args.epochs} epochs...")
    for epoch in tqdm(range(args.epochs),
                      desc="Epochs", 
                      unit=" Epoch", 
                      colour="yellow"):
        epoch += 1
        epoch_loss = 0
        batch_num = 0 # TODO: erase after check
        logger.info(f"Starting epoch {epoch}/{args.epochs}...")
        for (imgs, targets) in tqdm(iterable=train_dataloader, 
                                    total=trainloader_len,
                                    desc="Batches in training",
                                    unit=" Batch",
                                    colour="green",
                                    leave=False):
            caption = [""] * imgs.size(0) # NOTE: Is this necessary?
            imgs, targets = imgs.to(args.device), targets.to(args.device)
            # Extract features
            features = feature_extractor(model, imgs, args.layer_start, args.feat_output_path)
            # Make predictions
            output = classifier(features)
            # Compute loss, gradients and update weights
            loss = loss_fn(output, targets)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_num += 1 # TODO: erase after check

        logger.info(f"Epoch {epoch} completed.")
        # Sanity check: Validate the batch number
        logger.info(f"Batches per epoch: {batch_num}")
        if batch_num != trainloader_len:
            logger.exception("Estimated batch number not same as actual batch number!")

        # Compute epoch train loss
        epoch_loss /= trainloader_len

        # Log epoch metrics to wandb
        wandb.log({
            "epoch": epoch,
            "epoch_loss": epoch_loss,
        })

        # Evaluate model at specified interval
        if (epoch) % args.eval_interval == 0:
            logger.info(f"Evaluating model at epoch {epoch}...")
            val_accuracy = test(classifier, test_dataloader, args, "val", epoch)

            # Save if best performing model until now
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                save_dict = {
                    "classifier": classifier.classifier.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_accuracy": best_val_accuracy,
                }
                torch.save(save_dict, best_model_path)
                logger.info(f"New best model saved at {best_model_path} with val_accuracy: {val_accuracy:.4f} at epoch {epoch}.")


        # Save checkpoint every 10th epoch
        if (epoch) % 10 == 0:
            save_file = os.path.join(args.output_dir, f"epoch_{epoch}_{args.model_name}.pth")
            save_dict = {
                "classifier": classifier.classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(save_dict, save_file)
            logger.info(f"Model checkpoint saved at {save_file}.")

    logger.info("Training completed.")
    wandb.finish()

def test(classifier, dataloader, args, split="test", epoch=None):
    logger.info(f"Starting evaluation on {split} set...")
    classifier.eval()
    total_top1 = 0
    total_top5 = 0
    total = 0

    with torch.no_grad():
        for (imgs, targets) in tqdm(dataloader, desc="Batches in validation", unit=" Batch"):
            imgs, targets = imgs.to(args.device), targets.to(args.device)
            caption = [""] * imgs.size(0) # NOTE: is caption necessary?
             # Extract features
            features = feature_extractor(model, imgs, args.layer_start, args.feat_output_path)
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

        accuracy = total_top1 / total

        if split == "val":
            wandb.log({
                "val_accuracy": accuracy,
                "top-5": total_top5 / total,
                "epoch": epoch,
            })

        logger.info(f'{split} accuracy: {accuracy:.4f}')
        return accuracy

if __name__ == '__main__':
    # Define args and initialize the model  
    args = parse_args()
    cfg = OmegaConf.load(args.cfg_path)
    # cfg.fixed_t = args.timestep # TODO
    model = init_model(cfg, args.ckpt_pth)
    
    # Define device
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Exiting.")
        raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and drivers installed.")
    args.device = "cuda"

    # Retrieve the HF credentials
    HF_TOKEN = ""
    login(token=HF_TOKEN)
    # Call the DataLoader handler and DataLoaders
    handler = DatasetLoader(
        hf_token=get_token(),
        cache_dir="../../data/shards",
        preprocessed_data_dir="../../data/preprocessed_data",
        batch_size=args.batch_size
    )
    # If True, use toy data to expedite script validation
    if DEBUG_MODE:
        train_loader, test_loader = get_toy_data(args.batch_size)
    else:
        train_loader = handler.make_dataloader(split="train")
        test_loader = handler.make_dataloader(split="val")

    # Instantiate the Linear Classifier
    classifier = LinearProbeClassifier(layer_idx=args.layer_num)
    classifier.to(args.device)

    # Train the classifier
    try:
        train(classifier, train_loader, test_loader, args)
    except Exception as e:
        logger.exception("An error occurred during training.")
        save_file = os.path.join(args.output_dir, f"interrupted_{args.model_name}.pth")
        torch.save(classifier.classifier.state_dict(), save_file)
        logger.info(f"Checkpoint saved at {save_file} due to training interruption.")
