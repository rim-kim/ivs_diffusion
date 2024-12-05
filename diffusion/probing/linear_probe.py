import os
import sys
from datetime import datetime
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
import hydra
sys.path.append(os.path.abspath("/home/z004x5av/repos/ivs_diffusion")) # TODO: find alternative
from diffusion.probing.utils import parse_args
from diffusion.probing.classifier import feature_extractor, LinearProbeClassifier
from diffusion.dataset.dataset_preprocessing import DatasetLoader
from huggingface_hub import login, get_token

import torch
from torch import nn
import numpy as np
import wandb

wandb.login()

def init_model(cfg: DictConfig, ckpt_pth: str):
    model = hydra.utils.instantiate(cfg)
    model = model.cuda()
    model.load_state_dict(
        {k: v for k, v in torch.load(ckpt_pth, map_location='cuda').items() if not 'ae' in k},
        strict=False)
    model.eval()
    return model

def train(classifier, train_dataloader, test_dataloader, args):
    run_name = f'{args.model_name}_{args.layer_num}'
    os.makedirs(os.path.join(args.output_dir, run_name), exist_ok=True)
    wandb.init(project='linear_probe', name=run_name)
    wandb.watch(classifier, log='all', log_freq=args.batch_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.classifier.parameters(), lr=args.lr)
    classifier.train()
    batch_num = 0
    for i in range(args.epochs):
        for batch_idx, batch in enumerate(tqdm(train_dataloader, total=train_dataloader.nsamples)):
            imgs, targets = batch
            caption = [""] * imgs.size(0) # NOTE: Is this necessary?
            imgs, targets = imgs.to(args.device), targets.to(args.device)
            # Extract features
            features = feature_extractor(model, imgs, args.layer_start, args.feat_output_path)
            # Make predictions
            output = classifier(features)
            # Compute loss, gradients and update weights
            loss = loss_fn(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"Loss/train": loss.item(),
                       "epoch": int((batch_num+1) / train_dataloader.nsamples),
                       "batch_idx": batch_idx},)
            batch_num += 1

        if (i + 1) % args.eval_interval == 0:
            test(classifier, test_dataloader, args, "Val", i+1)

        # Save checkpoint every 10th epoch
        if (i + 1) % 10 == 0:
            save_file = os.path.join(args.output_dir, f"epoch_{i+1}_{args.model_name}.pth")
            save_dict ={
                        "classifier": classifier.classifier.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": i+1,
                        }
            torch.save(save_dict, save_file)                         

def test(classifier, dataloader, args, split="Test", epoch=None):
    model.eval()
    total_top1 = 0
    total_top5 = 0
    total = 0

    with torch.no_grad():
        for _, batch in tqdm(dataloader):
            imgs, targets = batch
            imgs, targets = imgs.to(args.device), targets.to(args.device)
            caption = [""] * imgs.size(0) # NOTE: is caption necessary?
            output = classifier(x=imgs)

            _, top1_preds = torch.max(output, dim=-1)
            total_top1 += (top1_preds == targets).sum().item()
            _, top5_preds = torch.topk(output, 5, dim=-1)
            for i in range(targets.size(0)):
                if targets[i].item() in top5_preds[i].tolist():
                    total_top5 += 1
            total += targets.size(0)

        if split == "Val":
            wandb.log({f"Accuracy": total_top1 / total,
                        "Top-5": total_top5 / total,
                        "epoch": epoch,})
        print(f'{split} accuracy: {total_top1 / total}, top-5: {total_top5 / total}')

if __name__ == '__main__':
    # Define args and initialize the model  
    args = parse_args()
    cfg = OmegaConf.load(args.cfg_path)
    # cfg.fixed_t = args.timestep # TODO
    model = init_model(cfg, args.ckpt_pth)

    # Define device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and drivers installed.")
    args.device = "cuda"

    # Retrieve the HF credentials
    HF_TOKEN = "" # TODO: Make secret value
    login(token=HF_TOKEN)

    # Call the DataLoader handler and DataLoaders
    handler = DatasetLoader(
        hf_token=get_token(),
        cache_dir="../../data/shards",
        preprocessed_data_dir="../../data/preprocessed_data",
        batch_size=args.batch_size
    )
    train_loader = handler.make_dataloader(split="train")
    test_loader = handler.make_dataloader(split="val")

    # Instantiate the Linear Classifier
    # TODO: Create new parse arg "layer_idx"
    classifier = LinearProbeClassifier(layer_idx=args.layer_num)
    classifier.to(args.device)

    # Train the classifier
    train(classifier, train_loader, test_loader, args)