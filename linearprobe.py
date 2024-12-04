import argparse
import os
from datetime import datetime
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
import hydra
from huggingface_hub import get_token

from diffusion.model.classifier import FeatureExtractor, LinearProbeClassifier
from dataset.dataset_preprocessing import DatasetLoader

import torch
from torch import nn
import numpy as np
import wandb


wandb.login()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='path of cached imagenet data'
    )
    parser.add_argument(
        '--cfg_path', type=str, required=True,
        help='path of model config file (.yaml)'
    )
    parser.add_argument(
        '--ckpt_pth', type=str, required=True,
        help='path of model checkpoint'
    )
    parser.add_argument(
        '--timestep', type=float, required=True,
        help='fixed timestep for linear probe (normalized)'
    )
    parser.add_argument(
        '--layer_num', type=int, required=True,
        help='transformer layer number for feature extraction'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
    )
    parser.add_argument(
        '--epochs', type=int, default=30,
    )
    parser.add_argument(
        '--batch_size', type=int, default=100,
    )
    parser.add_argument(
        '--eval_interval', type=int, default=5,
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='path to save train checkpoints'
    )
    parser.add_argument(
        '--model_name', type=str, default="unclip",
        help='name of model for lienar probing'
    )

    args = parser.parse_args()
    return args

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
            caption = [""] * imgs.size(0)
            imgs, targets = imgs.to(args.device), targets.to(args.device)
            output = classifier(x=imgs, txt=caption)

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

    # save checkpoint every 10th epoch
    if (i + 1) % 10 == 0:
        save_file = os.path.join(args.output_dir, f"epoch_{i+1}.pth")
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
            caption = [""] * imgs.size(0)
            output = classifier(x=imgs, txt=caption)

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

def get_dataloader(args, split):
    handler = DatasetLoader(
        hf_token=get_token(),
        cache_dir=args.data_dir,
        preprocessed_data_dir="",
        batch_size=args.batch_size
    )
    return handler.make_dataloader(split)


if __name__ == '__main__':
    args = parse_args()
    args.device = 'cuda'

    cfg = OmegaConf.load(args.cfg_path)
    cfg.fixed_t = args.timestep
    model = init_model(cfg, args.ckpt_pth)
    feature_extractor = FeatureExtractor(model, layer_num=args.layer_num)
    classifier = LinearProbeClassifier(feature_extractor)
    classifier.to(args.device)
    
    train_dataloader = get_dataloader(args, "train")
    test_dataloader = get_dataloader(args, "val")
    train(classifier, train_dataloader, test_dataloader, args)

    