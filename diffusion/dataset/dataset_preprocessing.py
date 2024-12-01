from huggingface_hub import login, get_token
import webdataset as wds
from torch.utils.data import DataLoader
import numpy as np
import torch

HF_TOKEN="..." # TODO: Add token as secret value
login(token=HF_TOKEN)

hf_token = get_token()
seed = 42

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

url = "https://huggingface.co/datasets/timm/imagenet-1k-wds/resolve/main/imagenet1k-train-{{0000..1023}}.tar"
url = f"pipe:curl -s -L {url} -H 'Authorization:Bearer {hf_token}'"

dataset = wds.WebDataset(url, shardshuffle=100, seed=seed).decode()

def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)
dataloader = DataLoader(dataset, 
                        batch_size=64, 
                        num_workers=4, 
                        worker_init_fn=worker_init_fn, 
                        )