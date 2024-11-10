# Diffusion
Basics codebase for inference on CompVis internal diffusion transformer checkpoints. Do not distribute.

## Setup
- Install PyTorch, at least v2.5 
- Install [k-diffusion](https://github.com/crowsonkb/k-diffusion) from source by cloning and running `pip install -e .` (optionally with `--config-settings editable_mode=strict` to get linter support in VSCode) in the repo root
- Install all the required packages: `pip install -r requirements.txt`

## Checkpoints
You can download the t2i and unclip checkpoint from [here](https://drive.google.com/drive/folders/1-cpr37zf_O7OBsj3a0a_s2f_hXw_s29D?usp=sharing).

## Sampling
There are example sampling notebooks available in the `notebooks` folder.


## Code Credit
- A lot of model code is adapted from k-diffusion by Katherine Crowson (MIT)
