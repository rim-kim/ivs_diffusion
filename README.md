- [Project Title](#project-title)
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

## Project Title
Investigating knowledge differences in unconditional and T2I diffusion models.

## Description
We investigate how different conditioning mechanisms in diffusion transformers, image-conditioned (unCLIP), text-conditioned (T2I), and unconditional, affect the acquisition and organization of world knowledge. Specifically, we examine whether conditioning obfuscates or enhances semantic richness in internal model representations. To achieve this, we extract internal feature representations from these three architectures and evaluate their semantic content by training a linear classifier on the ImageNet-1K dataset as a downstream classification task. This study aims to pinpoint where the most semantically rich representations reside and how conditioning influences knowledge acquisition in diffusion models of different conditioning.

## Installation

### Prerequisites
Ensure you have the following installed:
- [**Anaconda**](https://www.anaconda.com/products/distribution) or [**Miniconda**](https://docs.conda.io/en/latest/miniconda.html)
- [**Poetry**](https://python-poetry.org/docs/#installation) for dependency management

---

### Step 1: Clone the Repository
First, clone the repository to your local machine:
```sh
git clone https://github.com/rim-kim/ivs_diffusion.git
cd ivs_diffusion
```

### Step 2: Create and Activate the Conda Environment
Create a new Conda environment and activate it:
```sh
conda create --name myenv python=3.11
conda activate myenv
```
Replace _myenv_ with the desired environment name.

### Step 3: Install Poetry (if not already installed)
If Poetry is not installed globally, install it:
```sh
curl -sSL https://install.python-poetry.org | python3 -
```
To verify Poetry installation, run:
```sh
poetry --version
```

### Step 4: Configure Poetry to Work Inside the Conda Environment
By default, Poetry creates virtual environments in a separate location. Since you're using Conda, configure Poetry to use the current Conda environment:
```sh
poetry config virtualenvs.create false
```

### Step 5: Install Dependencies
Once inside the Conda environment, install the project dependencies using:
```sh
poetry install
```

### Step 6: Verify Installation
After installation, check that everything is set up correctly by running:
```sh
poetry run python -c "import torch; print(torch.__version__)"
```

### Step 7: Set Up Hugging Face Token
To access the **ImageNet dataset** from Hugging Face, you need to provide your **Hugging Face access token**.

 - Create `configs/tokens/tokens.py` and add:
     ```python
     HF_TOKEN = "your_huggingface_token_here"
     ```
  - This file is already included in `.gitignore`, so your token will not be accidentally pushed to GitHub.


### Step 8: Enable Pre-commit Hooks
To activate the pre-commit hooks, run:
```sh
poetry run pre-commit install
```


## Usage

### Step 1: Pre-computing latent representations and CLIP image embeddings
Aim is to make the setup internet-independent and to speed up the training process.
The computed representations are stored in compressed `.tar` shards.

Run the pre-computation using:
```sh
python -m dataset.utils --batch_size 64 --samples_per_shard 1000 --dataset both
```

### Step 2: Running the Project
Extracts frozen features from models listed in `hyperparameters.py` using precomputed latent representations of ImageNet images and their respective conditioning (if applicable). Then, trains a linear classifier on the extracted features.

Run the project using:
```sh
python -m main_probe --models all --timestep 0.25 --layer_idx 16
```

## Authors
- Jeongrim Kim – Lead Developer - [GitHub](https://github.com/rim-kim)
- Ioannis Partalas – Lead Developer - [GitHub](https://github.com/i-partalas)

## Acknowledgments
- We thank **Kolja Bauer (LMU)** for his guidance, valuable feedback, and supervision throughout this project.
- The initially provided codebase included inference scripts and checkpoints that **belong to CompVis** and are restricted from distribution.
- A lot of model code from the initially provided codebase is adapted from **k-diffusion by Katherine Crowson (MIT)**.
