EPOCHS, BATCH_SIZE, EVAL_INTERVAL, TIMESTEP, LAYER_NUM = 1, 32, 1000, 0.75, 14
ROOT_DIR = "/home/z004x5av/repos/ivs_diffusion"
data_config = {
    "cache_dir": f"{ROOT_DIR}/data/imagenet",
}
latent_data_config = {
    "train_shard_path": f"{ROOT_DIR}/data/imagenet_latent/train",
    "val_shard_path": f"{ROOT_DIR}/data/imagenet_latent/val",
}
models = {
    "unclip": {
        "cfg_path": f"{ROOT_DIR}/configs/model/rf_dit_unclip.yaml",
        "ckpt_path": f"{ROOT_DIR}/model_checkpoints/unclip.pt",
        "timestep": TIMESTEP,
        "layer_num": LAYER_NUM,
        "layer_start": 14,
        "feat_output_dir": f"{ROOT_DIR}/data/features/",
        "lr": 0.001,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "eval_interval": EVAL_INTERVAL,
        "output_dir": f"{ROOT_DIR}/model_checkpoints/probing/",
        "device": "cuda",
        "conditioning": "image",
    },
    "t2i": {
        "cfg_path": f"{ROOT_DIR}/configs/model/rf_dit_t2i.yaml",
        "ckpt_path": f"{ROOT_DIR}/model_checkpoints/t2i.pt",
        "timestep": TIMESTEP,
        "layer_num": LAYER_NUM,
        "layer_start": 14,
        "feat_output_dir": f"{ROOT_DIR}/data/features/",
        "lr": 0.001,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "eval_interval": EVAL_INTERVAL,
        "output_dir": f"{ROOT_DIR}/model_checkpoints/probing/",
        "device": "cuda",
        "conditioning": "text",
    },
    "t2i_uncond": {
        "cfg_path": f"{ROOT_DIR}/configs/model/rf_dit_t2i.yaml",
        "ckpt_path": f"{ROOT_DIR}/model_checkpoints/t2i.pt",
        "timestep": TIMESTEP,
        "layer_num": LAYER_NUM,
        "layer_start": 14,
        "feat_output_dir": f"{ROOT_DIR}/data/features/",
        "lr": 0.001,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "eval_interval": EVAL_INTERVAL,
        "output_dir": f"{ROOT_DIR}/model_checkpoints/probing/",
        "device": "cuda",
        "conditioning": "unconditional",
    }
}