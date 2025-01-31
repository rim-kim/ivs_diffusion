from configs.path_configs.path_configs import MODEL_CONFIGS_DIR, MODEL_CKPT_DIR

EPOCHS, BATCH_SIZE, EVAL_INTERVAL, LR = 1, 64, 1000, 0.001
TIMESTEP, LAYER_IDX = 0.25, 14
DEVICE = "cuda"

models = {
    "txt_emb": {
        "lr": LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "eval_interval": EVAL_INTERVAL,
    },
    "img_emb": {
        "lr": LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "eval_interval": EVAL_INTERVAL,
    },
    "uncond": {
        "cfg_path": f"{MODEL_CONFIGS_DIR}/uncond.yaml",
        "ckpt_path": f"{MODEL_CKPT_DIR}/uncond_320k_steps.pt",
        "timestep": TIMESTEP,
        "layer_idx": LAYER_IDX,
        "lr": LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "eval_interval": EVAL_INTERVAL,
        "conditioning": "unconditional"
    },
    "unclip": {
        "cfg_path": f"{MODEL_CONFIGS_DIR}/rf_dit_unclip.yaml",
        "ckpt_path": f"{MODEL_CKPT_DIR}/unclip.pt",
        "timestep": TIMESTEP,
        "layer_idx": LAYER_IDX,
        "lr": LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "eval_interval": EVAL_INTERVAL,
        "conditioning": "image",
    },
    "t2i": {
        "cfg_path": f"{MODEL_CONFIGS_DIR}/rf_dit_t2i.yaml",
        "ckpt_path":  f"{MODEL_CKPT_DIR}/t2i.pt",
        "timestep": TIMESTEP,
        "layer_idx": LAYER_IDX,
        "lr": LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "eval_interval": EVAL_INTERVAL,
        "conditioning": "text",
    },
    "t2i_uncond": {
        "cfg_path": f"{MODEL_CONFIGS_DIR}/rf_dit_t2i.yaml",
        "ckpt_path": f"{MODEL_CKPT_DIR}/t2i.pt",
        "timestep": TIMESTEP,
        "layer_idx": LAYER_IDX,
        "lr": LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "eval_interval": EVAL_INTERVAL,
        "conditioning": "unconditional",
    }
}