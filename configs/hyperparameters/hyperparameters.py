from configs.path_configs.path_configs import MODEL_CONFIGS_DIR, MODEL_CKPT_DIR

EPOCHS, BATCH_SIZE, EVAL_INTERVAL, LR = 1, 32, 1000, 0.001
TIMESTEP, LAYER_NUM, LAYER_START = 0.75, 14, 14
DEVICE = "cuda:0"

models = {
    "unclip": {
        "cfg_path": f"{MODEL_CONFIGS_DIR}/rf_dit_unclip.yaml",
        "ckpt_path": f"{MODEL_CKPT_DIR}/unclip.pt",
        "timestep": TIMESTEP,
        "layer_num": LAYER_NUM,
        "layer_start": LAYER_START,
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
        "layer_num": LAYER_NUM,
        "layer_start": LAYER_START,
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
        "layer_num": LAYER_NUM,
        "layer_start": LAYER_START,
        "lr": LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "eval_interval": EVAL_INTERVAL,
        "conditioning": "unconditional",
    }
}