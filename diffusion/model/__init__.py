from typing import Optional

import torch.nn as nn
from torch.utils.data import DataLoader


class Module(nn.Module):
    def validate(
        self,
        dataloader_val: DataLoader,
        global_rank: int,
        global_samples: int,
        max_steps: Optional[int],
        device,
        dtype,
        wandb,
        monitor,
    ) -> None:
        raise NotImplementedError()
