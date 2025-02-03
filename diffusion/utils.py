import random
import time
from typing import Any, Dict, Union

import numpy as np
import torch


class TimeMeasurement:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.ema = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        if self.ema is None:
            self.ema = elapsed_time
        else:
            self.ema = self.alpha * elapsed_time + (1 - self.alpha) * self.ema

    def reset(self):
        self.ema = None


class NullObject:
    def __getattr__(self, name) -> "NullObject":
        return NullObject()

    def __call__(self, *args: Any, **kwds: Any) -> "NullObject":
        return NullObject()

    def __enter__(self) -> "NullObject":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass


def set_seed(seed=42, cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def dict_to(d: Dict[str, Union[torch.Tensor, Any]], **to_kwargs) -> Dict[str, Union[torch.Tensor, Any]]:
    return {k: (v.to(**to_kwargs) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}
