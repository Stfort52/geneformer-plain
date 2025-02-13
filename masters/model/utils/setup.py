import os
import random

import numpy as np
import torch


def training_setup(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)

    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(deterministic)
