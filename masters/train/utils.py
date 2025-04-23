import os
import random
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch


class GeneClassificationTask(TypedDict):
    task: str
    dataset_dir: str
    label_file: str
    id_column: str
    target_column: str


def training_setup(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)

    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(deterministic)


def get_next_version(root: str | Path, prefix: str, sep: str = "_") -> str:
    root = Path(root)

    if not root.exists():
        root.mkdir(parents=True)
        return f"{prefix}{sep}0"

    if not root.is_dir():
        raise ValueError(f"{root} is not a directory")

    versions = []

    for item in root.iterdir():
        if item.is_dir() and item.name.startswith(prefix):
            version = item.name.split(sep)[-1]
            if version.isdigit():
                versions.append(int(version))

    if not versions:
        return f"{prefix}{sep}0"

    return f"{prefix}{sep}{max(versions) + 1}"
