"""
Reproducibility & Seed Management

Call set_seed() once at the start of every script.
"""

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Global seed value.
        deterministic: If True, enables CUDA deterministic mode
                       (may reduce performance slightly).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # PyTorch >= 1.8
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            try:
                # warn_only: some kernels (e.g. grid_sample backward) have no
                # deterministic CUDA impl; warn rather than crash.
                torch.use_deterministic_algorithms(True, warn_only=True)
            except AttributeError:
                pass
    except ImportError:
        pass
