"""Deterministic seeding for reproducibility."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: The random seed value.
        deterministic: If True, disables cuDNN algorithm autotuning for strict
            reproducibility. Has a significant throughput cost on V100/A100 —
            leave False for pre-training runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
