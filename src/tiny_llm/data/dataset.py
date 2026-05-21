"""PyTorch Dataset for memory-mapped token data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapDataset(Dataset):
    """Dataset that reads token IDs from a memory-mapped binary file.

    Each sample is a (input, target) pair of context_length tokens,
    where target is input shifted by one position.

    Args:
        data_path: Path to the .bin memmap file (uint16 token IDs).
        context_length: Number of tokens per training sample.
    """

    def __init__(self, data_path: str | Path, context_length: int = 1024) -> None:
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Tokenized data file not found: '{data_path}'. "
                "Run the data pipeline first, e.g. "
                "`uv run python main.py --stage data --num-samples 100000 --seed 42`."
            )

        self.data = np.memmap(str(data_path), dtype=np.uint16, mode="r")
        self.context_length = context_length

    def __len__(self) -> int:
        # Each sample needs context_length + 1 tokens (input + one shifted target)
        # Non-overlapping windows
        return (len(self.data) - 1) // self.context_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input_ids, target_ids) each of shape (context_length,)."""
        start = idx * self.context_length
        end = start + self.context_length + 1
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
