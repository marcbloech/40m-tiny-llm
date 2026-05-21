"""Automatic device detection and resolution."""

from __future__ import annotations

import torch


def resolve_device(device: str | None = None) -> torch.device:
    """Resolve a device string to a torch.device.

    Priority when device is None or 'auto': CUDA > MPS > CPU.

    Args:
        device: Device string ('cpu', 'cuda', 'mps', 'auto', or None).

    Returns:
        Resolved torch.device.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)
