"""Save and load model checkpoints with training metadata."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------


def _atomic_save(checkpoint: dict, path: Path) -> None:
    tmp_path = path.with_suffix(".pt.tmp")
    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)  # atomic on POSIX
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Save / load / prune
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer | None,
    step: int,
    config: dict,
    path: str | Path,
    val_loss: float | None = None,
    tokens_seen: int = 0,
    model_config: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "step": step,
        "config": config,
        "model_config": model_config,
        "val_loss": val_loss,
        "tokens_seen": tokens_seen,
    }

    _atomic_save(checkpoint, path)
    logger.info("Checkpoint saved: %s (step=%d, val_loss=%s)", path, step, val_loss)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None,
    optimizer: optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict:
    """Load a training checkpoint and restore model/optimiser state.

    Args:
        path: Path to .pt checkpoint file.
        model: Model to load weights into (must have matching architecture).
            Pass None to peek at metadata without loading weights.
        optimizer: Optimiser to restore state into (pass None to skip, e.g. for inference).
        device: Device to map tensors to (e.g. "cpu" or "cuda").

    Returns:
        Metadata dict with keys: step, config, model_config, val_loss, tokens_seen.
    """
    path = Path(path)
    # map_location ensures tensors are loaded onto the right device
    # (e.g. if checkpoint was saved on GPU but we're loading on CPU)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Model weights loaded from %s", path)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        opt_state = checkpoint["optimizer_state_dict"]
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)
            logger.info("Optimizer state restored")

    return {
        "step": checkpoint.get("step", 0),
        "config": checkpoint.get("config", {}),
        "model_config": checkpoint.get("model_config"),
        "val_loss": checkpoint.get("val_loss"),
        "tokens_seen": checkpoint.get("tokens_seen", 0),
    }


def prune_checkpoints(checkpoint_dir: str | Path, max_keep: int = 3) -> None:
    """Keep only the most recent periodic checkpoints, deleting older ones.

    Only prunes files matching ``step_*.pt``.  The ``best.pt`` file is
    never touched — it's the best model for evaluation/post-training.

    This prevents disk usage from growing unboundedly during long training
    runs.  At our model size, each full checkpoint is ~500 MB, so keeping
    3 means ~1.5 GB of disk.

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        max_keep: Maximum number of periodic checkpoints to retain.
    """
    ckpt_dir = Path(checkpoint_dir)
    # Sort by modification time so we keep the most recent ones
    periodic = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)

    if len(periodic) <= max_keep:
        return

    to_remove = periodic[: len(periodic) - max_keep]
    for p in to_remove:
        p.unlink()
        logger.info("Pruned old checkpoint: %s", p.name)


# ---------------------------------------------------------------------------
# Top-K best checkpoints
# ---------------------------------------------------------------------------


def _update_best_link(checkpoint_dir: Path, target_path: Path) -> None:
    """Make best.pt point to the given target file (symlink with copy fallback)."""
    link = checkpoint_dir / "best.pt"
    try:
        tmp_link = link.with_suffix(".pt.lnk")
        tmp_link.unlink(missing_ok=True)
        tmp_link.symlink_to(target_path.name)  # relative symlink
        os.replace(tmp_link, link)
    except OSError:
        import shutil

        shutil.copy2(target_path, link)


def save_best_checkpoint(
    model: nn.Module,
    step: int,
    config: dict,
    checkpoint_dir: str | Path,
    val_loss: float,
    tokens_seen: int = 0,
    model_config: dict | None = None,
    max_best: int = 3,
) -> Path | None:
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Discover existing best checkpoints and read their val_loss metadata
    existing: list[tuple[float, Path]] = []
    for p in ckpt_dir.glob("best_step_*.pt"):
        try:
            meta = torch.load(p, map_location="cpu", weights_only=False)
            loss = meta.get("val_loss")
            if loss is not None:
                existing.append((loss, p))
        except Exception:
            logger.warning("Could not read best checkpoint %s, skipping", p.name)

    existing.sort(key=lambda t: t[0])  # ascending = best first

    # Check if new val_loss qualifies for top-K
    if len(existing) >= max_best and val_loss >= existing[-1][0]:
        return None  # not in top-K

    # Save new best checkpoint (model weights only, no optimizer)
    new_path = ckpt_dir / f"best_step_{step:06d}.pt"
    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None,
        "step": step,
        "config": config,
        "model_config": model_config,
        "val_loss": val_loss,
        "tokens_seen": tokens_seen,
    }
    _atomic_save(checkpoint, new_path)

    # Add to list, re-sort, prune if over max_best
    existing.append((val_loss, new_path))
    existing.sort(key=lambda t: t[0])

    while len(existing) > max_best:
        _loss, worst_path = existing.pop()
        worst_path.unlink(missing_ok=True)
        logger.info("Pruned best checkpoint: %s (val_loss=%.4f)", worst_path.name, _loss)

    _update_best_link(ckpt_dir, existing[0][1])
    return new_path
