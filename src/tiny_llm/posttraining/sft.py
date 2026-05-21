"""Supervised fine-tuning orchestration.

Loads tokenised SFT data (building the cache if needed), constructs the
:class:`SFTTrainer` and runs the training loop.  The prompt format is
Stanford Alpaca (plain text ``### Instruction:`` / ``### Response:`` markers,
no new special tokens).  Loss is computed only on response tokens via
``ignore_index=-100``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from tiny_llm.model.transformer import GPTModel

logger = logging.getLogger(__name__)


def run_sft(
    model: "GPTModel",
    config: dict[str, Any],
    device: "torch.device",
    model_config: dict[str, Any] | None = None,
    seed: int = 42,
) -> None:
    """Fine-tune a pretrained model on instruction-following data.

    Args:
        model: Pretrained GPT model (weights already loaded, device-agnostic —
            the trainer will move it to ``device``).
        config: Post-training config dict (from ``PostTrainingConfig``).
        device: Device to train on.
        model_config: Model architecture dict — persisted into checkpoints so
            evaluation and inference can reconstruct the model.
        seed: Seed used for dataset shuffling.
    """
    from tiny_llm.posttraining.sft_dataset import SFTDataset, prepare_sft_dataset
    from tiny_llm.posttraining.sft_trainer import SFTTrainer

    datasets = config.get("datasets") or ["alpaca", "oasst1", "dolly"]
    dataset_weights = config.get("dataset_weights") or {}
    logger.info(
        "[sft] Preparing datasets: %s (weights=%s)",
        datasets,
        dataset_weights or "uniform",
    )

    train_examples, val_examples = prepare_sft_dataset(
        datasets=datasets,
        dataset_weights=dataset_weights,
        filter_code_and_math=bool(config.get("filter_code_and_math", True)),
        cache_dir=config.get("dataset_dir", "data/sft"),
        max_seq_len=int(config.get("max_seq_len", 1024)),
        val_fraction=float(config.get("val_fraction", 0.02)),
        max_samples=config.get("max_samples"),
        seed=seed,
    )

    train_ds = SFTDataset(train_examples)
    val_ds = SFTDataset(val_examples) if val_examples else None

    # Inject chat_template into model_config so every checkpoint this run
    # emits carries the tag.  Inference reads it back to auto-apply the
    # matching Alpaca template — no --chat-template flag needed.
    sft_model_config = dict(model_config or {})
    sft_model_config["chat_template"] = "alpaca"

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
        device=device,
        model_config=sft_model_config,
    )
    trainer.train()


__all__ = ["run_sft"]
