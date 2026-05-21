"""Direct Preference Optimization orchestration.

Loads tokenised DPO preference data (building the cache if needed), creates a
frozen reference model via ``copy.deepcopy``, constructs the
:class:`DPOTrainer` and runs the training loop.  Follows the EX08 DPO notebook
pattern adapted to Tiny LLM's repo-native pipeline architecture.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from tiny_llm.model.transformer import GPTModel

logger = logging.getLogger(__name__)


def run_dpo(
    model: "GPTModel",
    config: dict[str, Any],
    device: "torch.device",
    model_config: dict[str, Any] | None = None,
    seed: int = 42,
) -> None:
    """Run DPO training on preference data.

    Args:
        model: SFT-tuned GPT model (weights already loaded, on ``device``).
        config: Post-training config dict (from ``PostTrainingConfig``).
        device: Device to train on.
        model_config: Model architecture dict -- persisted into checkpoints.
        seed: Seed used for dataset shuffling.
    """
    from tiny_llm.posttraining.dpo_dataset import DPODataset, prepare_dpo_dataset
    from tiny_llm.posttraining.dpo_trainer import DPOTrainer

    datasets = config.get("dpo_datasets") or ["orca_dpo"]
    logger.info("[dpo] Preparing datasets: %s", datasets)

    train_examples, val_examples = prepare_dpo_dataset(
        datasets=datasets,
        cache_dir=config.get("dpo_dataset_dir", "data/dpo"),
        max_seq_len=int(config.get("max_seq_len", 1024)),
        val_fraction=float(config.get("val_fraction", 0.05)),
        max_samples=config.get("max_samples"),
        seed=seed,
    )

    train_ds = DPODataset(train_examples)
    val_ds = DPODataset(val_examples) if val_examples else None

    logger.info("[dpo] Creating frozen reference model (deepcopy)...")
    ref_model = copy.deepcopy(model)

    dpo_model_config = dict(model_config or {})
    dpo_model_config["chat_template"] = "alpaca"

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
        device=device,
        model_config=dpo_model_config,
    )
    trainer.train()


__all__ = ["run_dpo"]
