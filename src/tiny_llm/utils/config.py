"""Pydantic configuration models with YAML / TOML loading and CLI overrides.

Each section of a TOML config file (e.g. [model], [training]) maps to one of
the Pydantic models below.  Pydantic validates types and provides defaults, so
any field you omit in TOML just gets the default value shown here.

Example TOML config:
    [model]
    d_model = 384
    n_layers = 10

    [training]
    learning_rate = 6e-4
    max_steps = 5000

CLI overrides (dot-notation) are merged on top:
    --overrides training.learning_rate=1e-3 model.n_layers=8
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Transformer architecture hyperparameters.

    These define the model's size and structure.  The param count is roughly:
        params ≈ vocab_size × d_model  (embedding, shared if weight_tying)
               + n_layers × (~4 × d_model² + 2 × d_model × d_ff)
    """

    vocab_size: int = 50257  # GPT-2 tokeniser vocabulary size
    context_length: int = 1024  # max sequence length (tokens)
    n_layers: int = 12  # number of transformer blocks (depth)
    n_heads: int = 12  # number of attention heads
    d_model: int = 384  # embedding / hidden dimension (width)
    d_ff: int = 1536  # feed-forward hidden dim (typically 4× d_model)
    dropout: float = 0.1  # dropout rate (0.0 = off, recommended for pre-training)
    activation: str = "gelu"  # "gelu" or "swiglu"
    norm_position: str = "pre"  # "pre", "post", or "pre_post"
    weight_tying: bool = True  # share embedding and output projection weights
    bias: bool = False  # use bias in linear layers (modern default: False)
    qk_norm: bool = False  # apply RMSNorm to Q/K in attention (stabilises deep models)
    positional_encoding: str = "rope"  # "rope" or "learned"
    rope_theta: float = 10000.0  # RoPE base frequency


class DataSourceConfig(BaseModel):
    """Configuration for a single training data source."""

    name: str
    hf_dataset_id: str
    hf_subset: Optional[str] = None
    hf_split: str = "train"
    text_column: str = "text"
    ratio: float  # fixed proportion (must sum to ~1.0 across sources)


class DataConfig(BaseModel):
    """Paths to tokenised training data and optional multi-source config."""

    train_path: str = "data/tokenized/train.bin"
    val_path: Optional[str] = "data/tokenized/val.bin"
    total_tokens: int = 800_000_000  # single knob — scales all sources proportionally
    sources: Optional[list[DataSourceConfig]] = None  # None = legacy single-source
    decontaminate: bool = True


class TrainingConfig(BaseModel):
    """Pre-training hyperparameters."""

    # ── Batch & accumulation ──────────────────────────────────────────────
    batch_size: int = 64  # sequences per micro-batch
    eval_batch_size: Optional[int] = None  # eval batch size (default: batch_size × 4)
    grad_accum_steps: int = 4  # micro-batches per optimiser step (1 = off)
    # effective_batch = batch_size × grad_accum_steps (e.g. 64 × 4 = 256 seqs)

    # ── Optimiser ─────────────────────────────────────────────────────────
    learning_rate: float = 3e-4  # peak LR (after warmup)
    weight_decay: float = 0.1  # AdamW weight decay (applied to 2D+ params only)
    optimizer: str = "adamw"  # "adamw", "lion", or "muon"
    betas: list[float] = Field(default=[0.9, 0.95])

    # ── LR schedule (toggleable) ──────────────────────────────────────────
    use_scheduler: bool = True  # False = constant LR (no warmup, no cosine decay)
    lr_schedule: str = "cosine"  # "cosine" or "wsd"
    warmup_steps: int = 500  # linear warmup duration (only used if use_scheduler=True)
    min_lr_ratio: float = 0.1  # min LR = learning_rate × min_lr_ratio
    wsd_decay_fraction: float = 0.2  # fraction of total steps for WSD decay phase

    # ── Gradient clipping (toggleable) ────────────────────────────────────
    gradient_clip_norm: float = 1.0  # max global gradient L2 norm (0.0 = disabled)

    # ── torch.compile (toggleable) ─────────────────────────────────────
    use_compile: bool = False  # torch.compile the model for kernel fusion speedups

    # ── Training duration & intervals ─────────────────────────────────────
    max_steps: int = 20000  # total optimiser steps
    eval_interval: int = 500  # steps between validation runs
    checkpoint_interval: int = 2000  # steps between periodic checkpoints
    sample_interval: int = 1000    # steps between text sample generation
    log_interval: int = 10         # steps between metric logging
    quick_plot_interval: int = 100  # steps between quick training plot updates

    # ── Checkpointing ─────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints/final"
    max_checkpoints: int = 3  # only keep N most recent periodic checkpoints

    # ── DataLoader ────────────────────────────────────────────────────────
    num_workers: int = 4  # parallel data-loading processes
    pin_memory: bool = True  # faster CPU→GPU transfer (CUDA only)



class PostTrainingConfig(BaseModel):
    """Post-training (SFT / DPO) hyperparameters.

    For SFT, the loss is computed only on response tokens (prompt tokens carry
    ``target=-100`` and are ignored by cross-entropy).  Set ``method="sft"``
    to enable.  Monitoring is tuned for two failure modes:
      - **Overfitting**: gap between ``train/loss`` and ``val/loss``.
      - **Catastrophic forgetting**: pretrain-val perplexity drifts upward
        during SFT.  The probe samples ``pretrain_val_batches`` chunks from
        ``pretrain_val_path`` every ``pretrain_eval_interval`` steps.
    """

    # ── Method ────────────────────────────────────────────────────────────
    method: str = "sft"

    # ── Datasets (registry keys from sft_dataset.py; defaults mix Alpaca,
    # English OASST1, and English Dolly) ─────────────────────────────────
    datasets: list[str] = Field(default_factory=lambda: ["alpaca", "oasst1", "dolly"])
    dataset_weights: dict[str, float] = Field(default_factory=dict)
    dataset_dir: str = "data/sft"     # cache dir for tokenised SFT data
    val_fraction: float = 0.02         # held-out fraction for SFT val split
    max_samples: Optional[int] = None  # cap per dataset (None = all)
    max_seq_len: int = 1024            # truncation budget (prompt + response)
    filter_code_and_math: bool = True   # drop code/markup/math SFT rows by default

    # ── Catastrophic-forgetting probe ────────────────────────────────────
    pretrain_val_path: Optional[str] = "data/tokenized/val.bin"
    pretrain_val_batches: int = 40
    pretrain_eval_interval: int = 200  # steps between probes

    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = 16
    eval_batch_size: Optional[int] = None
    grad_accum_steps: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    betas: list[float] = Field(default=[0.9, 0.95])
    epochs: int = 3
    warmup_ratio: float = 0.1          # fraction of total optim steps for warmup
    gradient_clip_norm: float = 1.0
    # ── Intervals ────────────────────────────────────────────────────────
    log_interval: int = 10
    eval_interval: int = 100
    checkpoint_interval: int = 500
    quick_plot_interval: int = 50
    sample_interval: int = 0
    sample_max_tokens: int = 80
    sample_temperature: float = 0.0
    sample_top_k: Optional[int] = 40
    sample_repetition_penalty: float = 1.1
    sample_prompts: list[Any] = Field(default_factory=list)

    # ── Checkpointing ────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints/sft"
    max_checkpoints: int = 3
    max_best_checkpoints: int = 3

    # ── DataLoader ───────────────────────────────────────────────────────
    num_workers: int = 2
    pin_memory: bool = True

    # ── DPO-specific (ignored when method="sft") ──────────────────────────
    dpo_beta: float = 0.1
    dpo_datasets: list[str] = Field(default_factory=lambda: ["orca_dpo"])
    dpo_dataset_dir: str = "data/dpo"
    dpo_label_smoothing: float = 0.0
    dpo_loss_type: str = "sigmoid"
    dpo_reference_free: bool = False
    dpo_length_normalized: bool = False

    # ── Legacy (kept for backward-compat with older configs that set it) ─
    data_path: Optional[str] = None


class EvalConfig(BaseModel):
    """Evaluation settings."""

    test_data_path: str = "data/tokenized/test.bin"


class LoggingConfig(BaseModel):
    """Logging configuration — root level and per-component overrides.

    Example TOML::

        [logging]
        root_level = "INFO"

        [logging.levels]
        "tiny_llm.data" = "DEBUG"
        "tiny_llm.training" = "WARNING"
    """

    root_level: str = "INFO"
    levels: dict[str, str] = Field(default_factory=dict)


class TinyLLMConfig(BaseModel):
    """Top-level configuration combining all sections."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    posttraining: Optional[PostTrainingConfig] = None
    eval: EvalConfig = Field(default_factory=EvalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_raw(path: Path) -> dict[str, Any]:
    """Load a config file (YAML or TOML) and return a raw dict."""
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text) or {}

    if suffix == ".toml":
        import tomllib

        return tomllib.loads(text)

    raise ValueError(
        f"Unsupported config format: {suffix} (expected .yaml, .yml, or .toml)"
    )


def load_config(path: str | Path) -> TinyLLMConfig:
    """Load a YAML or TOML configuration file and return a validated model.

    Args:
        path: Path to the config file (``.yaml``, ``.yml``, or ``.toml``).

    Returns:
        A fully validated :class:`TinyLLMConfig` instance.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = _load_raw(path)
    logger.debug("Loaded raw config from %s: %s", path, raw)
    return TinyLLMConfig.model_validate(raw)


def merge_cli_overrides(config: TinyLLMConfig, overrides: list[str]) -> TinyLLMConfig:
    """Merge CLI ``key=value`` overrides into a Pydantic config.

    Supports dot-notation for nested keys, e.g. ``model.n_layers=8``.

    Args:
        config: The base configuration.
        overrides: List of ``"key=value"`` strings.

    Returns:
        A new :class:`TinyLLMConfig` with overrides applied.
    """
    if not overrides:
        return config

    raw = config.model_dump()

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override (expected key=value): {item!r}")

        key, value = item.split("=", 1)
        keys = key.split(".")

        # Navigate to parent dict
        target = raw
        for k in keys[:-1]:
            if k not in target or not isinstance(target[k], dict):
                raise KeyError(f"Config key not found: {key!r}")
            target = target[k]

        # Coerce value to the type of the existing field
        existing = target.get(keys[-1])
        if existing is not None:
            try:
                if isinstance(existing, bool):
                    value = value.lower() in ("true", "1", "yes")
                elif isinstance(existing, int):
                    value = int(value)
                elif isinstance(existing, float):
                    value = float(value)
                # str stays as-is
            except (ValueError, TypeError):
                pass  # keep as string

        target[keys[-1]] = value
        logger.debug("Override applied: %s = %s", key, value)

    return TinyLLMConfig.model_validate(raw)
