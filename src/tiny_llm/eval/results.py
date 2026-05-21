"""Persist evaluation results as JSON for cross-run comparison."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("eval_outputs")


def save_eval_results(
    checkpoint_path: str | Path,
    meta: dict,
    model_config: dict,
    n_params: int,
    perplexity: dict[str, float],
    benchmarks: dict[str, dict],
    output_dir: str | Path = OUTPUT_DIR,
) -> Path:
    """Save evaluation results to a timestamped JSON file.

    Args:
        checkpoint_path: Path to the evaluated checkpoint.
        meta: Checkpoint metadata (step, val_loss, tokens_seen).
        model_config: Model architecture config dict.
        n_params: Total parameter count.
        perplexity: Dict of dataset_name -> PPL score.
        benchmarks: Dict of benchmark_name -> result dict.
        output_dir: Directory to save results into.

    Returns:
        Path to the saved JSON file.
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive model name from checkpoint path
    # e.g. "checkpoints/small/best.pt" -> "small_best"
    try:
        # Try to get relative path from a checkpoints/ parent
        for parent in checkpoint_path.parents:
            if parent.name == "checkpoints":
                rel = checkpoint_path.relative_to(parent)
                model_name = str(rel.with_suffix("")).replace("/", "_")
                break
        else:
            model_name = checkpoint_path.stem
    except (ValueError, StopIteration):
        model_name = checkpoint_path.stem

    payload = {
        "model_name": model_name,
        "checkpoint": str(checkpoint_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": meta.get("step", 0),
        "val_loss": meta.get("val_loss"),
        "tokens_seen": meta.get("tokens_seen", 0),
        "n_params": n_params,
        "model_config": model_config,
        "perplexity": perplexity,
        "benchmarks": benchmarks,
    }

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    step = meta.get("step", 0)
    filename = f"{model_name}_step{step:06d}_{date_str}.json"
    output_file = output_dir / filename

    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info("Eval results saved to %s", output_file)
    return output_file
