"""Metrics logger tailored to supervised fine-tuning.

Writes per-step metrics to ``metrics.csv`` + ``metrics.jsonl`` and renders:

- a live 4-panel PNG dashboard under ``plots/training_quicklook.png``
- a final detailed 6-panel PNG under ``plots/training_detailed.png``

Design mirrors :class:`tiny_llm.utils.logging.MetricsLogger` so the CSV is
machine-parsable the same way, but the column set and plotting panels are
SFT-specific.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tiny_llm.utils.logging import get_logger

logger = logging.getLogger(__name__)


class SFTMetricsLogger:
    """Structured metrics logger for SFT runs."""

    _BASE_CSV_COLUMNS = [
        "step",
        "epoch",
        "timestamp",
        "train/loss",
        "train/lr",
        "train/grad_norm",
        "train/tokens_per_sec",
        "val/loss",
        "val/perplexity",
        "pretrain/val_loss",
        "pretrain/val_perplexity",
    ]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._logger = get_logger("tiny_llm.posttraining.metrics")

        self._csv_file = None
        self._csv_writer = None
        self._csv_columns = list(self._BASE_CSV_COLUMNS)
        self._jsonl_file = None
        self._current_row: dict[str, Any] = {}
        self._current_step: int | None = None
        self._current_epoch: int | None = None
        self._history_rows: list[dict[str, Any]] = []

        self._checkpoint_dir: Path | None = None
        self._plots_dir: Path | None = None
        self._quick_plot_path: Path | None = None
        self._detailed_plot_path: Path | None = None
        self._quick_plot_interval = 50
        self._last_quick_plot_step = -1
        self._plotting_enabled = False

        if config and config.get("checkpoint_dir"):
            for name in config.get("datasets") or []:
                metric_name = str(name).replace("/", "_")
                self._csv_columns.extend(
                    [
                        f"val/{metric_name}_loss",
                        f"val/{metric_name}_perplexity",
                    ]
                )

            self._checkpoint_dir = Path(config["checkpoint_dir"])
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

            csv_path = self._checkpoint_dir / "metrics.csv"
            file_exists = csv_path.exists() and csv_path.stat().st_size > 0
            self._csv_file = open(csv_path, "a", newline="", encoding="utf-8")  # noqa: SIM115
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=self._csv_columns,
                extrasaction="ignore",
            )
            if not file_exists:
                self._csv_writer.writeheader()
                self._csv_file.flush()
            self._logger.info("SFT CSV metrics → %s", csv_path)

            jsonl_path = self._checkpoint_dir / "metrics.jsonl"
            self._jsonl_file = open(jsonl_path, "a", encoding="utf-8")  # noqa: SIM115
            self._logger.info("SFT JSONL metrics → %s", jsonl_path)

            self._quick_plot_interval = max(
                1, int(config.get("quick_plot_interval", 50))
            )
            self._plots_dir = self._checkpoint_dir / "plots"
            self._plots_dir.mkdir(parents=True, exist_ok=True)
            self._quick_plot_path = self._plots_dir / "training_quicklook.png"
            self._detailed_plot_path = self._plots_dir / "training_detailed.png"
            self._plotting_enabled = True

    # ------------------------------------------------------------------
    # Logging API
    # ------------------------------------------------------------------

    def log(
        self,
        metrics: dict[str, Any],
        step: int,
        epoch: int | None = None,
    ) -> None:
        """Log metrics at a given step (stdout + CSV + JSONL buffering)."""
        parts = " | ".join(f"{k}={v}" for k, v in metrics.items())
        self._logger.info("step %d | %s", step, parts)

        if self._csv_writer is None:
            return
        if self._current_step is not None and step != self._current_step:
            self._flush_row()
        if self._current_step != step:
            self._current_step = step
            self._current_epoch = epoch
        elif epoch is not None and self._current_epoch is None:
            self._current_epoch = epoch
        self._current_row.update(metrics)

    def maybe_save_quick_plot(self, step: int) -> Path | None:
        if not self._plotting_enabled:
            return None
        if step <= 0 or step % self._quick_plot_interval != 0:
            return None
        if step == self._last_quick_plot_step:
            return None
        path = self._save_plot(step)
        if path is not None:
            self._last_quick_plot_step = step
        return path

    def save_detailed_plot(self) -> Path | None:
        if not self._plotting_enabled:
            return None
        return self._save_detailed_plot()

    def finish(self) -> Path | None:
        """Flush final row, write final plots, close files."""
        self._flush_row()
        final_plot = None
        if self._plotting_enabled:
            # Use whichever step is last in history for the final plot title.
            last_step = (
                self._history_rows[-1].get("step", 0)
                if self._history_rows
                else 0
            )
            final_plot = self._save_plot(last_step)
            if final_plot is not None:
                self._logger.info("Final SFT plot → %s", final_plot)
            detailed_plot = self.save_detailed_plot()
            if detailed_plot is not None:
                self._logger.info("Detailed SFT plot → %s", detailed_plot)

        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        if self._jsonl_file is not None:
            self._jsonl_file.close()
            self._jsonl_file = None
        self._logger.info("SFT metrics logging finished.")
        return final_plot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        return value

    def _flush_row(self) -> None:
        if not self._current_row or self._csv_writer is None:
            return
        row = {
            "step": self._current_step,
            "epoch": self._current_epoch,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        row.update(self._current_row)

        self._csv_writer.writerow(row)
        self._csv_file.flush()

        if self._jsonl_file is not None:
            safe_row = {k: self._json_safe(v) for k, v in row.items()}
            self._jsonl_file.write(json.dumps(safe_row) + "\n")
            self._jsonl_file.flush()

        self._history_rows.append(row)
        self._current_row = {}
        self._current_step = None
        self._current_epoch = None

    def _rows_for_plot(self) -> list[dict[str, Any]]:
        rows = list(self._history_rows)
        if self._current_row and self._current_step is not None:
            row = {"step": self._current_step, "epoch": self._current_epoch}
            row.update(self._current_row)
            rows.append(row)
        return rows

    def _series(
        self,
        rows: list[dict[str, Any]],
        key: str,
    ) -> tuple[list[int], list[float]]:
        xs: list[int] = []
        ys: list[float] = []
        for row in rows:
            step = row.get("step")
            value = row.get(key)
            if not isinstance(step, int):
                continue
            if not self._is_number(value):
                continue
            value_f = float(value)
            if not math.isfinite(value_f):
                continue
            xs.append(step)
            ys.append(value_f)
        return xs, ys

    def _save_plot(self, step: int) -> Path | None:
        rows = self._rows_for_plot()
        if not rows or self._quick_plot_path is None:
            return None

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            self._logger.warning("SFT plot skipped (matplotlib unavailable): %s", exc)
            self._plotting_enabled = False
            return None

        train_steps, train_loss = self._series(rows, "train/loss")
        val_steps, val_loss = self._series(rows, "val/loss")
        pre_steps, pre_ppl = self._series(rows, "pretrain/val_perplexity")
        lr_steps, lrs = self._series(rows, "train/lr")
        gn_steps, grad_norm = self._series(rows, "train/grad_norm")

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle(
            f"SFT Training Snapshot — step {step}", fontsize=12, fontweight="bold"
        )

        # (0,0) Train vs Val loss — overfitting
        ax = axes[0, 0]
        if train_steps:
            ax.plot(
                train_steps, train_loss,
                color="#1f77b4", linewidth=1.0, label="Train",
            )
        if val_steps:
            ax.plot(
                val_steps, val_loss,
                color="#ff7f0e", linewidth=1.3, marker="o", markersize=3,
                label="Validation",
            )
        ax.set_title("SFT Loss (train vs. val)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        if train_steps or val_steps:
            ax.legend()

        # (0,1) Pretrain val perplexity — catastrophic forgetting
        ax = axes[0, 1]
        if pre_steps:
            ax.plot(
                pre_steps, pre_ppl,
                color="#d62728", linewidth=1.3, marker="s", markersize=3,
            )
        ax.set_title("Pretrain-val Perplexity\n(rising = catastrophic forgetting)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Perplexity")
        ax.grid(True, alpha=0.3)

        # (1,0) LR schedule
        ax = axes[1, 0]
        if lr_steps:
            ax.plot(lr_steps, lrs, color="#2ca02c", linewidth=1.4)
        ax.set_title("Learning Rate Schedule")
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.grid(True, alpha=0.3)

        # (1,1) Grad norm
        ax = axes[1, 1]
        if gn_steps:
            ax.plot(gn_steps, grad_norm, color="#9467bd", linewidth=1.0)
        ax.set_title("Gradient Norm")
        ax.set_xlabel("Step")
        ax.set_ylabel("||grad||")
        ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        fig.savefig(self._quick_plot_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return self._quick_plot_path

    def _save_detailed_plot(self) -> Path | None:
        rows = self._rows_for_plot()
        if not rows or self._detailed_plot_path is None:
            return None

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            self._logger.warning("Detailed SFT plot skipped (matplotlib unavailable): %s", exc)
            self._plotting_enabled = False
            return None

        run_name = self._checkpoint_dir.name if self._checkpoint_dir else "sft_run"
        fig, axes = plt.subplots(2, 3, figsize=(17, 9))
        fig.suptitle(f"SFT Detailed Overview — {run_name}", fontsize=16, fontweight="bold")

        train_steps, train_loss = self._series(rows, "train/loss")
        val_steps, val_loss = self._series(rows, "val/loss")
        val_ppl_steps, val_ppl = self._series(rows, "val/perplexity")
        pre_steps, pre_ppl = self._series(rows, "pretrain/val_perplexity")
        lr_steps, lrs = self._series(rows, "train/lr")
        gn_steps, grad_norm = self._series(rows, "train/grad_norm")
        speed_steps, speed_vals = self._series(rows, "train/tokens_per_sec")

        ax = axes[0, 0]
        if train_steps:
            ax.plot(train_steps, train_loss, color="#1f77b4", linewidth=1.0, label="Train")
        if val_steps:
            ax.plot(
                val_steps, val_loss,
                color="#ff7f0e", linewidth=1.2, marker="o", markersize=3,
                label="Validation",
            )
        if train_steps or val_steps:
            ax.set_yscale("log")
            ax.legend()
        ax.set_title("SFT Loss (train vs. val)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        if val_ppl_steps:
            ax.plot(val_ppl_steps, val_ppl, color="#2ca02c", linewidth=1.2, marker="o", markersize=3)
        ax.set_title("SFT Validation Perplexity")
        ax.set_xlabel("Step")
        ax.set_ylabel("Perplexity")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
        if pre_steps:
            ax.plot(pre_steps, pre_ppl, color="#d62728", linewidth=1.2, marker="s", markersize=3)
        ax.set_title("Pretrain-val Perplexity")
        ax.set_xlabel("Step")
        ax.set_ylabel("Perplexity")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        if lr_steps:
            ax.plot(lr_steps, lrs, color="#9467bd", linewidth=1.4)
        ax.set_title("Learning Rate Schedule")
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        if gn_steps:
            ax.plot(gn_steps, grad_norm, color="#8c564b", linewidth=1.0)
        ax.set_title("Gradient Norm")
        ax.set_xlabel("Step")
        ax.set_ylabel("||grad||")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        if speed_steps:
            ax.plot(speed_steps, speed_vals, color="#17becf", linewidth=1.0)
        ax.set_title("Training Speed")
        ax.set_xlabel("Step")
        ax.set_ylabel("Tokens/sec")
        ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        fig.savefig(self._detailed_plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return self._detailed_plot_path


__all__ = ["SFTMetricsLogger"]
