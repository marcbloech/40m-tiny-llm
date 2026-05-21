"""Logging utilities: structured Python logging and training metrics.

Two separate logging systems:
  1. Python's logging module — for progress messages, warnings, errors.
     Use: logger = logging.getLogger(__name__); logger.info("...")
  2. MetricsLogger — for structured training metrics (loss, LR, perplexity).
     Outputs one line per step in a parseable format.

Logging is configured per-component via the [logging] section in TOML configs,
so you can e.g. set tiny_llm.data=DEBUG while keeping tiny_llm.model=WARNING.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Default format — includes component name for easy filtering
# ---------------------------------------------------------------------------
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False
_active_log_format = _DEFAULT_FORMAT
_active_date_format = _DEFAULT_DATE_FORMAT


def setup_logging(
    root_level: str = "INFO",
    component_levels: dict[str, str] | None = None,
    log_format: str | None = None,
    date_format: str | None = None,
) -> None:
    """Configure the Python logging framework for Tiny LLM.

    Sets up the ``tiny_llm`` root logger with a console handler and applies
    per-component log levels so different parts of the pipeline can be
    independently tuned (e.g. ``tiny_llm.data=DEBUG`` while keeping
    ``tiny_llm.model=WARNING``).

    Can be called more than once — existing handlers are replaced.

    Args:
        root_level: Default log level for the ``tiny_llm`` root logger.
        component_levels: Mapping of logger names to log levels, e.g.
            ``{"tiny_llm.data": "DEBUG", "tiny_llm.training": "WARNING"}``.
        log_format: Custom format string (falls back to a sensible default).
        date_format: Custom date format string.
    """
    global _initialized
    global _active_log_format
    global _active_date_format

    fmt = log_format or _DEFAULT_FORMAT
    datefmt = date_format or _DEFAULT_DATE_FORMAT
    _active_log_format = fmt
    _active_date_format = datefmt

    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # Shared handler used by all tiny_llm loggers
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Configure the root tiny_llm logger (avoids touching the global root)
    piko_root = logging.getLogger("tiny_llm")
    piko_root.setLevel(getattr(logging, root_level.upper(), logging.INFO))
    piko_root.handlers.clear()
    piko_root.addHandler(handler)
    piko_root.propagate = False  # don't double-emit via the global root

    # Also configure a logger for the CLI entry-point (main.py)
    cli_logger = logging.getLogger("tiny_llm.cli")
    cli_logger.setLevel(getattr(logging, root_level.upper(), logging.INFO))

    # Apply per-component overrides
    if component_levels:
        for name, level in component_levels.items():
            comp_logger = logging.getLogger(name)
            comp_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    _initialized = True


def add_file_logging(log_file: str | Path) -> Path:
    """Attach a file handler to the ``tiny_llm`` logger (idempotent).

    This persists all standard logs that already go to stdout into a run-local
    file (e.g. ``checkpoints/final/training.log``).
    """
    if not _initialized:
        setup_logging()

    log_path = Path(log_file).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    piko_root = logging.getLogger("tiny_llm")
    managed_handlers: list[logging.FileHandler] = []
    for handler in piko_root.handlers:
        if isinstance(handler, logging.FileHandler):
            existing = Path(getattr(handler, "baseFilename", ""))
            if existing == log_path:
                return log_path
            if getattr(handler, "_managed_file", False):
                managed_handlers.append(handler)

    for handler in managed_handlers:
        piko_root.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    setattr(file_handler, "_managed_file", True)
    file_handler.setFormatter(
        logging.Formatter(_active_log_format, datefmt=_active_date_format)
    )
    piko_root.addHandler(file_handler)
    return log_path


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, lazily initialising the framework if needed.

    If :func:`setup_logging` has not been called yet, it is invoked with
    default settings so that callers never get a silent logger.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).
    """
    if not _initialized:
        setup_logging()
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Metrics logger — training metrics to stdout
# ---------------------------------------------------------------------------


class MetricsLogger:
    """Training metrics logger that writes structured lines to stdout and files.

    Used by the Trainer to log per-step metrics in a machine-parseable format:
        step 100 | train/loss=8.234 | train/lr=0.0006 | train/tokens_per_sec=150000

    Persistence writes one row per optimiser step to ``metrics.csv`` and
    ``metrics.jsonl`` inside the checkpoint directory. At eval steps the trainer
    calls ``log()`` twice (train metrics, then val metrics) for the same step —
    the logger buffers by step number and merges both calls into a single row.

    Plotting:
      - quick-look PNG refreshed every N steps (default: 100)
      - detailed 6-panel PNG generated at training end

    Regular progress messages (e.g. "Checkpoint saved") should use
    Python's logging module via get_logger() instead.
    """

    _CSV_COLUMNS = [
        "step",
        "timestamp",
        "train/loss",
        "train/perplexity",
        "train/lr",
        "train/tokens_seen",
        "train/tokens_per_sec",
        "train/grad_norm",
        "val/loss",
        "val/perplexity",
    ]

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        self._logger = get_logger("tiny_llm.metrics")

        # ── CSV setup (graceful degradation if no checkpoint_dir) ──
        self._csv_file = None
        self._csv_writer = None
        self._jsonl_file = None
        self._current_row: dict[str, Any] = {}
        self._current_step: int | None = None
        self._current_epoch: int | None = None
        self._history_rows: list[dict[str, Any]] = []

        self._checkpoint_dir: Path | None = None
        self._plots_dir: Path | None = None
        self._quick_plot_path: Path | None = None
        self._detailed_plot_path: Path | None = None
        self._quick_plot_interval = 100
        self._last_quick_plot_step = -1

        self._plotting_enabled = False
        if config and config.get("checkpoint_dir"):
            self._checkpoint_dir = Path(config["checkpoint_dir"])
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

            csv_path = self._checkpoint_dir / "metrics.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            # Resume: append without re-writing header if file already has content
            file_exists = csv_path.exists() and csv_path.stat().st_size > 0
            self._csv_file = open(csv_path, "a", newline="", encoding="utf-8")  # noqa: SIM115
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=self._CSV_COLUMNS,
                extrasaction="ignore",
            )
            if not file_exists:
                self._csv_writer.writeheader()
                self._csv_file.flush()
            self._logger.info("CSV metrics → %s", csv_path)

            jsonl_path = self._checkpoint_dir / "metrics.jsonl"
            self._jsonl_file = open(jsonl_path, "a", encoding="utf-8")  # noqa: SIM115
            self._logger.info("JSONL metrics → %s", jsonl_path)

            self._quick_plot_interval = max(
                1,
                int(config.get("quick_plot_interval", 100)),
            )
            self._plots_dir = self._checkpoint_dir / "plots"
            self._plots_dir.mkdir(parents=True, exist_ok=True)
            self._quick_plot_path = self._plots_dir / "training_quicklook.png"
            self._detailed_plot_path = self._plots_dir / "training_detailed.png"
            self._plotting_enabled = True

    def log(
        self,
        metrics: dict[str, Any],
        step: int,
        epoch: int | None = None,
    ) -> None:
        """Log metrics at a given step (stdout + persisted rows)."""
        # ── Stdout ──
        parts = " | ".join(f"{k}={v}" for k, v in metrics.items())
        self._logger.info("step %d | %s", step, parts)

        # ── CSV buffering ──
        if self._csv_writer is not None:
            if self._current_step is not None and step != self._current_step:
                self._flush_row()
            if self._current_step != step:
                self._current_step = step
                self._current_epoch = epoch
            elif epoch is not None and self._current_epoch is None:
                self._current_epoch = epoch
            self._current_row.update(metrics)

    def maybe_save_quick_plot(self, step: int) -> Path | None:
        """Refresh a lightweight training plot every ``quick_plot_interval`` steps."""
        if not self._plotting_enabled:
            return None
        if step <= 0 or step % self._quick_plot_interval != 0:
            return None
        if step == self._last_quick_plot_step:
            return None
        path = self._save_quick_plot(step)
        if path is not None:
            self._last_quick_plot_step = step
        return path

    def save_detailed_plot(self) -> Path | None:
        """Create and save a detailed 6-panel training dashboard."""
        if not self._plotting_enabled:
            return None
        return self._save_detailed_plot()

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _rows_for_plot(self) -> list[dict[str, Any]]:
        """Return flushed rows plus the currently buffered step."""
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

    def _save_quick_plot(self, step: int) -> Path | None:
        """Render and save a compact four-panel snapshot."""
        rows = self._rows_for_plot()
        if not rows or self._quick_plot_path is None:
            return None

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - depends on runtime env
            self._logger.warning("Quick plot skipped (matplotlib unavailable): %s", exc)
            self._plotting_enabled = False
            return None

        train_steps, train_loss = self._series(rows, "train/loss")
        val_steps, val_loss = self._series(rows, "val/loss")
        train_ppl_steps, train_ppl = self._series(rows, "train/perplexity")
        val_ppl_steps, val_ppl = self._series(rows, "val/perplexity")
        lr_steps, lrs = self._series(rows, "train/lr")
        grad_steps, grad_norm = self._series(rows, "train/grad_norm")

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle(f"Quick Training Snapshot — step {step}", fontsize=12, fontweight="bold")

        ax = axes[0, 0]
        if train_steps:
            ax.plot(train_steps, train_loss, color="#1f77b4", linewidth=1.0, label="Train")
        if val_steps:
            ax.plot(val_steps, val_loss, color="#ff7f0e", linewidth=1.2, marker="o", markersize=3, label="Validation")
        ax.set_yscale("log")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Train vs Validation Loss")
        ax.grid(True, alpha=0.3)
        if train_steps or val_steps:
            ax.legend()

        ax = axes[0, 1]
        if train_ppl_steps:
            ax.plot(train_ppl_steps, train_ppl, color="#2ca02c", linewidth=1.0, label="Train")
        if val_ppl_steps:
            ax.plot(val_ppl_steps, val_ppl, color="#d62728", linewidth=1.2, marker="o", markersize=3, label="Validation")
        ax.set_yscale("log")
        ax.set_xlabel("Step")
        ax.set_ylabel("Perplexity")
        ax.set_title("Train vs Validation Perplexity")
        ax.grid(True, alpha=0.3)
        if train_ppl_steps or val_ppl_steps:
            ax.legend()

        ax = axes[1, 0]
        if lr_steps:
            ax.plot(lr_steps, lrs, color="#2ca02c", linewidth=1.4)
        ax.set_title("Learning Rate Schedule")
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        if grad_steps:
            ax.plot(grad_steps, grad_norm, color="#d62728", linewidth=1.0)
        ax.set_title("Gradient Norm Over Steps")
        ax.set_xlabel("Step")
        ax.set_ylabel("Gradient Norm")
        ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        fig.savefig(self._quick_plot_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return self._quick_plot_path

    def _save_detailed_plot(self) -> Path | None:
        """Render and save a detailed six-panel dashboard."""
        rows = self._rows_for_plot()
        if not rows or self._detailed_plot_path is None:
            return None

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - depends on runtime env
            self._logger.warning(
                "Detailed plot skipped (matplotlib unavailable): %s", exc
            )
            self._plotting_enabled = False
            return None

        run_name = self._checkpoint_dir.parent.name if self._checkpoint_dir else "training_run"
        fig, axes = plt.subplots(2, 3, figsize=(17, 9))
        fig.suptitle(f"Detailed Overview: {run_name}", fontsize=16, fontweight="bold")

        train_steps, train_loss = self._series(rows, "train/loss")
        train_ppl_steps, train_ppl = self._series(rows, "train/perplexity")
        lr_steps, lrs = self._series(rows, "train/lr")
        grad_steps, grad_norm = self._series(rows, "train/grad_norm")
        val_steps, val_loss = self._series(rows, "val/loss")
        speed_steps, speed_vals = self._series(rows, "train/tokens_per_sec")

        ax = axes[0, 0]
        if train_steps:
            ax.plot(train_steps, train_loss, color="#1f77b4", linewidth=1.0)
            ax.set_yscale("log")
        ax.set_title("Train Loss Over Steps")
        ax.set_xlabel("Step")
        ax.set_ylabel("Train Loss")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        if train_ppl_steps:
            ax.plot(train_ppl_steps, train_ppl, color="#ff7f0e", linewidth=1.0)
            ax.set_yscale("log")
        ax.set_title("Training Perplexity Over Steps")
        ax.set_xlabel("Step")
        ax.set_ylabel("Perplexity")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
        if lr_steps:
            ax.plot(lr_steps, lrs, color="#2ca02c", linewidth=1.4)
        ax.set_title("Learning Rate Schedule")
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        if grad_steps:
            ax.plot(grad_steps, grad_norm, color="#d62728", linewidth=1.0)
        ax.set_title("Gradient Norm Over Steps")
        ax.set_xlabel("Step")
        ax.set_ylabel("Gradient Norm")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        if train_steps:
            ax.plot(train_steps, train_loss, color="#ff7f0e", linewidth=1.0, label="Train")
        if val_steps:
            ax.plot(val_steps, val_loss, color="#1f77b4", linewidth=1.2, marker="o", markersize=3, label="Validation")
        if val_steps or train_steps:
            ax.set_yscale("log")
            ax.legend()
        ax.set_title("Loss per Eval Step")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        if speed_steps:
            ax.plot(speed_steps, speed_vals, color="#9467bd", linewidth=1.0)
        ax.set_title("Training Speed")
        ax.set_xlabel("Step")
        ax.set_ylabel("Tokens/sec")
        ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        fig.savefig(self._detailed_plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return self._detailed_plot_path

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
        """Write the buffered row to CSV and reset the buffer."""
        if not self._current_row or self._csv_writer is None:
            return
        row = {
            "step": self._current_step,
            "epoch": self._current_epoch,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        row.update(self._current_row)

        self._csv_writer.writerow(row)
        self._csv_file.flush()  # crash safety

        if self._jsonl_file is not None:
            safe_row = {k: self._json_safe(v) for k, v in row.items()}
            self._jsonl_file.write(json.dumps(safe_row) + "\n")
            self._jsonl_file.flush()

        self._history_rows.append(row)
        self._current_row = {}
        self._current_step = None
        self._current_epoch = None

    def finish(self) -> None:
        """Finalise logging — flush pending row and close CSV."""
        self._flush_row()
        detailed_plot_path = self.save_detailed_plot()
        if detailed_plot_path is not None:
            self._logger.info("Detailed training plot → %s", detailed_plot_path)

        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        if self._jsonl_file is not None:
            self._jsonl_file.close()
            self._jsonl_file = None
        self._logger.info("Training metrics logging finished.")
