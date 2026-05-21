"""Metrics logger for DPO training.

Subclasses :class:`SFTMetricsLogger` to reuse file I/O (CSV, JSONL) and
override the column set + plot panels for DPO-specific metrics.  Plot layout
mirrors EX08's ``plot_training_history()`` 3-panel design.
"""

from __future__ import annotations

from pathlib import Path

from tiny_llm.posttraining.sft_metrics import SFTMetricsLogger


class DPOMetricsLogger(SFTMetricsLogger):
    """Structured metrics logger for DPO runs."""

    _CSV_COLUMNS = [
        "step",
        "epoch",
        "timestamp",
        "train/loss",
        "train/lr",
        "train/grad_norm",
        "train/policy_logratios",
        "train/ref_logratios",
        "train/advantages",
        "train/implicit_reward_chosen",
        "train/implicit_reward_rejected",
        "train/implicit_reward_margin",
        "train/accuracy",
        "val/loss",
        "val/accuracy",
        "val/implicit_reward_margin",
        "pretrain/val_loss",
        "pretrain/val_perplexity",
    ]

    def _save_plot(self, step: int) -> Path | None:
        """Quick 3-panel DPO dashboard (mirrors EX08's plot_training_history)."""
        rows = self._rows_for_plot()
        if not rows or self._quick_plot_path is None:
            return None

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            self._logger.warning(
                "DPO plot skipped (matplotlib unavailable): %s", exc
            )
            self._plotting_enabled = False
            return None

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(
            f"DPO Training — step {step}",
            fontsize=12,
            fontweight="bold",
        )

        # Panel 1: DPO Loss
        ax = axes[0]
        steps, vals = self._series(rows, "train/loss")
        if steps:
            ax.plot(steps, vals, color="#1f77b4", linewidth=1.0, label="Train")
        vs, vv = self._series(rows, "val/loss")
        if vs:
            ax.plot(
                vs, vv, color="#ff7f0e", linewidth=1.3,
                marker="o", markersize=3, label="Val",
            )
        ax.set_title("DPO Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        if steps or vs:
            ax.legend()

        # Panel 2: Preference Signal (advantages + implicit reward margin)
        ax = axes[1]
        adv_s, adv_v = self._series(rows, "train/advantages")
        if adv_s:
            ax.plot(
                adv_s, adv_v, color="#2ca02c", linewidth=1.0,
                label="advantages",
            )
        mrg_s, mrg_v = self._series(rows, "train/implicit_reward_margin")
        if mrg_s:
            ax.plot(
                mrg_s, mrg_v, color="#d62728", linewidth=1.0,
                label="implicit_reward_margin",
            )
        ax.set_title("Preference Signal")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        if adv_s or mrg_s:
            ax.legend()

        # Panel 3: Policy vs Reference log-ratios
        ax = axes[2]
        pl_s, pl_v = self._series(rows, "train/policy_logratios")
        if pl_s:
            ax.plot(
                pl_s, pl_v, color="#1f77b4", linewidth=1.0,
                label="policy_logratio",
            )
        rl_s, rl_v = self._series(rows, "train/ref_logratios")
        if rl_s:
            ax.plot(
                rl_s, rl_v, color="#ff7f0e", linewidth=1.0,
                label="ref_logratio",
            )
        ax.set_title("Policy vs Reference")
        ax.set_xlabel("Step")
        ax.set_ylabel("Log-ratio")
        ax.grid(True, alpha=0.3)
        if pl_s or rl_s:
            ax.legend()

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        fig.savefig(self._quick_plot_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return self._quick_plot_path

    def _save_detailed_plot(self) -> Path | None:
        """Detailed 6-panel DPO overview."""
        rows = self._rows_for_plot()
        if not rows or self._detailed_plot_path is None:
            return None

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            self._logger.warning(
                "Detailed DPO plot skipped (matplotlib unavailable): %s", exc
            )
            self._plotting_enabled = False
            return None

        run_name = (
            self._checkpoint_dir.name if self._checkpoint_dir else "dpo_run"
        )
        fig, axes = plt.subplots(2, 3, figsize=(17, 9))
        fig.suptitle(
            f"DPO Detailed Overview — {run_name}",
            fontsize=16,
            fontweight="bold",
        )

        ax = axes[0, 0]
        ts, tv = self._series(rows, "train/loss")
        if ts:
            ax.plot(ts, tv, color="#1f77b4", linewidth=1.0, label="Train")
        vs, vv = self._series(rows, "val/loss")
        if vs:
            ax.plot(
                vs, vv, color="#ff7f0e", linewidth=1.2,
                marker="o", markersize=3, label="Val",
            )
        ax.set_title("DPO Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        if ts or vs:
            ax.legend()

        ax = axes[0, 1]
        ta_s, ta_v = self._series(rows, "train/accuracy")
        if ta_s:
            ax.plot(ta_s, ta_v, color="#2ca02c", linewidth=1.0, label="Train")
        va_s, va_v = self._series(rows, "val/accuracy")
        if va_s:
            ax.plot(
                va_s, va_v, color="#ff7f0e", linewidth=1.2,
                marker="o", markersize=3, label="Val",
            )
        ax.set_title("DPO Accuracy (chosen preferred)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if ta_s or va_s:
            ax.legend()

        ax = axes[0, 2]
        m_s, m_v = self._series(rows, "train/implicit_reward_margin")
        if m_s:
            ax.plot(m_s, m_v, color="#d62728", linewidth=1.0, label="Train")
        vm_s, vm_v = self._series(rows, "val/implicit_reward_margin")
        if vm_s:
            ax.plot(
                vm_s, vm_v, color="#ff7f0e", linewidth=1.2,
                marker="o", markersize=3, label="Val",
            )
        ax.set_title("Implicit Reward Margin")
        ax.set_xlabel("Step")
        ax.set_ylabel("Margin")
        ax.grid(True, alpha=0.3)
        if m_s or vm_s:
            ax.legend()

        ax = axes[1, 0]
        lr_s, lr_v = self._series(rows, "train/lr")
        if lr_s:
            ax.plot(lr_s, lr_v, color="#9467bd", linewidth=1.4)
        ax.set_title("Learning Rate Schedule")
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        gn_s, gn_v = self._series(rows, "train/grad_norm")
        if gn_s:
            ax.plot(gn_s, gn_v, color="#8c564b", linewidth=1.0)
        ax.set_title("Gradient Norm")
        ax.set_xlabel("Step")
        ax.set_ylabel("||grad||")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        pp_s, pp_v = self._series(rows, "pretrain/val_perplexity")
        if pp_s:
            ax.plot(
                pp_s, pp_v, color="#d62728", linewidth=1.2,
                marker="s", markersize=3,
            )
        ax.set_title("Pretrain-val Perplexity\n(rising = forgetting)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Perplexity")
        ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        fig.savefig(self._detailed_plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return self._detailed_plot_path


__all__ = ["DPOMetricsLogger"]
