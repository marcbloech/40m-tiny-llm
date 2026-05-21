"""Direct Preference Optimization trainer.

Follows the EX08 DPO notebook pattern -- four forward passes per batch
(policy-chosen, policy-rejected, ref-chosen, ref-rejected) with the DPO loss
from Rafailov et al. (2023).  The infrastructure (gradient accumulation,
checkpointing, cosine schedule) mirrors :class:`SFTTrainer`.
"""

from __future__ import annotations

import logging
import math
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from tiny_llm.posttraining.dpo_dataset import (
    EOS_TOKEN_ID,
    DPODataset,
    dpo_collate_fn,
)
from tiny_llm.posttraining.dpo_metrics import DPOMetricsLogger
from tiny_llm.training.checkpoint import (
    prune_checkpoints,
    save_best_checkpoint,
    save_checkpoint,
)
from tiny_llm.training.optimizer import create_optimizer, create_scheduler
from tiny_llm.utils.logging import add_file_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core DPO helpers (matching EX08 function signatures)
# ---------------------------------------------------------------------------


def sequence_logprob_from_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    length_normalized: bool = False,
) -> torch.Tensor:
    """Compute log p(target tokens) for each sequence in the batch.

    Args:
        logits: ``(B, S, V)`` model output.
        labels: ``(B, S)`` with ``-100`` on tokens to ignore (prompt / pad).
        length_normalized: If ``True``, return the *mean* log-prob per
            non-masked token instead of the sum.  This removes the bias
            toward shorter sequences in the DPO loss.

    Returns:
        ``(B,)`` -- sum (or mean) of log-probs over non-masked tokens.
    """
    if logits.size(1) < 2:
        return torch.zeros(logits.size(0), device=logits.device, dtype=logits.dtype)

    # Decoder logits at position t predict the token at t+1.  DPO labels use
    # full-sequence positions with prompt/pad masked, so shift labels here.
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]

    log_probs = F.log_softmax(shifted_logits.float(), dim=-1)

    safe_labels = shifted_labels.clone()
    safe_labels[safe_labels == -100] = 0

    token_log_probs = log_probs.gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)

    mask = (shifted_labels != -100).float()
    summed = (token_log_probs * mask).sum(dim=-1)
    if length_normalized:
        return summed / mask.sum(dim=-1).clamp(min=1)
    return summed


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
    reference_free: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """DPO loss following EX08.

    Returns ``(loss, metrics_dict)`` where *metrics_dict* contains the same
    keys as EX08: ``policy_logratios``, ``ref_logratios``, ``advantages``,
    ``implicit_reward_chosen``, ``implicit_reward_rejected``,
    ``implicit_reward_margin``.
    """
    if reference_free:
        ref_chosen_logp = torch.zeros_like(ref_chosen_logp)
        ref_rejected_logp = torch.zeros_like(ref_rejected_logp)

    policy_logratios = policy_chosen_logp - policy_rejected_logp
    ref_logratios = ref_chosen_logp - ref_rejected_logp
    advantages = policy_logratios - ref_logratios

    if loss_type == "sigmoid":
        losses = (
            -F.logsigmoid(beta * advantages) * (1 - label_smoothing)
            - F.logsigmoid(-beta * advantages) * label_smoothing
        )
    elif loss_type == "ipo":
        losses = (advantages - 1.0 / (2.0 * beta)) ** 2
    else:
        raise ValueError(f"Unknown DPO loss_type: {loss_type!r}")

    loss = losses.mean()

    implicit_reward_chosen = beta * (policy_chosen_logp - ref_chosen_logp)
    implicit_reward_rejected = beta * (policy_rejected_logp - ref_rejected_logp)

    metrics = {
        "policy_logratios": policy_logratios.detach().mean().item(),
        "ref_logratios": ref_logratios.detach().mean().item(),
        "advantages": advantages.detach().mean().item(),
        "implicit_reward_chosen": implicit_reward_chosen.detach().mean().item(),
        "implicit_reward_rejected": implicit_reward_rejected.detach().mean().item(),
        "implicit_reward_margin": (
            implicit_reward_chosen - implicit_reward_rejected
        )
        .detach()
        .mean()
        .item(),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# DPOTrainer
# ---------------------------------------------------------------------------


class DPOTrainer:
    """Single-device DPO trainer (CPU / MPS / CUDA)."""

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        train_dataset: DPODataset,
        val_dataset: DPODataset | None,
        config: dict[str, Any],
        device: torch.device,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        self.device = device
        self.config = config
        self.model_config = model_config or {}

        checkpoint_dir = Path(config["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        add_file_logging(checkpoint_dir / "dpo.log")

        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.ref_model.requires_grad_(False)
        self.ref_model.eval()

        self.beta = float(config.get("dpo_beta", 0.1))
        self.label_smoothing = float(config.get("dpo_label_smoothing", 0.0))
        self.loss_type = str(config.get("dpo_loss_type", "sigmoid"))
        self.reference_free = bool(config.get("dpo_reference_free", False))
        self.length_normalized = bool(
            config.get("dpo_length_normalized", False)
        )

        collate = partial(dpo_collate_fn, pad_token_id=EOS_TOKEN_ID)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config.get("num_workers", 2),
            pin_memory=config.get("pin_memory", True) and device.type == "cuda",
            drop_last=True,
            collate_fn=collate,
        )

        self.val_loader = None
        if val_dataset is not None and len(val_dataset) > 0:
            eval_bs = config.get("eval_batch_size") or config["batch_size"]
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=eval_bs,
                shuffle=False,
                num_workers=config.get("num_workers", 2),
                pin_memory=config.get("pin_memory", True)
                and device.type == "cuda",
                drop_last=False,
                collate_fn=collate,
            )

        grad_accum = int(config.get("grad_accum_steps", 1))
        steps_per_epoch = max(1, len(self.train_loader) // max(1, grad_accum))
        self.max_steps = steps_per_epoch * int(config["epochs"])
        self.warmup_steps = max(
            1, int(config.get("warmup_ratio", 0.0) * self.max_steps)
        )

        self.optimizer = create_optimizer(
            self.model,
            optimizer_name=config.get("optimizer", "adamw"),
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=config.get("betas", [0.9, 0.95]),
        )
        self.scheduler = create_scheduler(
            self.optimizer,
            lr_schedule="cosine",
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            min_lr_ratio=0.1,
        )

        self.pretrain_val_tokens: np.memmap | None = None
        pv_path = config.get("pretrain_val_path")
        if pv_path:
            p = Path(pv_path)
            if p.exists():
                self.pretrain_val_tokens = np.memmap(
                    str(p), dtype=np.uint16, mode="r"
                )
                logger.info(
                    "[dpo] Pretrain-val probe enabled: %s (%d tokens)",
                    p,
                    len(self.pretrain_val_tokens),
                )
            else:
                logger.warning(
                    "[dpo] pretrain_val_path=%s not found -- forgetting probe disabled",
                    pv_path,
                )

        self.best_val_loss = float("inf")
        self.metrics_logger = DPOMetricsLogger(config=config)

        total_params = sum(p.numel() for p in self.model.parameters())
        effective_batch = config["batch_size"] * grad_accum
        logger.info(
            "[dpo] Training setup: %s params, %d train / %d val examples, "
            "%d epochs x %d steps/epoch = %d optim steps, "
            "effective_batch=%d, warmup=%d, lr=%.2e, beta=%.3f, "
            "length_norm=%s, device=%s",
            f"{total_params:,}",
            len(train_dataset),
            len(val_dataset) if val_dataset else 0,
            config["epochs"],
            steps_per_epoch,
            self.max_steps,
            effective_batch,
            self.warmup_steps,
            config["learning_rate"],
            self.beta,
            self.length_normalized,
            device,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _assess_dpo_val(self) -> dict[str, float]:
        """DPO metrics on the val set."""
        if self.val_loader is None:
            return {}
        was_training = self.model.training
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_count = 0
        total_margin = 0.0

        for batch in self.val_loader:
            chosen_ids = batch["chosen_input_ids"].to(self.device)
            chosen_labels = batch["chosen_labels"].to(self.device)
            rejected_ids = batch["rejected_input_ids"].to(self.device)
            rejected_labels = batch["rejected_labels"].to(self.device)

            policy_chosen_logits = self.model(chosen_ids)[0]
            policy_rejected_logits = self.model(rejected_ids)[0]
            ref_chosen_logits = self.ref_model(chosen_ids)[0]
            ref_rejected_logits = self.ref_model(rejected_ids)[0]

            policy_chosen_logp = sequence_logprob_from_labels(
                policy_chosen_logits, chosen_labels,
                length_normalized=self.length_normalized,
            )
            policy_rejected_logp = sequence_logprob_from_labels(
                policy_rejected_logits, rejected_labels,
                length_normalized=self.length_normalized,
            )
            ref_chosen_logp = sequence_logprob_from_labels(
                ref_chosen_logits, chosen_labels,
                length_normalized=self.length_normalized,
            )
            ref_rejected_logp = sequence_logprob_from_labels(
                ref_rejected_logits, rejected_labels,
                length_normalized=self.length_normalized,
            )

            loss, _ = dpo_loss(
                policy_chosen_logp,
                policy_rejected_logp,
                ref_chosen_logp,
                ref_rejected_logp,
                beta=self.beta,
                label_smoothing=self.label_smoothing,
                loss_type=self.loss_type,
                reference_free=self.reference_free,
            )

            B = chosen_ids.size(0)
            total_loss += loss.item() * B
            total_count += B

            chosen_reward = self.beta * (
                policy_chosen_logp - ref_chosen_logp
            )
            rejected_reward = self.beta * (
                policy_rejected_logp - ref_rejected_logp
            )
            total_correct += int((chosen_reward > rejected_reward).sum().item())
            total_margin += (chosen_reward - rejected_reward).sum().item()

        if was_training:
            self.model.train(True)

        if total_count == 0:
            return {}

        return {
            "val/loss": round(total_loss / total_count, 4),
            "val/accuracy": round(total_correct / total_count, 4),
            "val/implicit_reward_margin": round(
                total_margin / total_count, 4
            ),
        }

    @torch.no_grad()
    def _assess_pretrain_val(self) -> float:
        """Cross-entropy on random pre-train val chunks (forgetting probe)."""
        if self.pretrain_val_tokens is None:
            return float("nan")

        ctx_len = int(self.model_config.get("context_length", 1024))
        n_batches = int(self.config.get("pretrain_val_batches", 40))
        batch_size = int(
            self.config.get("eval_batch_size") or self.config["batch_size"]
        )
        tokens = self.pretrain_val_tokens

        max_start = len(tokens) - ctx_len - 1
        if max_start <= 0:
            return float("nan")

        rng = np.random.default_rng(seed=12345)
        total_samples = n_batches * batch_size
        starts = rng.integers(low=0, high=max_start, size=total_samples)

        was_training = self.model.training
        self.model.eval()
        loss_sum = 0.0
        token_count = 0
        for b in range(n_batches):
            batch_starts = starts[b * batch_size : (b + 1) * batch_size]
            chunks = np.stack(
                [
                    np.asarray(tokens[s : s + ctx_len + 1], dtype=np.int64)
                    for s in batch_starts
                ]
            )
            chunk_t = torch.from_numpy(chunks).to(self.device)
            x = chunk_t[:, :-1]
            y = chunk_t[:, 1:]
            logits, _ = self.model(x)
            vocab = logits.size(-1)
            batch_loss = F.cross_entropy(
                logits.view(-1, vocab),
                y.reshape(-1),
                reduction="sum",
            )
            loss_sum += float(batch_loss.item())
            token_count += int(y.numel())

        if was_training:
            self.model.train(True)
        if token_count == 0:
            return float("nan")
        return loss_sum / token_count

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        cfg = self.config
        grad_accum_steps = int(cfg.get("grad_accum_steps", 1))
        gradient_clip_norm = float(cfg.get("gradient_clip_norm", 1.0))
        log_interval = int(cfg.get("log_interval", 10))
        eval_interval = int(cfg.get("eval_interval", 100))
        pretrain_eval_interval = int(cfg.get("pretrain_eval_interval", 200))
        checkpoint_interval = int(cfg.get("checkpoint_interval", 500))
        checkpoint_dir = Path(cfg["checkpoint_dir"])
        max_checkpoints = int(cfg.get("max_checkpoints", 3))
        max_best = int(cfg.get("max_best_checkpoints", 3))

        self.model.train(True)
        self.optimizer.zero_grad(set_to_none=True)

        step = 0
        total_batches = len(self.train_loader)
        train_t0 = time.time()

        for epoch in range(int(cfg["epochs"])):
            logger.info(
                "[dpo] Epoch %d/%d starting (%d micro-batches, %d optim steps expected)",
                epoch + 1, int(cfg["epochs"]),
                total_batches, self.max_steps,
            )

            window_loss = 0.0
            window_metrics: dict[str, float] = {}
            window_correct = 0
            window_count = 0
            micro_count = 0
            batch_idx = 0

            for batch in self.train_loader:
                batch_idx += 1
                chosen_ids = batch["chosen_input_ids"].to(self.device)
                chosen_labels = batch["chosen_labels"].to(self.device)
                rejected_ids = batch["rejected_input_ids"].to(self.device)
                rejected_labels = batch["rejected_labels"].to(self.device)

                policy_chosen_logits = self.model(chosen_ids)[0]
                policy_rejected_logits = self.model(rejected_ids)[0]

                with torch.no_grad():
                    ref_chosen_logits = self.ref_model(chosen_ids)[0]
                    ref_rejected_logits = self.ref_model(rejected_ids)[0]

                policy_chosen_logp = sequence_logprob_from_labels(
                    policy_chosen_logits, chosen_labels,
                    length_normalized=self.length_normalized,
                )
                policy_rejected_logp = sequence_logprob_from_labels(
                    policy_rejected_logits, rejected_labels,
                    length_normalized=self.length_normalized,
                )
                ref_chosen_logp = sequence_logprob_from_labels(
                    ref_chosen_logits, chosen_labels,
                    length_normalized=self.length_normalized,
                )
                ref_rejected_logp = sequence_logprob_from_labels(
                    ref_rejected_logits, rejected_labels,
                    length_normalized=self.length_normalized,
                )

                loss, metrics = dpo_loss(
                    policy_chosen_logp,
                    policy_rejected_logp,
                    ref_chosen_logp,
                    ref_rejected_logp,
                    beta=self.beta,
                    label_smoothing=self.label_smoothing,
                    loss_type=self.loss_type,
                    reference_free=self.reference_free,
                )

                scaled_loss = loss / grad_accum_steps
                scaled_loss.backward()

                B = chosen_ids.size(0)
                window_loss += loss.detach().item()
                for k, v in metrics.items():
                    window_metrics[k] = window_metrics.get(k, 0.0) + v
                chosen_reward = self.beta * (
                    policy_chosen_logp - ref_chosen_logp
                )
                rejected_reward = self.beta * (
                    policy_rejected_logp - ref_rejected_logp
                )
                window_correct += int(
                    (chosen_reward > rejected_reward).detach().sum().item()
                )
                window_count += B
                micro_count += 1

                # Heartbeat so long accumulation windows don't look stuck
                if grad_accum_steps > 1:
                    elapsed = time.time() - train_t0
                    print(
                        f"    micro {batch_idx}/{total_batches} "
                        f"(accum {micro_count}/{grad_accum_steps}) "
                        f"| loss={loss.item():.4f} "
                        f"| elapsed={elapsed:.0f}s",
                        flush=True,
                    )

                if micro_count < grad_accum_steps:
                    continue

                grad_norm = (
                    clip_grad_norm_(
                        self.model.parameters(), max_norm=gradient_clip_norm
                    )
                    if gradient_clip_norm > 0
                    else None
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                accumulated_loss = window_loss / micro_count
                accumulated_accuracy = (
                    window_correct / window_count if window_count else 0.0
                )
                avg_metrics = {
                    k: v / micro_count for k, v in window_metrics.items()
                }

                window_loss = 0.0
                window_metrics = {}
                window_correct = 0
                window_count = 0
                micro_count = 0
                step += 1

                if step % log_interval == 0 or step == 1:
                    current_lr = self.scheduler.get_last_lr()[0]
                    log_metrics: dict[str, Any] = {
                        "train/loss": round(accumulated_loss, 4),
                        "train/lr": current_lr,
                        "train/accuracy": round(accumulated_accuracy, 4),
                    }
                    if grad_norm is not None:
                        gn = (
                            grad_norm.item()
                            if isinstance(grad_norm, torch.Tensor)
                            else grad_norm
                        )
                        log_metrics["train/grad_norm"] = round(float(gn), 4)
                    for k, v in avg_metrics.items():
                        log_metrics[f"train/{k}"] = round(v, 4)
                    self.metrics_logger.log(
                        log_metrics, step=step, epoch=epoch
                    )

                pct = step / self.max_steps * 100
                elapsed = time.time() - train_t0
                if step > 0:
                    eta_s = elapsed / step * (self.max_steps - step)
                    eta_str = f"{eta_s / 60:.1f}min" if eta_s > 60 else f"{eta_s:.0f}s"
                else:
                    eta_str = "?"
                print(
                    f"  dpo step {step}/{self.max_steps} "
                    f"({pct:.0f}%) | loss={accumulated_loss:.4f} "
                    f"| acc={accumulated_accuracy:.3f} "
                    f"| elapsed={elapsed:.0f}s | eta={eta_str}",
                    flush=True,
                )

                if step > 0 and step % eval_interval == 0:
                    val_metrics = self._assess_dpo_val()
                    if val_metrics:
                        self.metrics_logger.log(
                            val_metrics, step=step, epoch=epoch
                        )
                        val_loss = val_metrics.get("val/loss", float("inf"))
                        saved = save_best_checkpoint(
                            self.model,
                            step=step,
                            config=self.config,
                            checkpoint_dir=checkpoint_dir,
                            val_loss=val_loss,
                            tokens_seen=0,
                            model_config=self.model_config,
                            max_best=max_best,
                        )
                        if saved is not None:
                            logger.info(
                                "[dpo] Best checkpoint saved: %s (val_loss=%.4f)",
                                saved.name,
                                val_loss,
                            )
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss

                if (
                    self.pretrain_val_tokens is not None
                    and step > 0
                    and step % pretrain_eval_interval == 0
                ):
                    pre_loss = self._assess_pretrain_val()
                    if not math.isnan(pre_loss):
                        pre_ppl = math.exp(min(pre_loss, 20))
                        self.metrics_logger.log(
                            {
                                "pretrain/val_loss": round(pre_loss, 4),
                                "pretrain/val_perplexity": round(pre_ppl, 2),
                            },
                            step=step,
                            epoch=epoch,
                        )

                if step > 0 and step % checkpoint_interval == 0:
                    save_checkpoint(
                        self.model,
                        optimizer=self.optimizer,
                        step=step,
                        config=self.config,
                        path=checkpoint_dir / f"step_{step:06d}.pt",
                        val_loss=self.best_val_loss,
                        tokens_seen=0,
                        model_config=self.model_config,
                    )
                    prune_checkpoints(
                        checkpoint_dir, max_keep=max_checkpoints
                    )

                quick_plot = self.metrics_logger.maybe_save_quick_plot(step)
                if quick_plot is not None:
                    logger.info("[dpo] Quick plot -> %s", quick_plot)

        final_val = self._assess_dpo_val()
        if final_val:
            logger.info(
                "[dpo] Final val_loss=%.4f val_acc=%.4f",
                final_val.get("val/loss", float("nan")),
                final_val.get("val/accuracy", float("nan")),
            )

        save_checkpoint(
            self.model,
            optimizer=self.optimizer,
            step=step,
            config=self.config,
            path=checkpoint_dir / f"step_{step:06d}.pt",
            val_loss=(
                final_val.get("val/loss")
                if final_val
                else None
            ),
            tokens_seen=0,
            model_config=self.model_config,
        )

        self.metrics_logger.finish()
        logger.info("[dpo] Training complete.")


__all__ = ["DPOTrainer", "dpo_loss", "sequence_logprob_from_labels"]
