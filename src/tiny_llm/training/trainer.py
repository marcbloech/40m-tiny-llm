"""Training loop with gradient accumulation and gradient clipping."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from tiny_llm.training.checkpoint import (
    load_checkpoint,
    prune_checkpoints,
    save_best_checkpoint,
    save_checkpoint,
)
from tiny_llm.training.optimizer import create_optimizer, create_scheduler
from tiny_llm.utils.logging import MetricsLogger, add_file_logging

logger = logging.getLogger(__name__)


class Trainer:
    """Handles the pre-training loop.

    Features (all toggleable via config):
        - Gradient accumulation  (grad_accum_steps > 1)
        - Gradient clipping      (gradient_clip_norm > 0)
        - Cosine LR schedule     (use_scheduler = true)
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        config: dict[str, Any],
        device: torch.device,
        val_dataset: Dataset | None = None,
        model_config: dict[str, Any] | None = None,
        resume_from: str | None = None,
    ) -> None:
        self.device = device
        self.config = config
        self.model_config = model_config or {}
        self._stats_dtype = (
            torch.float32 if self.device.type == "mps" else torch.float64
        )

        log_path = add_file_logging(Path(config["checkpoint_dir"]) / "training.log")
        logger.info("Training logs → %s", log_path)

        self.model = model.to(device)

        # ── torch.compile (optional) ──────────────────────────
        self.use_compile = config.get("use_compile", False)
        if self.use_compile and device.type == "cuda":
            logger.info("Compiling model with torch.compile (mode=reduce-overhead)")
            self.model = torch.compile(self.model, mode="reduce-overhead")
        elif self.use_compile:
            logger.warning(
                "use_compile=True but device=%s — torch.compile works best on CUDA, skipping",
                device.type,
            )

        # ── DataLoaders ───────────────────────────────────────
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config.get("num_workers", 4),
            pin_memory=config.get("pin_memory", True) and device.type == "cuda",
            drop_last=True,
        )
        self.val_loader = None
        if val_dataset is not None:
            default_eval_mult = 4 if device.type == "cuda" else 1
            eval_bs = (
                config.get("eval_batch_size")
                or config["batch_size"] * default_eval_mult
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=eval_bs,
                shuffle=False,
                num_workers=config.get("num_workers", 4),
                pin_memory=config.get("pin_memory", True) and device.type == "cuda",
                drop_last=False,
            )

        # ── Optimizer ─────────────────────────────────────────
        self.optimizer = create_optimizer(
            self.model,
            optimizer_name=config.get("optimizer", "adamw"),
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=config.get("betas", [0.9, 0.95]),
        )

        # ── LR Scheduler (toggleable) ────────────────────────
        self.use_scheduler = config.get("use_scheduler", True)
        if self.use_scheduler:
            self.scheduler = create_scheduler(
                self.optimizer,
                lr_schedule=config.get("lr_schedule", "cosine"),
                warmup_steps=config["warmup_steps"],
                max_steps=config["max_steps"],
                min_lr_ratio=config.get("min_lr_ratio", 0.1),
                wsd_decay_fraction=config.get("wsd_decay_fraction", 0.2),
            )
        else:
            self.scheduler = None

        # ── Training state ────────────────────────────────────
        self.start_step = 0
        self.tokens_seen = 0
        self.best_val_loss = float("inf")

        if resume_from is not None:
            self._resume(resume_from)

        self.metrics_logger = MetricsLogger(config=config)

        # ── Log training setup ────────────────────────────────
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        ctx_len = self.model_config.get("context_length", 1024)
        tokens_per_step = config["batch_size"] * config["grad_accum_steps"] * ctx_len
        features = []
        if config["grad_accum_steps"] > 1:
            features.append(f"grad_accum={config['grad_accum_steps']}")
        if config.get("gradient_clip_norm", 1.0) > 0:
            features.append(f"clip={config['gradient_clip_norm']}")
        if self.use_scheduler:
            features.append("cosine+warmup")
        else:
            features.append("constant_lr")
        if self.use_compile and device.type == "cuda":
            features.append("torch.compile")

        logger.info(
            "Training setup: %s total params (%s trainable), "
            "%s nominal tokens/step, effective_batch=%d seqs (%d batch × %d accum), "
            "device=%s, features=[%s]",
            f"{total_params:,}",
            f"{trainable_params:,}",
            f"{tokens_per_step:,}",
            config["batch_size"] * config["grad_accum_steps"],
            config["batch_size"],
            config["grad_accum_steps"],
            device,
            ", ".join(features) if features else "none",
        )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def _resume(self, path: str) -> None:
        logger.info("Resuming training from %s", path)
        meta = load_checkpoint(
            path,
            self.model,
            optimizer=self.optimizer,
            device=self.device,
        )
        self.start_step = meta["step"] + 1
        self.tokens_seen = meta.get("tokens_seen", 0)
        if meta.get("val_loss") is not None:
            self.best_val_loss = meta["val_loss"]

        if self.scheduler is not None:
            for _ in range(self.start_step):
                self.scheduler.step()

        logger.info(
            "Resumed from step %d (tokens_seen=%d, best_val_loss=%.4f)",
            self.start_step,
            self.tokens_seen,
            self.best_val_loss,
        )

    # ------------------------------------------------------------------
    # Infinite data iterator
    # ------------------------------------------------------------------

    def _get_infinite_loader(self):
        epoch = 0
        while True:
            for batch in self.train_loader:
                yield epoch, batch
            epoch += 1

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _assess(self) -> float:
        if self.val_loader is None:
            return float("nan")

        was_training = self.model.training
        self.model.train(False)
        loss_sum = torch.zeros((), device=self.device, dtype=self._stats_dtype)
        token_count = torch.zeros((), device=self.device, dtype=self._stats_dtype)

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits, _ = self.model(x)
            vocab_size = logits.size(-1)
            batch_loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            valid_tokens = (y != -100).sum()
            loss_sum += batch_loss.detach().to(self._stats_dtype)
            token_count += valid_tokens.detach().to(self._stats_dtype)

        if was_training:
            self.model.train(True)

        total_tokens = max(token_count.item(), 1.0)
        return loss_sum.item() / total_tokens

    # ------------------------------------------------------------------
    # Sample generation (qualitative monitoring)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_sample(self, prompt_text: str = "The meaning of life is") -> str:
        was_training = self.model.training
        self.model.train(False)
        try:
            from tiny_llm.tokenizer import decode_tokens, encode_text

            prompt_ids = encode_text(prompt_text)
            idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

            ctx_len = self.model_config.get("context_length", 1024)
            max_new = min(100, ctx_len - len(prompt_ids))

            generated = self.model.generate(
                idx, max_new_tokens=max_new, temperature=0.8, top_k=40
            )
            return decode_tokens(generated[0].tolist())
        except Exception as e:
            return f"[generation failed: {e}]"
        finally:
            if was_training:
                self.model.train(True)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        config = self.config
        max_steps = config["max_steps"]
        grad_accum_steps = config["grad_accum_steps"]
        gradient_clip_norm = config.get("gradient_clip_norm", 1.0)
        log_interval = config.get("log_interval", 10)
        eval_interval = config["eval_interval"]
        checkpoint_interval = config["checkpoint_interval"]
        sample_interval = config.get("sample_interval", 1000)
        checkpoint_dir = Path(config["checkpoint_dir"])
        max_checkpoints = config.get("max_checkpoints", 3)
        max_best = config.get("max_best_checkpoints", 3)
        ctx_len = self.model_config.get("context_length", 1024)
        nominal_tokens_per_step = config["batch_size"] * grad_accum_steps * ctx_len

        self.model.train(True)
        data_iter = self._get_infinite_loader()
        self.optimizer.zero_grad(set_to_none=True)

        # ── Model architecture summary ────────────────────────
        import io
        from tiny_llm.utils.model_summary import model_summary as _model_summary

        dummy_ids = torch.zeros(
            (1, ctx_len), dtype=torch.long, device=self.device
        )
        buf = io.StringIO()
        _model_summary(
            self.model,
            input_data=(dummy_ids,),
            max_depth=4,
            device=self.device,
            file=buf,
        )
        summary_str = buf.getvalue()
        print(summary_str)
        logger.info("Model architecture summary:\n%s", summary_str)

        logger.info(
            "Starting training from step %d to %d (nominal_tokens_per_step=%d)",
            self.start_step,
            max_steps,
            nominal_tokens_per_step,
        )

        step_t0 = time.time()

        for step in range(self.start_step, max_steps):
            epoch = 0
            window_loss_sum = torch.zeros(
                (), device=self.device, dtype=self._stats_dtype
            )
            window_tokens = torch.zeros(
                (), device=self.device, dtype=self._stats_dtype
            )

            for micro_step in range(grad_accum_steps):
                epoch, (x, y) = next(data_iter)
                x, y = x.to(self.device), y.to(self.device)

                logits, _ = self.model(x)
                vocab_size = logits.size(-1)
                loss_sum = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    y.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )

                valid_tokens = (y != -100).sum()
                loss_sum.backward()

                window_loss_sum += loss_sum.detach().to(self._stats_dtype)
                window_tokens += valid_tokens.detach().to(self._stats_dtype)

            total_tokens = max(window_tokens.item(), 1.0)
            grad_scale = 1.0 / total_tokens
            accumulated_loss = window_loss_sum.item() / total_tokens

            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.mul_(grad_scale)

            grad_norm = (
                clip_grad_norm_(self.model.parameters(), max_norm=gradient_clip_norm)
                if gradient_clip_norm > 0
                else None
            )

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            self.optimizer.zero_grad(set_to_none=True)
            self.tokens_seen += int(total_tokens)

            # ── Periodic Actions ─────────────────────────────

            pct = (step - self.start_step + 1) / (max_steps - self.start_step) * 100
            print(
                f"  step {step}/{max_steps} "
                f"({pct:.0f}%) | loss={accumulated_loss:.4f}",
                flush=True,
            )

            if step % log_interval == 0:
                step_dt = time.time() - step_t0
                tokens_per_sec = total_tokens / max(step_dt, 1e-6)
                current_lr = (
                    self.scheduler.get_last_lr()[0]
                    if self.scheduler is not None
                    else config["learning_rate"]
                )

                metrics = {
                    "train/loss": round(accumulated_loss, 4),
                    "train/perplexity": round(math.exp(accumulated_loss), 2)
                    if accumulated_loss < 20
                    else float("inf"),
                    "train/lr": current_lr,
                    "train/tokens_seen": self.tokens_seen,
                    "train/tokens_per_sec": round(tokens_per_sec, 0),
                }
                if grad_norm is not None:
                    metrics["train/grad_norm"] = (
                        round(grad_norm.item(), 4)
                        if isinstance(grad_norm, torch.Tensor)
                        else round(grad_norm, 4)
                    )

                self.metrics_logger.log(metrics, step=step, epoch=epoch)

            step_t0 = time.time()

            if step > 0 and step % eval_interval == 0:
                val_loss = self._assess()

                val_ppl = (
                    math.exp(val_loss) if not math.isnan(val_loss) else float("nan")
                )
                val_metrics = {
                    "val/loss": round(val_loss, 4),
                    "val/perplexity": round(val_ppl, 2),
                }
                self.metrics_logger.log(val_metrics, step=step, epoch=epoch)
                logger.info(
                    "Step %d | val_loss=%.4f | val_ppl=%.2f",
                    step,
                    val_loss,
                    val_ppl,
                )

                saved_path = save_best_checkpoint(
                    self.model,
                    step=step,
                    config=self.config,
                    checkpoint_dir=checkpoint_dir,
                    val_loss=val_loss,
                    tokens_seen=self.tokens_seen,
                    model_config=self.model_config,
                    max_best=max_best,
                )
                if saved_path is not None:
                    logger.info(
                        "Best checkpoint saved: %s (val_loss=%.4f)",
                        saved_path.name,
                        val_loss,
                    )
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

            if step > 0 and step % checkpoint_interval == 0:
                save_checkpoint(
                    self.model,
                    optimizer=self.optimizer,
                    step=step,
                    config=self.config,
                    path=checkpoint_dir / f"step_{step:06d}.pt",
                    val_loss=self.best_val_loss,
                    tokens_seen=self.tokens_seen,
                    model_config=self.model_config,
                )
                prune_checkpoints(checkpoint_dir, max_keep=max_checkpoints)

            if sample_interval > 0 and step > 0 and step % sample_interval == 0:
                sample = self._generate_sample()
                logger.info("Step %d sample:\n%s", step, sample)

            quick_plot_path = self.metrics_logger.maybe_save_quick_plot(step)
            if quick_plot_path is not None:
                logger.info("Quick training plot → %s", quick_plot_path)

        # ── Training complete ─────────────────────────────────
        logger.info(
            "Training complete. %d steps, %d tokens seen.",
            max_steps,
            self.tokens_seen,
        )

        val_loss = self._assess()
        if not math.isnan(val_loss):
            logger.info(
                "Final val_loss=%.4f, val_ppl=%.2f",
                val_loss,
                math.exp(val_loss),
            )

        save_checkpoint(
            self.model,
            optimizer=self.optimizer,
            step=max_steps,
            config=self.config,
            path=checkpoint_dir / f"step_{max_steps:06d}.pt",
            val_loss=val_loss if not math.isnan(val_loss) else None,
            tokens_seen=self.tokens_seen,
            model_config=self.model_config,
        )

        self.metrics_logger.finish()
