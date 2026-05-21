"""Supervised fine-tuning trainer.

Epoch-based loop over a tokenised SFT dataset with response-only loss.
Mirrors the pre-training :class:`tiny_llm.training.trainer.Trainer` for the
core optimiser plumbing (grad accumulation, clipping, cosine schedule)
but replaces the pre-training dataloader, validation routine and metrics
panels with SFT-specific equivalents.

Monitoring signals written every ``eval_interval`` / ``pretrain_eval_interval``
steps:

- ``train/loss``, ``val/loss``, ``val/perplexity``       — overfitting
- ``pretrain/val_loss``, ``pretrain/val_perplexity``     — catastrophic forgetting
"""

from __future__ import annotations

import json
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

from tiny_llm.posttraining.sft_dataset import (
    EOS_TOKEN_ID,
    SFTDataset,
    format_alpaca_prompt,
    sft_collate_fn,
)
from tiny_llm.posttraining.sft_metrics import SFTMetricsLogger
from tiny_llm.tokenizer import decode_tokens, encode_text
from tiny_llm.training.checkpoint import (
    prune_checkpoints,
    save_best_checkpoint,
    save_checkpoint,
)
from tiny_llm.training.optimizer import create_optimizer, create_scheduler
from tiny_llm.utils.logging import add_file_logging

logger = logging.getLogger(__name__)


_DEFAULT_SAMPLE_PROMPTS: list[dict[str, str]] = [
    {
        "name": "mc_science",
        "instruction": (
            "Answer the multiple-choice question with only the correct letter.\n\n"
            "Which gas do plants mainly absorb from the air during photosynthesis?\n"
            "A) Oxygen\n"
            "B) Carbon dioxide\n"
            "C) Nitrogen\n"
            "D) Helium"
        ),
        "input": "",
    },
    {
        "name": "concise_explanation",
        "instruction": "Explain why the sky appears blue in two short sentences.",
        "input": "",
    },
    {
        "name": "classification",
        "instruction": (
            "Classify the sentiment of the review as Positive, Neutral, or Negative. "
            "Answer with one word."
        ),
        "input": "The service was slow, but the food was excellent.",
    },
]


class SFTTrainer:
    """Single-device SFT trainer (CPU / MPS / CUDA)."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: SFTDataset,
        val_dataset: SFTDataset | None,
        config: dict[str, Any],
        device: torch.device,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        self.device = device
        self.config = config
        self.model_config = model_config or {}

        checkpoint_dir = Path(config["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        add_file_logging(checkpoint_dir / "sft.log")

        self.model = model.to(device)

        # ── DataLoaders ──────────────────────────────────────────────
        collate = partial(sft_collate_fn, pad_token_id=EOS_TOKEN_ID)
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
        self.val_source_loaders: dict[str, DataLoader] = {}
        if val_dataset is not None and len(val_dataset) > 0:
            eval_bs = config.get("eval_batch_size") or config["batch_size"]
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=eval_bs,
                shuffle=False,
                num_workers=config.get("num_workers", 2),
                pin_memory=config.get("pin_memory", True) and device.type == "cuda",
                drop_last=False,
                collate_fn=collate,
            )
            by_source: dict[str, list] = {}
            for ex in val_dataset.examples:
                source = str(getattr(ex, "source", "unknown") or "unknown")
                by_source.setdefault(source, []).append(ex)
            for source, examples in sorted(by_source.items()):
                self.val_source_loaders[source] = DataLoader(
                    SFTDataset(examples),
                    batch_size=eval_bs,
                    shuffle=False,
                    num_workers=config.get("num_workers", 2),
                    pin_memory=config.get("pin_memory", True) and device.type == "cuda",
                    drop_last=False,
                    collate_fn=collate,
                )

        # ── Derived step counts ──────────────────────────────────────
        grad_accum = int(config.get("grad_accum_steps", 1))
        steps_per_epoch = max(1, len(self.train_loader) // max(1, grad_accum))
        self.max_steps = steps_per_epoch * int(config["epochs"])
        self.warmup_steps = max(1, int(config.get("warmup_ratio", 0.0) * self.max_steps))

        # ── Optimiser + cosine schedule ──────────────────────────────
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


        # ── Catastrophic-forgetting probe setup ──────────────────────
        self.pretrain_val_tokens: np.memmap | None = None
        pv_path = config.get("pretrain_val_path")
        if pv_path:
            p = Path(pv_path)
            if p.exists():
                self.pretrain_val_tokens = np.memmap(str(p), dtype=np.uint16, mode="r")
                logger.info(
                    "[sft] Pretrain-val probe enabled: %s (%d tokens)",
                    p, len(self.pretrain_val_tokens),
                )
            else:
                logger.warning(
                    "[sft] pretrain_val_path=%s not found — forgetting probe disabled",
                    pv_path,
                )

        self.best_val_loss = float("inf")
        self.metrics_logger = SFTMetricsLogger(config=config)
        self.sample_prompts = self._normalise_sample_prompts(
            config.get("sample_prompts")
        )
        self.sample_generations_path = checkpoint_dir / "sample_generations.jsonl"

        # Log setup summary
        total_params = sum(p.numel() for p in self.model.parameters())
        effective_batch = config["batch_size"] * grad_accum
        logger.info(
            "[sft] Training setup: %s params, %d train / %d val examples, "
            "%d epochs × %d steps/epoch = %d optim steps, "
            "effective_batch=%d, warmup=%d, lr=%.2e, device=%s%s",
            f"{total_params:,}",
            len(train_dataset),
            len(val_dataset) if val_dataset else 0,
            config["epochs"], steps_per_epoch, self.max_steps,
            effective_batch, self.warmup_steps, config["learning_rate"],
            device,
        )
        self._log_truncation_summary("train", train_dataset.examples)
        if val_dataset is not None:
            self._log_truncation_summary("val", val_dataset.examples)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_sample_prompts(raw_prompts: Any) -> list[dict[str, str]]:
        """Return sample prompts as name/instruction/input dicts."""
        if not raw_prompts:
            return list(_DEFAULT_SAMPLE_PROMPTS)
        if not isinstance(raw_prompts, list):
            raise ValueError("sample_prompts must be a list of strings or tables")

        prompts: list[dict[str, str]] = []
        for idx, prompt in enumerate(raw_prompts, start=1):
            if isinstance(prompt, str):
                instruction = prompt.strip()
                if instruction:
                    prompts.append(
                        {
                            "name": f"sample_{idx}",
                            "instruction": instruction,
                            "input": "",
                        }
                    )
                continue
            if isinstance(prompt, dict):
                instruction = str(prompt.get("instruction", "")).strip()
                if not instruction:
                    continue
                name = str(prompt.get("name") or f"sample_{idx}").strip()
                prompts.append(
                    {
                        "name": name or f"sample_{idx}",
                        "instruction": instruction,
                        "input": str(prompt.get("input", "")).strip(),
                    }
                )
                continue
            raise ValueError("sample_prompts entries must be strings or tables")
        return prompts

    @torch.no_grad()
    def _log_instruction_samples(self, *, step: int, epoch: int | None) -> None:
        """Generate fixed instruction-following samples for qualitative tracking."""
        if not self.sample_prompts:
            return

        max_new = int(self.config.get("sample_max_tokens", 80))
        if max_new <= 0:
            return
        temperature = float(self.config.get("sample_temperature", 0.0))
        top_k_raw = self.config.get("sample_top_k")
        top_k = int(top_k_raw) if top_k_raw is not None else None
        repetition_penalty = float(self.config.get("sample_repetition_penalty", 1.1))

        was_training = self.model.training
        self.model.eval()
        records: list[dict[str, Any]] = []

        for sample in self.sample_prompts:
            prompt_text = format_alpaca_prompt(
                sample["instruction"],
                sample.get("input", ""),
            )
            prompt_ids = encode_text(prompt_text)
            idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            output_ids = self.model.generate(
                idx,
                max_new_tokens=max_new,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                eos_token_id=EOS_TOKEN_ID,
            )
            new_ids = output_ids[0].tolist()[len(prompt_ids):]
            if new_ids and new_ids[-1] == EOS_TOKEN_ID:
                new_ids = new_ids[:-1]
            response = decode_tokens(new_ids).strip()
            records.append(
                {
                    "step": step,
                    "epoch": epoch,
                    "name": sample["name"],
                    "instruction": sample["instruction"],
                    "input": sample.get("input", ""),
                    "response": response,
                }
            )

        with open(self.sample_generations_path, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        for record in records:
            logger.info(
                "[sft sample] step=%d name=%s prompt=%r response=%r",
                step,
                record["name"],
                record["instruction"],
                record["response"],
            )
            input_block = (
                f"\ninput:\n{record['input']}" if record.get("input") else ""
            )
            response = record["response"] or "<empty>"
            print(
                f"\n[sft sample] step={step} name={record['name']}\n"
                f"instruction:\n{record['instruction']}"
                f"{input_block}\n"
                f"response:\n{response}\n",
                flush=True,
            )

        if was_training:
            self.model.train(True)

    def _metric_source_name(self, source: str) -> str:
        return source.replace("/", "_")

    @torch.no_grad()
    def _assess_sft_loader(self, loader: DataLoader | None) -> float:
        """Mean cross-entropy over one SFT loader (response tokens only)."""
        if loader is None:
            return float("nan")
        was_training = self.model.training
        self.model.eval()
        loss_sum = 0.0
        token_count = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits, _ = self.model(x)
            vocab = logits.size(-1)
            batch_loss = F.cross_entropy(
                logits.view(-1, vocab),
                y.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            valid = (y != -100).sum().item()
            loss_sum += float(batch_loss.item())
            token_count += int(valid)
        if was_training:
            self.model.train(True)
        if token_count == 0:
            return float("nan")
        return loss_sum / token_count

    def _assess_sft_val(self) -> float:
        """Mean cross-entropy over the full SFT val split."""
        return self._assess_sft_loader(self.val_loader)

    def _collect_sft_val_metrics(self) -> tuple[float, dict[str, float]]:
        val_loss = self._assess_sft_val()
        metrics: dict[str, float] = {}
        if not math.isnan(val_loss):
            metrics["val/loss"] = round(val_loss, 4)
            metrics["val/perplexity"] = round(math.exp(min(val_loss, 20)), 2)

        for source, loader in self.val_source_loaders.items():
            source_loss = self._assess_sft_loader(loader)
            if math.isnan(source_loss):
                continue
            metric_source = self._metric_source_name(source)
            metrics[f"val/{metric_source}_loss"] = round(source_loss, 4)
            metrics[f"val/{metric_source}_perplexity"] = round(
                math.exp(min(source_loss, 20)),
                2,
            )

        return val_loss, metrics

    def _log_sft_validation(
        self,
        *,
        step: int,
        epoch: int | None,
        checkpoint_dir: Path | None = None,
        max_best: int | None = None,
        only_if_improved: bool = False,
    ) -> float:
        val_loss, metrics = self._collect_sft_val_metrics()
        if metrics:
            self.metrics_logger.log(metrics, step=step, epoch=epoch)

        if (
            checkpoint_dir is not None
            and max_best is not None
            and not math.isnan(val_loss)
            and (not only_if_improved or val_loss < self.best_val_loss)
        ):
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
                    "[sft] Best checkpoint saved: %s (val_loss=%.4f)",
                    saved.name,
                    val_loss,
                )
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

        return val_loss

    @torch.no_grad()
    def _assess_pretrain_val(self) -> float:
        """Cross-entropy on random pre-train val chunks (forgetting probe)."""
        if self.pretrain_val_tokens is None:
            return float("nan")

        ctx_len = int(self.model_config.get("context_length", 1024))
        n_batches = int(self.config.get("pretrain_val_batches", 40))
        batch_size = int(self.config.get("eval_batch_size") or self.config["batch_size"])
        tokens = self.pretrain_val_tokens

        # Pre-compute possible window starts, deterministic across calls.
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
                [np.asarray(tokens[s : s + ctx_len + 1], dtype=np.int64) for s in batch_starts]
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

    def _log_pretrain_validation(self, *, step: int, epoch: int | None) -> float:
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
        return pre_loss

    def _log_truncation_summary(self, split: str, examples: list) -> None:
        by_source: dict[str, list] = {}
        for ex in examples:
            source = str(getattr(ex, "source", "unknown") or "unknown")
            by_source.setdefault(source, []).append(ex)
        for source, source_examples in sorted(by_source.items()):
            truncated = sum(
                1 for ex in source_examples if bool(getattr(ex, "truncated", False))
            )
            logger.info(
                "[sft] %s/%s truncation: %d/%d examples (%.2f%%)",
                split,
                source,
                truncated,
                len(source_examples),
                truncated / max(len(source_examples), 1) * 100,
            )

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
        sample_interval = int(cfg.get("sample_interval", 0))
        checkpoint_interval = int(cfg.get("checkpoint_interval", 500))
        checkpoint_dir = Path(cfg["checkpoint_dir"])
        max_checkpoints = int(cfg.get("max_checkpoints", 3))
        max_best = int(cfg.get("max_best_checkpoints", 3))

        self.model.train(True)
        self.optimizer.zero_grad(set_to_none=True)

        step = 0
        step_t0 = time.time()

        # Baseline probes before any SFT optimiser step. These make SFT gain
        # and pretrain-val drift measurable against the loaded base checkpoint.
        self._log_sft_validation(step=0, epoch=0)
        self._log_pretrain_validation(step=0, epoch=0)
        if sample_interval > 0:
            self._log_instruction_samples(step=0, epoch=0)

        for epoch in range(int(cfg["epochs"])):
            logger.info("[sft] Epoch %d/%d starting", epoch + 1, int(cfg["epochs"]))

            window_loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            window_tokens = torch.zeros((), device=self.device, dtype=torch.float32)
            micro_count = 0

            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                logits, _ = self.model(x)
                vocab = logits.size(-1)
                loss_sum = F.cross_entropy(
                    logits.view(-1, vocab),
                    y.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )

                valid = (y != -100).sum()
                loss_sum.backward()

                window_loss_sum += loss_sum.detach().to(torch.float32)
                window_tokens += valid.detach().to(torch.float32)
                micro_count += 1

                if micro_count < grad_accum_steps:
                    continue

                # ── Optimiser step ──────────────────────────────
                total_tokens = max(window_tokens.item(), 1.0)
                # Normalise accumulated gradients by number of valid tokens.
                # The scale factor is (1 / total_tokens) since each micro-batch
                # contributed a *summed* loss.
                grad_scale = 1.0 / total_tokens
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(grad_scale)

                grad_norm = (
                    clip_grad_norm_(self.model.parameters(), max_norm=gradient_clip_norm)
                    if gradient_clip_norm > 0 else None
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                accumulated_loss = window_loss_sum.item() / total_tokens

                # Reset accumulation window
                window_loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
                window_tokens = torch.zeros((), device=self.device, dtype=torch.float32)
                micro_count = 0
                step += 1

                # ── Metrics ─────────────────────────────────────
                if step % log_interval == 0 or step == 1:
                    step_dt = time.time() - step_t0
                    tokens_per_sec = total_tokens / max(step_dt, 1e-6)
                    current_lr = self.scheduler.get_last_lr()[0]
                    metrics = {
                        "train/loss": round(accumulated_loss, 4),
                        "train/lr": current_lr,
                        "train/tokens_per_sec": round(tokens_per_sec, 0),
                    }
                    if grad_norm is not None:
                        gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                        metrics["train/grad_norm"] = round(float(gn), 4)
                    self.metrics_logger.log(metrics, step=step, epoch=epoch)
                step_t0 = time.time()

                # Progress heartbeat so long epochs don't look stuck.
                pct = step / self.max_steps * 100
                print(
                    f"  sft step {step}/{self.max_steps} "
                    f"({pct:.0f}%) | loss={accumulated_loss:.4f}",
                    flush=True,
                )

                # ── SFT validation ─────────────────────────────
                if step > 0 and step % eval_interval == 0:
                    self._log_sft_validation(
                        step=step,
                        epoch=epoch,
                        checkpoint_dir=checkpoint_dir,
                        max_best=max_best,
                    )

                # ── Catastrophic-forgetting probe ──────────────
                if (
                    self.pretrain_val_tokens is not None
                    and step > 0
                    and step % pretrain_eval_interval == 0
                ):
                    self._log_pretrain_validation(step=step, epoch=epoch)

                # ── Qualitative instruction-following samples ───
                if sample_interval > 0 and step > 0 and step % sample_interval == 0:
                    self._log_instruction_samples(step=step, epoch=epoch)

                # ── Periodic full checkpoint ───────────────────
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
                    prune_checkpoints(checkpoint_dir, max_keep=max_checkpoints)

                quick_plot = self.metrics_logger.maybe_save_quick_plot(step)
                if quick_plot is not None:
                    logger.info("[sft] Quick plot → %s", quick_plot)

            # Flush leftover micro-batches at the end of an epoch (partial window
            # discarded: the DataLoader already uses drop_last=True for full
            # micro-batches; any residual is smaller than one micro-batch).

        # ── Final validation + checkpoint ────────────────────────
        final_val = self._log_sft_validation(
            step=step,
            epoch=max(0, int(cfg["epochs"]) - 1),
            checkpoint_dir=checkpoint_dir,
            max_best=max_best,
            only_if_improved=True,
        )
        self._log_pretrain_validation(
            step=step,
            epoch=max(0, int(cfg["epochs"]) - 1),
        )
        if not math.isnan(final_val):
            logger.info(
                "[sft] Final val_loss=%.4f val_ppl=%.2f",
                final_val, math.exp(min(final_val, 20)),
            )

        save_checkpoint(
            self.model,
            optimizer=self.optimizer,
            step=step,
            config=self.config,
            path=checkpoint_dir / f"step_{step:06d}.pt",
            val_loss=final_val if not math.isnan(final_val) else None,
            tokens_seen=0,
            model_config=self.model_config,
        )

        self.metrics_logger.finish()
        logger.info("[sft] Training complete.")


__all__ = ["SFTTrainer"]
