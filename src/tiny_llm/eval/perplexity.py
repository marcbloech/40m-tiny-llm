"""Perplexity evaluation on held-out test splits."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tiny_llm.model.transformer import GPTModel
from tiny_llm.tokenizer import encode_text

logger = logging.getLogger(__name__)


def _load_token_ids(data_path: str | Path) -> torch.Tensor:
    """Load token IDs from a .bin memmap file or a HuggingFace dataset name.

    Supported inputs:
      - Path ending in .bin: reads uint16 memmap (existing tokenized format)
      - "wikitext-103": downloads Salesforce/wikitext wikitext-103-raw-v1 test split
    """
    data_path = str(data_path)

    if data_path.endswith(".bin"):
        data = np.memmap(data_path, dtype=np.uint16, mode="r")
        return torch.from_numpy(data.astype(np.int64))

    if "wikitext" in data_path.lower():
        from datasets import load_dataset

        logger.info("Downloading wikitext-103 test split from HuggingFace...")
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
        # Concatenate all text and tokenize
        all_text = "\n\n".join(row["text"] for row in ds if row["text"].strip())
        token_ids = encode_text(all_text)
        return torch.tensor(token_ids, dtype=torch.long)

    raise ValueError(
        f"Unsupported data_path: {data_path}. Use a .bin file or 'wikitext-103'."
    )


def evaluate_perplexity(
    model: GPTModel,
    data_path: str | Path,
    device: torch.device,
    context_length: int = 1024,
    stride: int | None = None,
    max_tokens: int | None = None,
) -> float:
    """Compute perplexity on a test split.

    Uses a sliding window approach with the given stride (defaults to
    context_length // 2 for a balance of accuracy and speed).

    Args:
        model: Trained GPT model in eval mode.
        data_path: Path to tokenized test data (.bin memmap) or dataset name.
        device: Evaluation device.
        context_length: Window size in tokens.
        stride: Step size for sliding window (default: context_length // 2).
        max_tokens: Optional cap on the number of tokens to read from the
            file. Perplexity converges quickly with N, so 2–5 M tokens is
            ample for comparative ranking; the full file (often 100 M+
            tokens) is only worth it for final fact-sheet numbers.

    Returns:
        Perplexity (float).
    """
    if stride is None:
        stride = context_length // 2

    token_ids = _load_token_ids(data_path)
    if max_tokens is not None and len(token_ids) > max_tokens:
        token_ids = token_ids[:max_tokens]
    n_tokens = len(token_ids)
    logger.info(
        "Evaluating perplexity on %d tokens (ctx=%d, stride=%d)",
        n_tokens,
        context_length,
        stride,
    )

    total_loss = 0.0
    total_count = 0

    model.eval()
    with torch.no_grad():
        for begin in range(0, n_tokens - 1, stride):
            end = min(begin + context_length, n_tokens - 1)
            input_ids = token_ids[begin:end].unsqueeze(0).to(device)  # (1, S)
            target_ids = (
                token_ids[begin + 1 : end + 1].unsqueeze(0).to(device)
            )  # (1, S)

            logits, _ = model(input_ids)  # (1, S, V)

            # Only score tokens that haven't been scored in a previous window
            # For overlapping windows, skip tokens before the stride boundary
            if begin > 0:
                score_start = stride
            else:
                score_start = 0

            seq_len = logits.size(1)
            if score_start >= seq_len:
                continue

            logits_slice = logits[0, score_start:]  # (S', V)
            targets_slice = target_ids[0, score_start:]  # (S',)

            loss = F.cross_entropy(logits_slice, targets_slice, reduction="sum")
            n = targets_slice.numel()
            total_loss += loss.item()
            total_count += n

            if total_count % (100_000) < stride:
                logger.info("  ... processed %d / %d tokens", total_count, n_tokens)

    avg_loss = total_loss / total_count
    ppl = float(np.exp(avg_loss))
    logger.info(
        "Perplexity: %.2f (avg CE loss: %.4f, scored %d tokens)",
        ppl,
        avg_loss,
        total_count,
    )
    return ppl
