"""DPO preference dataset utilities.

Implements dataset loading, tokenisation, and collation for Direct Preference
Optimization.  Each example is a ``(prompt, chosen, rejected)`` triple.  The
collate function produces the dict format from EX08::

    {"chosen_input_ids", "chosen_labels", "chosen_padding_mask",
     "rejected_input_ids", "rejected_labels", "rejected_padding_mask"}

Prompt tokens carry ``-100`` in labels so gradients only flow through response
tokens, mirroring the SFT masking convention.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch.utils.data import Dataset

from tiny_llm.posttraining.sft_dataset import (
    EOS_TOKEN_ID,
    format_alpaca_prompt,
)
from tiny_llm.tokenizer import get_gpt2_tokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokeniser access (GPT-2 via transformers -- same as pre-training & SFT)
# ---------------------------------------------------------------------------


def _get_tokenizer():
    return get_gpt2_tokenizer()


def _encode(enc, text: str) -> list[int]:
    return enc.encode(text, add_special_tokens=False)


# ---------------------------------------------------------------------------
# Tokenisation of a single DPO example
# ---------------------------------------------------------------------------


@dataclass
class DPOExample:
    """A tokenised DPO preference pair.

    Attributes:
        prompt_ids: Token IDs for the shared prompt.
        chosen_ids: Token IDs for the chosen response (excluding prompt, EOS appended).
        rejected_ids: Token IDs for the rejected response (excluding prompt, EOS appended).
    """

    prompt_ids: list[int]
    chosen_ids: list[int]
    rejected_ids: list[int]


def tokenise_dpo_pair(
    enc,
    instruction: str,
    input_text: str,
    chosen: str,
    rejected: str,
    max_seq_len: int,
) -> DPOExample | None:
    """Tokenise one DPO preference pair.

    Returns ``None`` if the prompt alone exceeds ``max_seq_len``, or if either
    response is empty.  Responses are truncated to fit within ``max_seq_len``.
    """
    chosen = (chosen or "").strip()
    rejected = (rejected or "").strip()
    if not chosen or not rejected:
        return None

    prompt_str = format_alpaca_prompt(instruction, input_text)
    prompt_ids = _encode(enc, prompt_str)

    if len(prompt_ids) >= max_seq_len:
        return None

    budget = max_seq_len - len(prompt_ids)

    chosen_ids = _encode(enc, chosen)
    if len(chosen_ids) > budget - 1:
        chosen_ids = chosen_ids[: budget - 1]
    chosen_ids.append(EOS_TOKEN_ID)

    rejected_ids = _encode(enc, rejected)
    if len(rejected_ids) > budget - 1:
        rejected_ids = rejected_ids[: budget - 1]
    rejected_ids.append(EOS_TOKEN_ID)

    return DPOExample(
        prompt_ids=prompt_ids,
        chosen_ids=chosen_ids,
        rejected_ids=rejected_ids,
    )


# ---------------------------------------------------------------------------
# Dataset loaders -- registry
# ---------------------------------------------------------------------------


def _load_orca_dpo(
    max_samples: int | None, seed: int
) -> Iterable[dict[str, str]]:
    """Yield ``{instruction, input, chosen, rejected}`` from Orca DPO pairs."""
    from datasets import load_dataset

    logger.info(
        "Downloading argilla/distilabel-intel-orca-dpo-pairs from HuggingFace..."
    )
    ds = load_dataset(
        "argilla/distilabel-intel-orca-dpo-pairs", split="train"
    )
    if max_samples is not None and max_samples > 0 and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))
        logger.info("Orca DPO: capped to %d samples (max_samples)", max_samples)

    kept, skipped = 0, 0
    for row in ds:
        status = str(row.get("status", "")).strip().lower()
        if status == "tie":
            skipped += 1
            continue

        instruction = str(row.get("input", "")).strip()
        chosen = str(row.get("chosen", "")).strip()
        rejected = str(row.get("rejected", "")).strip()

        if not instruction or not chosen or not rejected:
            skipped += 1
            continue

        kept += 1
        yield {
            "instruction": instruction,
            "input": "",
            "chosen": chosen,
            "rejected": rejected,
        }

    if skipped:
        logger.info(
            "Orca DPO: kept %d, skipped %d (ties / empty)", kept, skipped
        )


def _load_hh_rlhf(
    max_samples: int | None, seed: int
) -> Iterable[dict[str, str]]:
    """Yield preference pairs from Anthropic/hh-rlhf."""
    from datasets import load_dataset

    logger.info("Downloading Anthropic/hh-rlhf from HuggingFace...")
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    if max_samples is not None and max_samples > 0 and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))
        logger.info("hh-rlhf: capped to %d samples (max_samples)", max_samples)

    for row in ds:
        chosen_text = str(row.get("chosen", "")).strip()
        rejected_text = str(row.get("rejected", "")).strip()
        if not chosen_text or not rejected_text:
            continue

        def _extract_last_turn(text: str) -> tuple[str, str]:
            parts = text.split("\n\nHuman: ")
            if len(parts) < 2:
                return text, ""
            last_exchange = parts[-1]
            if "\n\nAssistant: " in last_exchange:
                human_part, assistant_part = last_exchange.split(
                    "\n\nAssistant: ", 1
                )
                return human_part.strip(), assistant_part.strip()
            return last_exchange.strip(), ""

        instruction, chosen = _extract_last_turn(chosen_text)
        _, rejected = _extract_last_turn(rejected_text)

        if not instruction or not chosen or not rejected:
            continue

        yield {
            "instruction": instruction,
            "input": "",
            "chosen": chosen,
            "rejected": rejected,
        }


DPO_DATASET_LOADERS: dict[
    str, Callable[[int | None, int], Iterable[dict[str, str]]]
] = {
    "orca_dpo": _load_orca_dpo,
    "hh_rlhf": _load_hh_rlhf,
}


# ---------------------------------------------------------------------------
# Build + cache the tokenised DPO dataset
# ---------------------------------------------------------------------------


def _cache_paths(cache_dir: Path, dataset_key: str) -> tuple[Path, Path]:
    base = cache_dir / dataset_key
    return base / "train.pt", base / "val.pt"


def prepare_dpo_dataset(
    datasets: list[str],
    cache_dir: str | Path,
    *,
    max_seq_len: int = 1024,
    val_fraction: float = 0.05,
    max_samples: int | None = None,
    seed: int = 42,
    force: bool = False,
) -> tuple[list[DPOExample], list[DPOExample]]:
    """Prepare (or load cached) tokenised DPO train/val splits."""
    cache_dir = Path(cache_dir)
    dataset_key = (
        f"{'+'.join(sorted(datasets))}"
        f"_L{max_seq_len}"
        f"_N{max_samples if max_samples else 'all'}"
    )
    train_path, val_path = _cache_paths(cache_dir, dataset_key)

    if not force and train_path.exists() and val_path.exists():
        logger.info(
            "[dpo] Using cached tokenised DPO data at %s", train_path.parent
        )
        train = [
            DPOExample(**d) for d in torch.load(train_path, weights_only=False)
        ]
        val = [
            DPOExample(**d) for d in torch.load(val_path, weights_only=False)
        ]
        logger.info(
            "[dpo] Loaded %d train + %d val examples from cache",
            len(train),
            len(val),
        )
        return train, val

    enc = _get_tokenizer()
    all_examples: list[DPOExample] = []
    dropped = 0

    for name in datasets:
        if name not in DPO_DATASET_LOADERS:
            raise ValueError(
                f"Unknown DPO dataset: {name!r}. "
                f"Registered: {sorted(DPO_DATASET_LOADERS)}"
            )
        logger.info("[dpo] Tokenising %s ...", name)
        for row in DPO_DATASET_LOADERS[name](max_samples, seed):
            ex = tokenise_dpo_pair(
                enc,
                row["instruction"],
                row.get("input", ""),
                row["chosen"],
                row["rejected"],
                max_seq_len=max_seq_len,
            )
            if ex is None:
                dropped += 1
                continue
            all_examples.append(ex)

    if dropped:
        logger.warning(
            "[dpo] Dropped %d examples (too long or empty responses)",
            dropped,
        )

    rng = random.Random(seed)
    rng.shuffle(all_examples)
    n_val = max(1, int(len(all_examples) * val_fraction))
    val = all_examples[:n_val]
    train = all_examples[n_val:]
    logger.info(
        "[dpo] Built %d train + %d val examples (val_fraction=%.2f%%)",
        len(train),
        len(val),
        val_fraction * 100,
    )

    train_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save([ex.__dict__ for ex in train], train_path)
    torch.save([ex.__dict__ for ex in val], val_path)
    logger.info("[dpo] Cached tokenised data to %s", train_path.parent)

    return train, val


# ---------------------------------------------------------------------------
# torch Dataset + collator
# ---------------------------------------------------------------------------


class DPODataset(Dataset):
    """In-memory dataset of tokenised DPO preference pairs."""

    def __init__(self, examples: list[DPOExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> DPOExample:
        return self.examples[idx]


def dpo_collate_fn(
    batch: list[DPOExample],
    pad_token_id: int = EOS_TOKEN_ID,
) -> dict[str, torch.Tensor]:
    """Collate DPO examples into padded tensors (EX08 dict format).

    Returns a dict with keys matching the EX08 notebook convention:
    ``chosen_input_ids``, ``chosen_labels``, ``chosen_padding_mask``,
    ``rejected_input_ids``, ``rejected_labels``, ``rejected_padding_mask``.

    Labels have prompt positions set to ``-100`` and pad positions set to ``-100``.
    """
    B = len(batch)

    def _build_tensors(
        examples: list[DPOExample], side: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seqs: list[list[int]] = []
        prompt_lens: list[int] = []
        for ex in examples:
            prompt = ex.prompt_ids
            response = ex.chosen_ids if side == "chosen" else ex.rejected_ids
            seqs.append(prompt + response)
            prompt_lens.append(len(prompt))

        max_len = max(len(s) for s in seqs)

        input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
        labels = torch.full((B, max_len), -100, dtype=torch.long)
        padding_mask = torch.zeros((B, max_len), dtype=torch.bool)

        for i, (seq, k) in enumerate(zip(seqs, prompt_lens)):
            L = len(seq)
            input_ids[i, :L] = torch.tensor(seq, dtype=torch.long)
            padding_mask[i, :L] = True

            lbl = torch.tensor(seq, dtype=torch.long)
            lbl[:k] = -100
            labels[i, :L] = lbl

        return input_ids, labels, padding_mask

    c_ids, c_labels, c_mask = _build_tensors(batch, "chosen")
    r_ids, r_labels, r_mask = _build_tensors(batch, "rejected")

    return {
        "chosen_input_ids": c_ids,
        "chosen_labels": c_labels,
        "chosen_padding_mask": c_mask,
        "rejected_input_ids": r_ids,
        "rejected_labels": r_labels,
        "rejected_padding_mask": r_mask,
    }


__all__ = [
    "DPO_DATASET_LOADERS",
    "DPODataset",
    "DPOExample",
    "EOS_TOKEN_ID",
    "dpo_collate_fn",
    "prepare_dpo_dataset",
    "tokenise_dpo_pair",
]
