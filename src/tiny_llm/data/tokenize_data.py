"""Tokenize preprocessed text into memory-mapped binary files."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk
from transformers import GPT2TokenizerFast

from tiny_llm.tokenizer import get_gpt2_tokenizer

logger = logging.getLogger(__name__)

_UINT16_MAX = np.iinfo(np.uint16).max


def _encode_document(text: str, tokenizer: GPT2TokenizerFast) -> np.ndarray:
    """Tokenize a single document, append EOS, return uint16 array.

    Raises ValueError if any token ID exceeds uint16 range.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id; GPT-2 tokenizer expected.")
    
    tokens.append(tokenizer.eos_token_id) # APPEND EOS token to each document
    
    for tid in tokens:
        if tid > _UINT16_MAX:
            raise ValueError(
                f"Token ID {tid} exceeds uint16 max ({_UINT16_MAX}). "
                f"Vocab size {tokenizer.vocab_size} is too large for uint16 storage."
            )
    return np.array(tokens, dtype=np.uint16)


def _write_docs_to_memmap(
    dataset: Dataset,
    indices: list[int],
    output_path: Path,
    tokenizer: GPT2TokenizerFast,
    label: str,
) -> int:
    """Two-pass write: count tokens, then write directly into memmap.

    Returns the total number of tokens written.
    """
    # Pass 1 — count total tokens
    token_counts: list[int] = []
    for idx in indices:
        tokens = tokenizer.encode(dataset[idx]["text"], add_special_tokens=False)
        token_counts.append(len(tokens) + 1)  # +1 for EOS

    total_tokens = sum(token_counts)

    if total_tokens == 0:
        # Edge case: no tokens to write (empty split)
        np.memmap(str(output_path), dtype=np.uint16, mode="w+", shape=(1,))
        logger.warning("[tokenize] WARNING: %s split is empty.", label)
        return 0

    # Pass 2 — write directly into pre-allocated memmap
    mm = np.memmap(str(output_path), dtype=np.uint16, mode="w+", shape=(total_tokens,))
    offset = 0
    for i, idx in enumerate(indices):
        arr = _encode_document(dataset[idx]["text"], tokenizer)
        mm[offset : offset + len(arr)] = arr
        offset += len(arr)

        if (i + 1) % 10_000 == 0:
            logger.info(
                "[tokenize]   %s: wrote %s/%s docs (%s tokens) ...",
                label,
                f"{i + 1:,}",
                f"{len(indices):,}",
                f"{offset:,}",
            )

    mm.flush()
    del mm

    logger.info("[tokenize] Wrote %s tokens → %s", f"{total_tokens:,}", output_path)
    return total_tokens


def tokenize_to_memmap(
    dataset: Dataset,
    output_dir: str | Path = "data/tokenized",
    val_fraction: float = 0.05,
    seed: int = 42,
) -> Path:
    """Tokenize a HuggingFace Dataset into uint16 memmap files.

    Splits at the **document level** (shuffled, last ``val_fraction`` docs
    become validation) and writes each split directly into a pre-allocated
    memmap to avoid doubling peak memory.

    Args:
        dataset: HuggingFace Dataset with a ``"text"`` column.
        output_dir: Directory to write ``train.bin`` and ``val.bin``.
        val_fraction: Fraction of *documents* reserved for validation.
        seed: Random seed for document-level shuffle.

    Returns:
        Path to the output directory containing the memmap files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = get_gpt2_tokenizer()
    n_docs = len(dataset)

    logger.info(
        "[tokenize] Tokenizing %s documents with GPT-2 tokenizer ...", f"{n_docs:,}"
    )

    # Shuffle document indices and split into train / val
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_docs).tolist()

    val_count = max(1, int(n_docs * val_fraction))
    train_indices = indices[: n_docs - val_count]
    val_indices = indices[n_docs - val_count :]

    logger.info(
        "[tokenize] Document-level split: %s train, %s val",
        f"{len(train_indices):,}",
        f"{len(val_indices):,}",
    )

    # Write train.bin
    train_path = output_dir / "train.bin"
    train_tokens = _write_docs_to_memmap(
        dataset, train_indices, train_path, tokenizer, "train"
    )

    # Write val.bin
    val_path = output_dir / "val.bin"
    val_tokens = _write_docs_to_memmap(dataset, val_indices, val_path, tokenizer, "val")

    total = train_tokens + val_tokens
    logger.info(
        "[tokenize] Total: %s tokens from %s documents.", f"{total:,}", f"{n_docs:,}"
    )
    logger.info(
        "[tokenize] Train: %.1f MB, Val: %.1f MB",
        train_path.stat().st_size / 1e6,
        val_path.stat().st_size / 1e6,
    )

    return output_dir


def tokenize_multi_source_to_memmap(
    datasets: dict[str, Dataset],
    output_dir: str | Path = "data/tokenized",
    val_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Tokenize multiple data sources into unified train/val memmap files.

    Concatenates all source datasets, shuffles at document level, splits
    into train/val, and writes uint16 memmap files using the same two-pass
    approach as :func:`tokenize_to_memmap`.

    Args:
        datasets: Dict mapping source name → HuggingFace Dataset (each with
            a ``"text"`` column).
        output_dir: Directory to write ``train.bin`` and ``val.bin``.
        val_fraction: Fraction of *documents* reserved for validation.
        seed: Random seed for document-level shuffle.

    Returns:
        Tuple of (train_path, val_path).
    """
    from datasets import concatenate_datasets

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log per-source sizes before merging
    for name, ds in datasets.items():
        logger.info("[tokenize] Source %s: %s documents", name, f"{len(ds):,}")

    # Concatenate all sources into a single dataset
    all_datasets = list(datasets.values())
    combined = concatenate_datasets(all_datasets)
    n_docs = len(combined)
    logger.info(
        "[tokenize] Combined %d sources → %s documents total.",
        len(datasets),
        f"{n_docs:,}",
    )

    # Shuffle and split (same logic as tokenize_to_memmap)
    tokenizer = get_gpt2_tokenizer()
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_docs).tolist()

    val_count = max(1, int(n_docs * val_fraction))
    train_indices = indices[: n_docs - val_count]
    val_indices = indices[n_docs - val_count :]

    logger.info(
        "[tokenize] Document-level split: %s train, %s val",
        f"{len(train_indices):,}",
        f"{len(val_indices):,}",
    )

    train_path = output_dir / "train.bin"
    train_tokens = _write_docs_to_memmap(
        combined, train_indices, train_path, tokenizer, "train"
    )

    val_path = output_dir / "val.bin"
    val_tokens = _write_docs_to_memmap(
        combined, val_indices, val_path, tokenizer, "val"
    )

    total = train_tokens + val_tokens
    logger.info(
        "[tokenize] Total: %s tokens from %s documents.", f"{total:,}", f"{n_docs:,}"
    )
    logger.info(
        "[tokenize] Train: %.1f MB (%s tokens), Val: %.1f MB (%s tokens)",
        train_path.stat().st_size / 1e6,
        f"{train_tokens:,}",
        val_path.stat().st_size / 1e6,
        f"{val_tokens:,}",
    )

    return train_path, val_path


def tokenize_eval_test_set(
    eval_test_dir: str | Path,
    output_dir: str | Path = "data/tokenized",
) -> Path:
    """Tokenize the HF eval test set from disk into ``test.bin``.

    Args:
        eval_test_dir: Path to the eval test set saved with ``save_to_disk()``.
        output_dir: Directory to write ``test.bin``.

    Returns:
        Path to the ``test.bin`` file.
    """
    eval_test_dir = Path(eval_test_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not eval_test_dir.exists():
        raise FileNotFoundError(
            f"Eval test set not found at {eval_test_dir}. "
            f"Please ensure the eval/test/ directory is present."
        )

    logger.info("[tokenize] Loading eval test set from %s ...", eval_test_dir)
    test_ds = load_from_disk(str(eval_test_dir))

    tokenizer = get_gpt2_tokenizer()
    test_indices = list(range(len(test_ds)))

    test_path = output_dir / "test.bin"
    n_tokens = _write_docs_to_memmap(
        test_ds, test_indices, test_path, tokenizer, "test"
    )
    logger.info(
        "[tokenize] Test: %.1f MB (%s tokens)",
        test_path.stat().st_size / 1e6,
        f"{n_tokens:,}",
    )

    return test_path
