"""Comprehensive decontamination against all evaluation benchmarks.

Uses 13-gram overlap detection (as in GPT-3 / Llama) to remove training
documents that contain passages from any eval benchmark.  This is stricter
than sentence-level MD5 hashing because it catches *partial* overlaps.

Benchmarks covered:
  - OpenWebText eval test set  (local, ``data/eval/test/``)
  - WikiText-103 raw           (test + validation splits)
  - LAMBADA (OpenAI variant)   (test split)
  - HellaSwag                  (validation split)
  - Winogrande (XL)            (validation split)
  - OpenBookQA                 (test split)
"""

# AI Disclaimer: Large parts of this code were developed with Claude Code.
# Reasoning: Decontamination logic is not core research — it's a standard
# data hygiene step.  We prioritised correctness and coverage.

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 13-gram index construction
# ---------------------------------------------------------------------------

_DEFAULT_N = 13


def _text_to_ngram_hashes(text: str, n: int) -> list[bytes]:
    """Convert text into a list of n-gram MD5 hashes (16-byte digests)."""
    words = text.lower().split()
    if len(words) < n:
        return []
    hashes: list[bytes] = []
    for i in range(len(words) - n + 1):
        gram = " ".join(words[i : i + n])
        hashes.append(hashlib.md5(gram.encode("utf-8")).digest())
    return hashes


def _add_texts_to_index(
    texts: list[str], index: set[bytes], n: int, source_name: str
) -> None:
    """Extract n-gram hashes from a list of texts and add them to *index*."""
    count = 0
    for text in texts:
        ngrams = _text_to_ngram_hashes(text, n)
        index.update(ngrams)
        count += len(ngrams)
    logger.info(
        "[decontaminate] %s: added %s %d-grams from %s texts.",
        source_name,
        f"{count:,}",
        n,
        f"{len(texts):,}",
    )


def _load_benchmark_texts(
    name: str,
    hf_id: str,
    hf_subset: str | None,
    hf_split: str,
    text_columns: list[str],
) -> list[str]:
    """Download a HF benchmark and extract all relevant text fields."""
    logger.info("[decontaminate] Loading %s (%s/%s) ...", name, hf_id, hf_split)
    ds = load_dataset(hf_id, hf_subset, split=hf_split)
    texts: list[str] = []
    for row in ds:
        for col in text_columns:
            # Handle nested fields like choices.text in OpenBookQA
            val = row.get(col)
            if val is None:
                continue
            if isinstance(val, str):
                texts.append(val)
            elif isinstance(val, dict):
                # Handle nested dicts like choices: {"text": [...], "label": [...]}
                nested = val.get("text")
                if isinstance(nested, list):
                    texts.extend(s for s in nested if isinstance(s, str))
                elif isinstance(nested, str):
                    texts.append(nested)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        texts.append(item)
                    elif isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])
    logger.info(
        "[decontaminate]   Got %s text segments from %s.", f"{len(texts):,}", name
    )
    return texts


# Benchmark registry: (name, hf_id, subset, split, text_columns)
_BENCHMARKS: list[tuple[str, str, str | None, str, list[str]]] = [
    # WikiText-103 test + validation
    (
        "wikitext-103-test",
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        "test",
        ["text"],
    ),
    (
        "wikitext-103-val",
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        "validation",
        ["text"],
    ),
    # LAMBADA
    ("lambada", "EleutherAI/lambada_openai", None, "test", ["text"]),
    # HellaSwag
    ("hellaswag", "Rowan/hellaswag", None, "validation", ["ctx_a", "ctx_b", "endings"]),
    # Winogrande
    (
        "winogrande",
        "allenai/winogrande",
        "winogrande_xl",
        "validation",
        ["sentence", "option1", "option2"],
    ),
    # OpenBookQA — include answer choices (used by eval via row["choices"]["text"])
    ("openbookqa", "allenai/openbookqa", "main", "test", ["question_stem", "choices"]),
]


def build_ngram_index(
    n: int = _DEFAULT_N,
    owt_eval_test_dir: str | Path | None = "data/eval/test",
) -> frozenset[bytes]:
    """Build a set of n-gram hashes from all evaluation benchmarks.

    Downloads each benchmark from HuggingFace, extracts text fields,
    computes n-gram hashes, and returns a frozen set.  Also includes
    the local OpenWebText eval test set if the directory exists.

    Args:
        n: Size of n-grams (default 13, following GPT-3/Llama).
        owt_eval_test_dir: Path to the OWT eval test set on disk.
            Set to ``None`` to skip.

    Returns:
        Frozen set of MD5 hashes (16 bytes each).
    """
    index: set[bytes] = set()

    # 1. Local OWT eval test set
    if owt_eval_test_dir is not None:
        owt_path = Path(owt_eval_test_dir)
        if owt_path.exists():
            logger.info("[decontaminate] Loading local OWT eval test set ...")
            owt_ds = load_from_disk(str(owt_path))
            owt_texts = [row["text"] for row in owt_ds]
            _add_texts_to_index(owt_texts, index, n, "OWT-eval-test")
        else:
            logger.warning(
                "[decontaminate] OWT eval test dir not found at %s, skipping.",
                owt_path,
            )

    # 2. All HuggingFace benchmarks
    for name, hf_id, subset, split, text_cols in _BENCHMARKS:
        try:
            texts = _load_benchmark_texts(name, hf_id, subset, split, text_cols)
            _add_texts_to_index(texts, index, n, name)
        except Exception:
            logger.exception("[decontaminate] Failed to load %s — skipping.", name)

    logger.info(
        "[decontaminate] Built n-gram index: %s unique %d-gram hashes (%.1f MB).",
        f"{len(index):,}",
        n,
        len(index) * 16 / 1e6,
    )
    return frozenset(index)


# ---------------------------------------------------------------------------
# Decontamination filter
# ---------------------------------------------------------------------------


def _count_matching_ngrams(text: str, ngram_index: frozenset[bytes], n: int) -> int:
    """Count how many n-grams in *text* appear in *ngram_index*."""
    count = 0
    for h in _text_to_ngram_hashes(text, n):
        if h in ngram_index:
            count += 1
    return count


def decontaminate(
    dataset: Dataset,
    ngram_index: frozenset[bytes],
    n: int = _DEFAULT_N,
    threshold: int = 0,
) -> Dataset:
    """Remove documents containing matching n-grams from eval benchmarks.

    A document is removed if it contains more than *threshold* n-grams
    that appear in the evaluation benchmark index.  The default of 0
    (following GPT-3 / Llama) removes any document with at least one
    matching 13-gram — a single 13-word match is near-certain contamination.

    Args:
        dataset: HuggingFace Dataset with a ``"text"`` column.
        ngram_index: Frozen set of n-gram hashes from :func:`build_ngram_index`.
        n: Size of n-grams (must match what was used to build the index).
        threshold: Maximum number of matching n-grams before removal
            (default 0 = remove on first match).

    Returns:
        Filtered dataset with contaminated documents removed.
    """
    n_before = len(dataset)
    logger.info(
        "[decontaminate] Filtering %s documents (threshold=%d matching %d-grams) ...",
        f"{n_before:,}",
        threshold,
        n,
    )

    dataset = dataset.filter(
        lambda x: _count_matching_ngrams(x["text"], ngram_index, n) <= threshold,
        desc="Decontamination",
    )

    n_after = len(dataset)
    logger.info(
        "[decontaminate] Removed %s contaminated documents → %s remain.",
        f"{n_before - n_after:,}",
        f"{n_after:,}",
    )
    return dataset
