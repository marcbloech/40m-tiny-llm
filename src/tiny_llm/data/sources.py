"""Multi-source data downloading with ratio-based token budget allocation.

Each source is streamed from HuggingFace, shuffled, and truncated to an
estimated document count that yields approximately the target number of
tokens.  Over-sampling by 10% compensates for preprocessing losses.

Typical usage::

    from tiny_llm.data.sources import download_all_sources

    datasets = download_all_sources(config.data.sources, config.data.total_tokens)
"""

# AI Disclaimer: Large parts of this code were developed with Claude Code.
# Reasoning: Data downloading logic is not core research.  We prioritised
# correctness and reliability of the streaming + sampling approach.

from __future__ import annotations

import logging
from dataclasses import dataclass

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

# Rough heuristic: average characters per GPT-2 token across English web text.
_CHARS_PER_TOKEN = 4.0

# Over-sample factor to account for preprocessing losses (filtering, dedup, etc.)
_OVERSAMPLE_FACTOR = 1.10


@dataclass
class SourceDownloadPlan:
    """Computed download plan for a single source."""

    name: str
    hf_dataset_id: str
    hf_subset: str | None
    hf_split: str
    text_column: str
    target_tokens: int
    estimated_docs: int


# Per-source average document length heuristics (characters).
# These are rough empirical estimates; adjust if observed doc lengths differ.
_AVG_DOC_CHARS: dict[str, float] = {
    "openwebtext": 2000.0,
    "fineweb": 2500.0,
    "fineweb_edu": 2500.0,
    "c4": 2000.0,
    "wikipedia": 8000.0,   # Wikipedia articles are much longer than web docs
    "bookcorpus": 5000.0,
}
_DEFAULT_AVG_DOC_CHARS = 2000.0


def _estimate_doc_count(
    target_tokens: int,
    avg_doc_chars: float = _DEFAULT_AVG_DOC_CHARS,
) -> int:
    """Estimate how many documents to download for a given token target.

    Uses the heuristic:  target_tokens * chars_per_token / avg_doc_chars.
    Multiplies by the over-sample factor to compensate for preprocessing losses.
    """
    target_chars = target_tokens * _CHARS_PER_TOKEN * _OVERSAMPLE_FACTOR
    return max(1, int(target_chars / avg_doc_chars))


def download_source(
    name: str,
    hf_dataset_id: str,
    hf_subset: str | None,
    hf_split: str,
    text_column: str,
    target_tokens: int,
    seed: int = 42,
    max_docs: int | None = None,
) -> Dataset:
    """Stream-download a single source with enough docs for the token target.

    Args:
        name: Human-readable source name (for logging).
        hf_dataset_id: HuggingFace dataset identifier.
        hf_subset: Dataset subset/config (e.g. ``"20231101.en"``), or ``None``.
        hf_split: Dataset split (e.g. ``"train"``).
        text_column: Name of the text column.
        target_tokens: Approximate number of tokens desired from this source.
        seed: Random seed for shuffle reproducibility.
        max_docs: If set, cap the number of documents downloaded (useful for
            quick test runs via ``--num-samples``).

    Returns:
        HuggingFace Dataset with at least a ``"text"`` column.
    """
    avg_doc_chars = _AVG_DOC_CHARS.get(name, _DEFAULT_AVG_DOC_CHARS)
    estimated_docs = _estimate_doc_count(target_tokens, avg_doc_chars=avg_doc_chars)
    if max_docs is not None:
        estimated_docs = min(estimated_docs, max_docs)

    logger.info(
        "[sources] Downloading %s: %s (subset=%s, split=%s) — "
        "target %s tokens, est. %s docs%s ...",
        name,
        hf_dataset_id,
        hf_subset,
        hf_split,
        f"{target_tokens:,}",
        f"{estimated_docs:,}",
        " (capped by --num-samples)"
        if max_docs is not None and max_docs < _estimate_doc_count(target_tokens, avg_doc_chars)
        else "",
    )

    stream = load_dataset(
        hf_dataset_id,
        hf_subset,
        split=hf_split,
        streaming=True,
        trust_remote_code=False,
    )
    # Shuffle buffer: prefilling this buffer is the first silent phase — log a warning
    # so it's clear the pipeline hasn't stalled.
    shuffle_buffer = min(1_000, estimated_docs)
    logger.info(
        "[sources] %s: filling shuffle buffer (%s items) — no output until buffer is full ...",
        name,
        f"{shuffle_buffer:,}",
    )
    stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)
    stream = stream.take(estimated_docs)

    # Materialize streaming iterator, keeping only the text column
    rows = []
    for i, item in enumerate(stream):
        text = item.get(text_column, "")
        if text:
            rows.append({"text": text})
        if (i + 1) % 500 == 0:
            logger.info(
                "[sources] %s: streamed %s / ~%s docs ...",
                name,
                f"{i + 1:,}",
                f"{estimated_docs:,}",
            )

    dataset = Dataset.from_list(rows)
    logger.info(
        "[sources] %s: downloaded %s documents.",
        name,
        f"{len(dataset):,}",
    )
    return dataset


def download_all_sources(
    source_configs: list,
    total_tokens: int,
    seed: int = 42,
) -> dict[str, Dataset]:
    """Download all configured data sources.

    Computes ``target_tokens = config.ratio * total_tokens`` for each source,
    then downloads sequentially (streaming, so memory-efficient).

    Args:
        source_configs: List of :class:`DataSourceConfig` instances.
        total_tokens: Total token budget across all sources.
        seed: Random seed for shuffle reproducibility.

    Returns:
        Dict mapping source name → HuggingFace Dataset.
    """
    datasets: dict[str, Dataset] = {}

    ratio_sum = sum(s.ratio for s in source_configs)
    if abs(ratio_sum - 1.0) > 0.01:
        logger.warning(
            "[sources] Source ratios sum to %.3f (expected ~1.0). "
            "Token allocation will be proportional anyway.",
            ratio_sum,
        )

    for src in source_configs:
        target = int(src.ratio * total_tokens)
        ds = download_source(
            name=src.name,
            hf_dataset_id=src.hf_dataset_id,
            hf_subset=src.hf_subset,
            hf_split=src.hf_split,
            text_column=src.text_column,
            target_tokens=target,
            seed=seed,
        )
        datasets[src.name] = ds

    total_docs = sum(len(ds) for ds in datasets.values())
    logger.info(
        "[sources] All sources downloaded: %s total documents across %d sources.",
        f"{total_docs:,}",
        len(datasets),
    )
    return datasets
