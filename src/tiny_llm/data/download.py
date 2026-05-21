"""Download OpenWebText subset via HuggingFace streaming."""

from __future__ import annotations

import logging

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


def download_dataset(num_samples: int = 100_000, seed: int = 42) -> Dataset:
    """Stream-download a random subset of OpenWebText.

    Uses streaming mode to avoid downloading the full ~38 GB dataset.
    Shuffles with a rolling buffer and takes ``num_samples`` documents,
    then materializes into an in-memory HuggingFace Dataset.

    Args:
        num_samples: Number of documents to download.
        seed: Random seed for shuffle reproducibility.

    Returns:
        HuggingFace Dataset with a single ``"text"`` column.
    """
    logger.info(
        "[download] Streaming %s docs from Skylion007/openwebtext ...",
        f"{num_samples:,}",
    )

    stream = load_dataset(
        "Skylion007/openwebtext",
        split="train",
        streaming=True,
    )
    stream = stream.shuffle(seed=seed, buffer_size=10_000)
    stream = stream.take(num_samples)

    # Materialize streaming iterator into an in-memory Dataset
    rows = list(stream)
    dataset = Dataset.from_list(rows)

    logger.info("[download] Downloaded %s documents.", f"{len(dataset):,}")
    return dataset
