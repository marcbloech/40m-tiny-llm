"""Clean, filter, deduplicate, and remove eval-set overlap from training data."""

# AI Disclaimer: Large parts of this code were developed with Claude Code.
# Reasoning: Text cleaning logic is not core of the research and no major architectural decisions were made here,
# so we prioritized rapid development and reliability.

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from datasets import Dataset, load_from_disk

from tiny_llm.data.text_cleaning import clean_text, is_probably_english

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns compiled once
# ---------------------------------------------------------------------------
# Sentence boundary: period/!/? followed by whitespace and an uppercase letter,
# but NOT preceded by a common abbreviation. Python re doesn't support variable-
# length lookbehinds, so we use finditer + manual abbreviation check instead.
_RE_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_ABBREVIATIONS = frozenset(
    "Mr Mrs Ms Dr Prof St Jr Sr Inc Ltd Corp vs etc al cf approx dept est govt "
    "No Vol Jan Feb Mar Apr Jun Jul Aug Sep Oct Nov Dec Gen Sgt Col Maj Lt Rev "
    "Sgt Ave Blvd Rd Univ Assn Bros Co Dist Mt Ft".split()
)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, respecting abbreviations and decimals."""
    sentences: list[str] = []
    last = 0
    for m in _RE_SENTENCE_BOUNDARY.finditer(text):
        candidate_end = m.start()
        # Check the word before the period — skip if it's an abbreviation
        prefix = text[last:candidate_end].rstrip()
        # Extract last word before the punctuation
        last_word = prefix.rsplit(None, 1)[-1].rstrip(".!?") if prefix else ""
        if last_word in _ABBREVIATIONS:
            continue
        # Skip decimal numbers: check if char before '.' is a digit
        if prefix and prefix[-1] == "." and len(prefix) >= 2 and prefix[-2].isdigit():
            continue
        sentences.append(text[last : m.start()])
        last = m.end()
    sentences.append(text[last:])
    return [s for s in sentences if s]


# ---------------------------------------------------------------------------
# Step 1 — Text cleaning
# ---------------------------------------------------------------------------


def _clean_text(text: str) -> str:
    """Apply all text-level cleaning transformations."""
    return clean_text(text)


def _clean_batch(batch: dict) -> dict:
    """Batch-mapped cleaning function for HF Dataset.map()."""
    batch["text"] = [_clean_text(t) for t in batch["text"]]
    return batch


# ---------------------------------------------------------------------------
# Step 3 — English-only filter (fast-langdetect / fasttext LID)
# ---------------------------------------------------------------------------


def _is_english(text: str, threshold: float = 0.65) -> bool:
    """Check if text is English using fasttext language identification.

    Uses fast-langdetect (fasttext LID wrapper). Checks the first 500
    characters for speed. Returns True if the detected language is English
    with confidence above the threshold.
    """
    return is_probably_english(text, threshold=threshold)


# ---------------------------------------------------------------------------
# Wikipedia-specific cleaning (Phase 4)
# ---------------------------------------------------------------------------

_RE_WIKI_HEADING = re.compile(r"={2,6}\s*[^=]+\s*={2,6}")
_RE_WIKI_REF = re.compile(r"\[\d+\]|\[citation needed\]|\[edit\]")
_RE_WIKI_INFOBOX = re.compile(r"\{\{[^}]*\}\}")
_RE_WIKI_CATEGORY = re.compile(r"\[\[Category:[^\]]*\]\]", re.IGNORECASE)
_RE_WIKI_FILE = re.compile(r"\[\[(?:File|Image):[^\]]*\]\]", re.IGNORECASE)


def _clean_wikipedia(text: str) -> str:
    """Strip Wikipedia markup artifacts from article text.

    Handles section headers (``== Title ==``), reference markers (``[1]``),
    infobox templates, category links, and file/image links.
    """
    text = _RE_WIKI_HEADING.sub("", text)
    text = _RE_WIKI_REF.sub("", text)
    text = _RE_WIKI_INFOBOX.sub("", text)
    text = _RE_WIKI_CATEGORY.sub("", text)
    text = _RE_WIKI_FILE.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Step 4 — Test-set sentence overlap removal
# ---------------------------------------------------------------------------


def _build_test_sentence_hashes(eval_test_dir: str | Path) -> set[bytes]:
    """Build a set of MD5 hashes of normalized sentences from the eval test set."""
    eval_test_dir = Path(eval_test_dir)
    if not eval_test_dir.exists():
        raise FileNotFoundError(
            f"Eval test set not found at {eval_test_dir}. "
            f"Please copy the eval/test/ directory there, or use --eval-test-dir."
        )

    logger.info("[preprocess] Loading eval test set from %s ...", eval_test_dir)
    test_ds = load_from_disk(str(eval_test_dir))

    hashes: set[bytes] = set()
    for doc in test_ds:
        text = doc["text"]
        sentences = _split_sentences(text)
        for sent in sentences:
            norm = " ".join(sent.lower().split())
            if len(norm) < 10:
                continue
            h = hashlib.md5(norm.encode("utf-8")).digest()
            hashes.add(h)

    logger.info("[preprocess] Built %s test-set sentence hashes.", f"{len(hashes):,}")
    return hashes


def _remove_overlapping_sentences(text: str, blocked: set[bytes]) -> str:
    """Remove sentences from text that overlap with the test set."""
    sentences = _split_sentences(text)
    kept = []
    for sent in sentences:
        norm = " ".join(sent.lower().split())
        if len(norm) < 10:
            kept.append(sent)
            continue
        h = hashlib.md5(norm.encode("utf-8")).digest()
        if h not in blocked:
            kept.append(sent)
    return " ".join(kept)


# ---------------------------------------------------------------------------
# Step 5 — Near-duplicate removal (MinHash LSH)
# ---------------------------------------------------------------------------


def _minhash_dedup(
    dataset: Dataset, threshold: float = 0.8, num_perm: int = 128
) -> Dataset:
    """Remove near-duplicate documents using MinHash LSH.

    Builds an LSH index from word 5-gram shingles, finds clusters of
    near-duplicates by Jaccard similarity, and keeps only the first
    document encountered in each cluster.
    """
    from datasketch import MinHash, MinHashLSH

    n = len(dataset)
    texts = dataset["text"]
    logger.info(
        "[preprocess]   Computing MinHash signatures for %s documents ...", f"{n:,}"
    )

    # Build MinHash signatures
    signatures: list[MinHash] = []
    for text in texts:
        m = MinHash(num_perm=num_perm)
        words = text.lower().split()
        for i in range(len(words) - 4):
            shingle = " ".join(words[i : i + 5])
            m.update(shingle.encode("utf-8"))
        signatures.append(m)

    # Build LSH index — query before insert to find duplicates
    logger.info("[preprocess]   Building LSH index (threshold=%.2f) ...", threshold)
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    duplicate_indices: set[int] = set()

    for i, sig in enumerate(signatures):
        if i in duplicate_indices:
            continue
        # Documents too short to produce shingles get an empty MinHash;
        # treat them as unique to avoid false matches.
        words = texts[i].lower().split()
        if len(words) < 5:
            lsh.insert(str(i), sig)
            continue
        try:
            result = lsh.query(sig)
            if result:
                duplicate_indices.add(i)
                continue
            lsh.insert(str(i), sig)
        except ValueError:
            duplicate_indices.add(i)

    keep_indices = [i for i in range(n) if i not in duplicate_indices]
    logger.info(
        "[preprocess]   MinHash found %s near-duplicates.",
        f"{len(duplicate_indices):,}",
    )
    return dataset.select(keep_indices)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def preprocess(
    dataset: Dataset,
    eval_test_dir: str | Path,
    min_length: int = 50,
    seed: int = 42,
    source_name: str | None = None,
    ngram_index: "frozenset[bytes] | None" = None,
    decontam_threshold: int = 0,
) -> Dataset:
    """Full preprocessing pipeline: clean → filter length → filter English → decontaminate.

    Args:
        dataset: HuggingFace Dataset with a ``"text"`` column.
        eval_test_dir: Path to the eval test set saved with ``save_to_disk()``.
        min_length: Minimum document length in characters after cleaning.
        seed: Random seed (unused here but kept for API consistency).
        source_name: Optional source identifier for source-specific cleaning
            (e.g. ``"wikipedia"`` triggers wiki markup removal).
        ngram_index: Pre-built 13-gram index for comprehensive decontamination.
            If provided, replaces the legacy sentence-hash overlap removal with
            n-gram decontamination against all eval benchmarks.  If ``None``,
            falls back to the original OWT-only sentence-hash approach.
        decontam_threshold: Maximum matching n-grams before a document is removed
            (default 0 = remove on first match, following GPT-3/Llama).

    Returns:
        Cleaned and filtered HuggingFace Dataset.
    """
    n_start = len(dataset)
    # Use single-process map/filter throughout. Multi-proc (num_proc > 1) causes
    # worker subprocess crashes under memory pressure (OOM kill) and offers
    # little benefit for a one-time preprocessing pipeline.
    num_workers = 1
    logger.info(
        "[preprocess] Starting with %s documents (source=%s).",
        f"{n_start:,}",
        source_name or "default",
    )

    # Step 1 — Text cleaning (pure function, safe for multiprocessing)
    logger.info("[preprocess] Step 1/5: Cleaning text ...")

    if source_name == "wikipedia":
        # Wikipedia-specific cleaning before standard pipeline
        def _clean_wiki_batch(batch: dict) -> dict:
            batch["text"] = [_clean_wikipedia(t) for t in batch["text"]]
            return batch

        dataset = dataset.map(
            _clean_wiki_batch, batched=True, num_proc=num_workers, desc="Wiki cleanup"
        )

    dataset = dataset.map(
        _clean_batch, batched=True, num_proc=num_workers, desc="Cleaning"
    )

    # Step 2 — Minimum length filter (pure function)
    logger.info("[preprocess] Step 2/5: Filtering short documents ...")
    # Wikipedia uses a higher minimum (200 chars) to skip stubs
    effective_min = max(min_length, 200) if source_name == "wikipedia" else min_length
    dataset = dataset.filter(
        lambda x: len(x["text"]) >= effective_min,
        num_proc=num_workers,
        desc="Min-length filter",
    )
    n_after_length = len(dataset)
    logger.info(
        "[preprocess]   Removed %s short docs → %s remain.",
        f"{n_start - n_after_length:,}",
        f"{n_after_length:,}",
    )

    # Step 4 — Exact dedup safety net (mutable state, must be single-threaded)
    logger.info("[preprocess] Step 4/7: Exact dedup ...")
    seen_hashes: set[bytes] = set()
    n_before_dedup = len(dataset)

    def _is_unique(example: dict) -> bool:
        h = hashlib.md5(example["text"].encode("utf-8")).digest()
        if h in seen_hashes:
            return False
        seen_hashes.add(h)
        return True

    dataset = dataset.filter(_is_unique, desc="Exact dedup")
    n_after_dedup = len(dataset)
    logger.info(
        "[preprocess]   Removed %s exact duplicates → %s remain.",
        f"{n_before_dedup - n_after_dedup:,}",
        f"{n_after_dedup:,}",
    )

    # Step 3 — English-only filter (uses fast-langdetect / fasttext LID)
    # Skip for sources that are already English-only
    if source_name in ("wikipedia", "bookcorpus"):
        logger.info(
            "[preprocess] Step 3/5: Skipping English filter (source=%s is English-only).",
            source_name,
        )
        n_after_lang = n_after_dedup
    else:
        logger.info("[preprocess] Step 3/5: Filtering non-English documents (fasttext) ...")
        dataset = dataset.filter(
            lambda x: _is_english(x["text"]),
            num_proc=1,
            desc="English filter",
        )
        n_after_lang = len(dataset)
        logger.info(
            "[preprocess]   Removed %s non-English docs → %s remain.",
            f"{n_after_dedup - n_after_lang:,}",
            f"{n_after_lang:,}",
        )

    # Step 4 — Decontamination
    if ngram_index is not None:
        # New comprehensive n-gram decontamination against all benchmarks
        from tiny_llm.data.decontaminate import decontaminate

        logger.info("[preprocess] Step 4/5: N-gram decontamination (comprehensive) ...")
        dataset = decontaminate(dataset, ngram_index, threshold=decontam_threshold)
    else:
        # Legacy: sentence-hash overlap removal against OWT test set only
        logger.info(
            "[preprocess] Step 4/5: Removing test-set overlapping sentences (legacy) ..."
        )
        blocked_hashes = _build_test_sentence_hashes(eval_test_dir)

        def _remove_overlap_batch(batch: dict) -> dict:
            batch["text"] = [
                _remove_overlapping_sentences(t, blocked_hashes) for t in batch["text"]
            ]
            return batch

        dataset = dataset.map(
            _remove_overlap_batch,
            batched=True,
            num_proc=num_workers,
            desc="Overlap removal",
        )

    # Step 5 — Re-filter docs that became too short after decontamination
    logger.info("[preprocess] Step 5/5: Post-decontamination length filter ...")
    dataset = dataset.filter(
        lambda x: len(x["text"]) >= effective_min,
        num_proc=num_workers,
        desc="Post-decontamination length filter",
    )
    n_final = len(dataset)
    logger.info(
        "[preprocess]   Removed %s docs after decontamination → %s remain.",
        f"{n_after_lang - n_final:,}",
        f"{n_final:,}",
    )

    logger.info(
        "[preprocess] Done. %s → %s documents (%s retained).",
        f"{n_start:,}",
        f"{n_final:,}",
        f"{n_final / n_start:.1%}",
    )
    return dataset


# ---------------------------------------------------------------------------
# Cross-source deduplication
# ---------------------------------------------------------------------------


def cross_source_dedup(
    datasets: dict[str, Dataset],
    priority_order: list[str],
) -> dict[str, Dataset]:
    """Remove exact duplicates across sources, keeping the higher-priority copy.

    Iterates sources in *priority_order*.  For each document, computes an MD5
    hash of its text.  If the hash was already seen in a higher-priority source,
    the document is removed.

    Args:
        datasets: Mapping from source name → HuggingFace Dataset.
        priority_order: Source names in descending priority (first = keep).

    Returns:
        New dict with the same keys, but lower-priority sources have
        cross-source duplicates removed.
    """
    global_hashes: set[bytes] = set()
    result: dict[str, Dataset] = {}

    for name in priority_order:
        ds = datasets[name]
        n_before = len(ds)

        # Track new hashes from this source (single-threaded for set mutation)
        new_in_source: set[bytes] = set()

        def _is_unique_cross(example: dict) -> bool:
            h = hashlib.md5(example["text"].encode("utf-8")).digest()
            if h in global_hashes:
                return False
            new_in_source.add(h)
            return True

        ds = ds.filter(_is_unique_cross, desc=f"Cross-dedup {name}")
        global_hashes.update(new_in_source)

        n_after = len(ds)
        removed = n_before - n_after
        if removed > 0:
            logger.info(
                "[cross-dedup] %s: removed %s cross-source duplicates → %s remain.",
                name,
                f"{removed:,}",
                f"{n_after:,}",
            )
        else:
            logger.info("[cross-dedup] %s: no cross-source duplicates found.", name)

        result[name] = ds

    return result
