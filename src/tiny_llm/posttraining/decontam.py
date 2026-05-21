"""Public-eval decontamination helpers for SFT data."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import string
import time
import unicodedata
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 24 * 60 * 60
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE_PATH = _PROJECT_ROOT / "data" / "eval_decontam.json"
_PUNCT_TRANSLATION = str.maketrans({ch: " " for ch in string.punctuation})
_RE_WHITESPACE = re.compile(r"\s+")

_eval_hash_set: set[str] | None = None
_eval_hash_sources: dict[str, set[str]] = {}
_contamination_counter: Counter[str] = Counter()


def _normalise_for_decontam(text: str) -> str:
    """Normalise text before exact-hash public-eval matching."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = text.translate(_PUNCT_TRANSLATION)
    text = _RE_WHITESPACE.sub(" ", text)
    return text.strip()


def _hash_normalised_text(text: str) -> str:
    normalised = _normalise_for_decontam(text)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _cache_is_fresh(path: Path) -> bool:
    return path.exists() and time.time() - path.stat().st_mtime < _CACHE_TTL_SECONDS


def _load_cached_hashes(path: Path) -> set[str] | None:
    if not _cache_is_fresh(path):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        return None
    logger.info("[sft] Loaded %d public-eval decontam hashes from %s", len(payload), path)
    return set(payload)


def _write_cached_hashes(path: Path, hashes: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(hashes), indent=2) + "\n", encoding="utf-8")
    logger.info("[sft] Cached %d public-eval decontam hashes to %s", len(hashes), path)


def _add_hash(
    hashes: set[str],
    sources: dict[str, set[str]],
    eval_set: str,
    text: str,
) -> None:
    normalised = _normalise_for_decontam(text)
    if not normalised:
        return
    digest = hashlib.sha256(normalised.encode("utf-8")).hexdigest()
    hashes.add(digest)
    sources.setdefault(digest, set()).add(eval_set)


def _load_lambada_test(load_dataset):
    try:
        return load_dataset("EleutherAI/lambada", split="test")
    except Exception:
        try:
            return load_dataset("EleutherAI/lambada", "plain_text", split="test")
        except Exception:
            return load_dataset("EleutherAI/lambada_openai", split="test")


def build_eval_hash_set(
    *,
    force: bool = False,
    cache_path: str | Path = _DEFAULT_CACHE_PATH,
) -> set[str]:
    """Build or load exact public-eval hashes used for SFT decontamination."""
    global _eval_hash_set, _eval_hash_sources

    cache_path = Path(cache_path)
    if not force and _eval_hash_set is not None:
        return _eval_hash_set

    if not force:
        cached = _load_cached_hashes(cache_path)
        if cached is not None:
            _eval_hash_set = cached
            _eval_hash_sources = {}
            return _eval_hash_set

    from datasets import load_dataset

    logger.info("[sft] Building public-eval decontam hash set from HuggingFace splits…")
    hashes: set[str] = set()
    sources: dict[str, set[str]] = {}

    hellaswag = load_dataset("Rowan/hellaswag", split="validation")
    for row in hellaswag:
        _add_hash(hashes, sources, "hellaswag_validation", str(row.get("ctx") or ""))

    winogrande = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    for row in winogrande:
        _add_hash(hashes, sources, "winogrande_validation", str(row.get("sentence") or ""))

    openbookqa = load_dataset("allenai/openbookqa", "main", split="validation")
    for row in openbookqa:
        _add_hash(
            hashes,
            sources,
            "openbookqa_validation",
            str(row.get("question_stem") or ""),
        )

    lambada = _load_lambada_test(load_dataset)
    for row in lambada:
        _add_hash(hashes, sources, "lambada_test", str(row.get("text") or ""))

    _eval_hash_set = hashes
    _eval_hash_sources = sources
    _write_cached_hashes(cache_path, hashes)
    return _eval_hash_set


def _candidate_hashes(instruction: str, input_text: str) -> set[str]:
    instruction = instruction or ""
    input_text = input_text or ""
    texts = {
        instruction,
        f"{instruction}\n\n{input_text}",
    }
    prompt_stem = instruction.split("\n\n", 1)[0].strip()
    if prompt_stem:
        texts.add(prompt_stem)
    return {_hash_normalised_text(text) for text in texts if _normalise_for_decontam(text)}


def matching_eval_sets(instruction: str, input_text: str) -> set[str]:
    """Return public eval sets whose leakage hash matches this SFT row."""
    eval_hashes = build_eval_hash_set()
    matching_hashes = _candidate_hashes(instruction, input_text) & eval_hashes
    if not matching_hashes:
        return set()

    eval_sets: set[str] = set()
    for digest in matching_hashes:
        eval_sets.update(_eval_hash_sources.get(digest, set()))
    return eval_sets or {"unknown"}


def is_contaminated(instruction: str, input_text: str) -> bool:
    """Return whether the instruction/input pair appears in public eval splits."""
    eval_hashes = build_eval_hash_set()
    return bool(_candidate_hashes(instruction, input_text) & eval_hashes)


def record_contamination(instruction: str, input_text: str) -> None:
    """Increment per-eval-set counters for a contaminated row."""
    for eval_set in matching_eval_sets(instruction, input_text):
        _contamination_counter[eval_set] += 1


def reset_decontam_counters() -> None:
    _contamination_counter.clear()


def get_decontam_counters() -> dict[str, int]:
    return dict(sorted(_contamination_counter.items()))


__all__ = [
    "build_eval_hash_set",
    "get_decontam_counters",
    "is_contaminated",
    "matching_eval_sets",
    "record_contamination",
    "reset_decontam_counters",
]
