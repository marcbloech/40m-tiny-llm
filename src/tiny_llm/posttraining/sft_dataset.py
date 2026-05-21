"""Supervised fine-tuning dataset utilities.

Implements the Stanford Alpaca plain-text prompt format and a dataset registry
so additional SFT corpora (Dolly, OASST1, …) can be added without touching the
trainer.  A sample is represented on disk as::

    {"input_ids": [int, ...], "prompt_len": int}

where ``prompt_len`` is the number of tokens up to *and including* the
``### Response:\\n`` marker.  The trainer masks ``target_ids[:prompt_len - 1]``
with ``-100`` so gradients only flow through response-token predictions.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch.utils.data import Dataset

from tiny_llm.data.text_cleaning import clean_text, is_clean_instruction_example
from tiny_llm.posttraining import decontam
from tiny_llm.tokenizer import get_gpt2_tokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alpaca prompt template (Stanford, plain text — no special tokens added)
# ---------------------------------------------------------------------------

_PREAMBLE_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
)
_PREAMBLE_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request."
)


def format_alpaca_prompt(instruction: str, input_text: str = "") -> str:
    """Return the prompt string *up to and including* ``### Response:\\n``.

    The response body is appended separately so we can precisely count the
    prompt-token boundary used for loss masking.
    """
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return (
            f"{_PREAMBLE_WITH_INPUT}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    return (
        f"{_PREAMBLE_NO_INPUT}\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n"
    )


# ---------------------------------------------------------------------------
# Tokeniser access (GPT-2 via transformers.GPT2TokenizerFast)
# ---------------------------------------------------------------------------


def _get_tokenizer():
    return get_gpt2_tokenizer()


def _encode(enc, text: str) -> list[int]:
    return enc.encode(text, add_special_tokens=False)


EOS_TOKEN_ID = 50256  # GPT-2 "<|endoftext|>" — also serves as pad for SFT


# ---------------------------------------------------------------------------
# Tokenisation of a single example
# ---------------------------------------------------------------------------


@dataclass
class SFTExample:
    """A tokenised SFT example.

    Attributes:
        input_ids: Full token sequence [prompt tokens || response tokens || eos].
        prompt_len: Number of tokens that belong to the prompt (mask boundary).
        source: SFT dataset key that produced the example.
        truncated: Whether the response was shortened to fit ``max_seq_len``.
    """

    input_ids: list[int]
    prompt_len: int
    source: str = "unknown"
    truncated: bool = False


def tokenise_pair(
    enc,
    instruction: str,
    input_text: str,
    output: str,
    max_seq_len: int,
    raw_prompt: str | None = None,
) -> SFTExample | None:
    """Tokenise one instruction/response pair.

    Returns ``None`` if the prompt alone is longer than ``max_seq_len`` (we
    cannot fit any response) or if the response is empty.  If prompt+response
    exceeds ``max_seq_len``, the response is truncated.

    When ``raw_prompt`` is provided, it is used directly as the prompt string
    instead of wrapping instruction/input in the Alpaca template. This allows
    MC benchmark examples to use the exact multiple-choice prompt format.
    """
    output = (output or "").strip()
    if not output:
        return None

    prompt_str = raw_prompt if raw_prompt else format_alpaca_prompt(instruction, input_text)
    prompt_ids = _encode(enc, prompt_str)
    response_body_ids = _encode(enc, output)

    if len(prompt_ids) >= max_seq_len:
        # No budget left for the response — dropping.
        return None

    budget = max_seq_len - len(prompt_ids)
    truncated = len(response_body_ids) + 1 > budget
    if truncated:
        # Always keep EOS so generation learns a stop boundary even when the
        # response body has to be shortened.
        response_body_ids = response_body_ids[: max(0, budget - 1)]
    response_ids = response_body_ids + [EOS_TOKEN_ID]

    input_ids = prompt_ids + response_ids
    return SFTExample(
        input_ids=input_ids,
        prompt_len=len(prompt_ids),
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# Dataset loaders — registry
# ---------------------------------------------------------------------------


def _load_alpaca(max_samples: int | None, seed: int) -> Iterable[dict[str, str]]:
    """Yield raw ``{instruction, input, output}`` dicts from tatsu-lab/alpaca."""
    from datasets import load_dataset

    logger.info("Downloading tatsu-lab/alpaca from HuggingFace…")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    if max_samples is not None and max_samples > 0 and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))
        logger.info("Alpaca: capped to %d samples (max_samples)", max_samples)
    for row in ds:
        yield {
            "instruction": row.get("instruction", ""),
            "input": row.get("input", ""),
            "output": row.get("output", ""),
        }


def _load_dolly(max_samples: int | None, seed: int) -> Iterable[dict[str, str]]:
    """Yield English Dolly rows mapped into Alpaca ``instruction/input/output`` triples."""
    from datasets import load_dataset

    logger.info("Downloading databricks/databricks-dolly-15k from HuggingFace…")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    if max_samples is not None and max_samples > 0 and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))
        logger.info("Dolly: capped to %d samples (max_samples)", max_samples)

    kept = 0
    for row in ds:
        instruction = str(row.get("instruction") or "").strip()
        output = str(row.get("response") or "").strip()
        if not instruction or not output:
            continue
        kept += 1
        yield {
            "instruction": instruction,
            "input": str(row.get("context") or "").strip(),
            "output": output,
        }

    logger.info("Dolly: kept %d English prompt/response pairs", kept)


def _oasst1_row_is_usable(row: dict[str, Any]) -> bool:
    """Return whether an OASST1 message is safe to turn into SFT data."""
    text = str(row.get("text") or "").strip()
    role = str(row.get("role") or "")
    return bool(
        text
        and row.get("lang") == "en"
        and role in {"prompter", "assistant"}
        and bool(row.get("review_result", False))
        and not bool(row.get("deleted", False))
        and not bool(row.get("synthetic", False))
        and row.get("tree_state") == "ready_for_export"
    )


def _oasst1_reply_sort_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    """Prefer the best-ranked reviewed assistant reply for each prompt."""
    rank = row.get("rank")
    rank_key = int(rank) if isinstance(rank, int) else 1_000_000
    review_count = int(row.get("review_count") or 0)
    created = str(row.get("created_date") or "")
    message_id = str(row.get("message_id") or "")
    return (rank_key, -review_count, created, message_id)


def _oasst1_conversation_chain(
    message_id: str,
    rows_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Return the root→leaf chain for one assistant message if it alternates cleanly."""
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    current = rows_by_id.get(message_id)
    while current is not None:
        current_id = str(current.get("message_id") or "")
        if not current_id or current_id in seen:
            return None
        seen.add(current_id)
        chain.append(current)

        parent_id = str(current.get("parent_id") or "")
        if not parent_id:
            break
        current = rows_by_id.get(parent_id)
        if current is None:
            return None

    chain.reverse()
    if not chain or chain[0]["role"] != "prompter" or chain[-1]["role"] != "assistant":
        return None

    for idx, row in enumerate(chain):
        expected_role = "prompter" if idx % 2 == 0 else "assistant"
        if row["role"] != expected_role:
            return None
    return chain


def _format_oasst1_history(history_rows: list[dict[str, Any]]) -> str:
    """Serialise earlier chat turns into the Alpaca ``### Input:`` block."""
    if not history_rows:
        return ""

    blocks = ["Previous conversation:"]
    for row in history_rows:
        speaker = "User" if row["role"] == "prompter" else "Assistant"
        blocks.append(f"{speaker}:\n{row['text']}")
    return "\n\n".join(blocks)


def _load_oasst1(max_samples: int | None, seed: int) -> Iterable[dict[str, str]]:
    """Yield English OASST1 chats mapped into Alpaca ``instruction/input/output`` triples."""
    from datasets import load_dataset

    logger.info("Downloading OpenAssistant/oasst1 from HuggingFace…")
    dataset_dict = load_dataset("OpenAssistant/oasst1")

    usable_rows: list[dict[str, Any]] = []
    for split_name in sorted(dataset_dict.keys()):
        for raw_row in dataset_dict[split_name]:
            if not _oasst1_row_is_usable(raw_row):
                continue
            row = dict(raw_row)
            row["message_id"] = str(row.get("message_id") or "")
            row["parent_id"] = str(row.get("parent_id") or "")
            row["text"] = str(row.get("text") or "").strip()
            usable_rows.append(row)

    rows_by_id = {
        row["message_id"]: row for row in usable_rows if row["message_id"]
    }

    assistant_children_by_parent: dict[str, list[dict[str, Any]]] = {}
    for row in usable_rows:
        if row["role"] != "assistant" or not row["parent_id"]:
            continue
        parent = rows_by_id.get(row["parent_id"])
        if parent is None or parent["role"] != "prompter":
            continue
        assistant_children_by_parent.setdefault(row["parent_id"], []).append(row)

    examples: list[dict[str, str]] = []
    for children in assistant_children_by_parent.values():
        assistant_row = min(children, key=_oasst1_reply_sort_key)
        chain = _oasst1_conversation_chain(assistant_row["message_id"], rows_by_id)
        if chain is None or len(chain) < 2:
            continue

        prompt_row = chain[-2]
        examples.append(
            {
                "instruction": prompt_row["text"],
                "input": _format_oasst1_history(chain[:-2]),
                "output": assistant_row["text"],
            }
        )

    if max_samples is not None and max_samples > 0 and max_samples < len(examples):
        rng = random.Random(seed)
        rng.shuffle(examples)
        examples = examples[:max_samples]
        logger.info("OASST1: capped to %d samples (max_samples)", max_samples)

    logger.info(
        "OASST1: kept %d English reviewed prompt/response pairs",
        len(examples),
    )
    yield from examples


_MC_LETTERS = ("A", "B", "C", "D")
_LAMBADA_LIKE_INSTRUCTION = (
    "Read the passage and predict the single word that best completes it. "
    "Respond with one word, no punctuation."
)
_LAMBADA_LIKE_STOPWORDS = frozenset(
    (
        "the",
        "and",
        "with",
        "that",
        "this",
        "from",
        "into",
        "have",
        "been",
        "were",
        "will",
        "would",
        "could",
        "should",
        "their",
        "there",
        "which",
        "when",
        "what",
        "where",
        "while",
    )
)
_LAMBADA_LIKE_CONTENT_SUFFIXES = ("ing", "ed", "ion", "ness", "ment", "ity")
_RE_ARTICLE_HEADER = re.compile(r"^=+\s*[^=].*?\s*=+$")
_RE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_RE_ALPHA_WORD = re.compile(r"[A-Za-z]+")
_RE_TRAILING_WORD = re.compile(r"([A-Za-z]+)(?:[^A-Za-z]*)$")
_RE_EXPLICIT_MC_PROMPT = re.compile(
    r"(?i)\b("
    r"multiple[-\s]?choice|"
    r"which of the following|"
    r"choose\s+(?:the\s+)?(?:best|correct|right)?\s*(?:answer|option)?|"
    r"select\s+(?:the\s+)?(?:best|correct|right)?\s*(?:answer|option)?|"
    r"answer\s+choices?|"
    r"options?"
    r")\b"
)
_RE_MC_CHOICE_LABEL = re.compile(
    r"(?im)(?:^|[\n\r]|[ \t])(?:[\(\[]?([A-D])[\)\].:]|([A-D])\s*-)\s*"
)
_RE_LETTER_ONLY_ANSWER = re.compile(
    r"(?is)^\s*(?:the\s+)?(?:(?:correct|final)\s+)?"
    r"(?:answer|option|choice)?\s*(?:is|:|=)?\s*"
    r"[\(\[]?([A-D])[\)\].:\-]?\s*$"
)
_MC_ANSWER_LETTER_PATTERNS = (
    re.compile(
        r"(?i)\b(?:answer|option|choice)\s*(?:is|:|=)\s*"
        r"[\(\[]?([A-D])[\)\].:]?\b"
    ),
    re.compile(
        r"(?i)\b(?:correct|right|best|final)\s+(?:answer|option|choice)\s*"
        r"(?:is|:|=)?\s*[\(\[]?([A-D])[\)\].:]?\b"
    ),
    re.compile(
        r"(?i)\b(?:i\s+)?(?:choose|select|pick)\s*(?:option|choice)?\s*"
        r"[\(\[]?([A-D])[\)\].:]?\b"
    ),
    re.compile(
        r"(?i)\b(?:option|choice)\s*[\(\[]?([A-D])[\)\].:]?\s+"
        r"(?:is\s+)?(?:correct|right|best)\b"
    ),
    re.compile(
        r"(?i)\b([A-D])\s+(?:is\s+)?(?:the\s+)?"
        r"(?:correct|right|best)\s+(?:answer|option|choice)\b"
    ),
)
_RE_LEADING_MC_LETTER = re.compile(
    r"(?is)^\s*[\(\[]?([A-D])[\)\].:\-](?:\s|$)"
)


def _format_mc_instruction(question: str, options: list[str], letters: tuple[str, ...] = _MC_LETTERS) -> str:
    """Return the benchmark-style question plus labelled answer options."""
    question = str(question or "").strip()
    labelled_options = [
        f"{letter}) {str(option or '').strip()}"
        for letter, option in zip(letters, options, strict=False)
    ]
    return f"{question}\n\n" + "\n".join(labelled_options)


def _format_mc_prompt(
    context: str,
    options: list[str],
    *,
    prefix: str = "Context:",
    letters: tuple[str, ...] = _MC_LETTERS,
    instruction_line: str = "",
) -> str:
    """Build the exact prompt format used by the standard benchmark.

    Format:
        {prefix} {context}
        A) option1
        B) option2
        ...
        [instruction_line]
        Answer:
    """
    context = str(context or "").strip()
    lines = [f"{prefix} {context}"]
    for i, opt in enumerate(options):
        lines.append(f"{letters[i]}) {str(opt or '').strip()}")
    if instruction_line:
        lines.append(instruction_line)
    lines.append("Answer:")
    return "\n".join(lines)


def _maybe_cap_rows(
    rows: list[dict[str, Any]],
    *,
    max_samples: int | None,
    seed: int,
    label: str,
    internal_cap: int | None = None,
) -> list[dict[str, Any]]:
    """Apply deterministic per-loader sample caps."""
    cap = max_samples if max_samples is not None and max_samples > 0 else internal_cap
    if cap is not None and cap > 0 and cap < len(rows):
        rng = random.Random(seed)
        rows = list(rows)
        rng.shuffle(rows)
        rows = rows[:cap]
        cap_source = "max_samples" if max_samples is not None and max_samples > 0 else "internal cap"
        logger.info("%s: capped to %d samples (%s)", label, cap, cap_source)
    return rows


def _normalise_choice_answer(
    answer_key: Any,
    labels: list[Any],
    *,
    expected_count: int = 4,
) -> str | None:
    """Map dataset-native choice labels or 1-based numeric keys to A-D."""
    raw_answer = str(answer_key or "").strip()
    normalised_labels = [str(label or "").strip() for label in labels]
    if len(normalised_labels) != expected_count:
        return None

    if raw_answer in normalised_labels:
        idx = normalised_labels.index(raw_answer)
        return _MC_LETTERS[idx]

    upper_answer = raw_answer.upper()
    upper_labels = [label.upper() for label in normalised_labels]
    if upper_answer in upper_labels:
        idx = upper_labels.index(upper_answer)
        return _MC_LETTERS[idx]

    if raw_answer.isdigit():
        idx = int(raw_answer) - 1
        if 0 <= idx < expected_count:
            return _MC_LETTERS[idx]

    if upper_answer in _MC_LETTERS[:expected_count]:
        return upper_answer

    return None


def _choice_texts_and_labels(row: dict[str, Any]) -> tuple[list[str], list[Any]]:
    choices = row.get("choices") or {}
    if not isinstance(choices, dict):
        return [], []
    texts = [str(text or "").strip() for text in choices.get("text", [])]
    labels = list(choices.get("label", []))
    return texts, labels


def _load_arc(max_samples: int | None, seed: int) -> Iterable[dict[str, str]]:
    """Yield ARC-Easy and ARC-Challenge train examples as A-D MC rows."""
    from datasets import load_dataset

    logger.info("Downloading allenai/ai2_arc ARC-Easy/ARC-Challenge train from HuggingFace…")
    rows: list[dict[str, Any]] = []
    for config_name in ("ARC-Easy", "ARC-Challenge"):
        ds = load_dataset("allenai/ai2_arc", config_name, split="train")
        rows.extend(dict(row) for row in ds)
    rows = _maybe_cap_rows(rows, max_samples=max_samples, seed=seed, label="ARC")

    kept = 0
    for row in rows:
        texts, labels = _choice_texts_and_labels(row)
        if len(texts) != 4:
            continue
        answer = _normalise_choice_answer(row.get("answerKey"), labels)
        if answer is None:
            continue
        question = str(row.get("question") or "")
        kept += 1
        yield {
            "instruction": _format_mc_instruction(question, texts),
            "input": "",
            "output": answer,
            "raw_prompt": _format_mc_prompt(question, texts, prefix="Question:"),
        }

    logger.info("ARC: kept %d train multiple-choice pairs", kept)


def _load_openbookqa(max_samples: int | None, seed: int) -> Iterable[dict[str, str]]:
    """Yield OpenBookQA train examples as A-D MC rows without fact1 leakage."""
    from datasets import load_dataset

    logger.info("Downloading allenai/openbookqa main train from HuggingFace…")
    rows = [dict(row) for row in load_dataset("allenai/openbookqa", "main", split="train")]
    rows = _maybe_cap_rows(rows, max_samples=max_samples, seed=seed, label="OpenBookQA")

    kept = 0
    for row in rows:
        texts, labels = _choice_texts_and_labels(row)
        if len(texts) != 4:
            continue
        answer = _normalise_choice_answer(row.get("answerKey"), labels)
        if answer is None:
            continue
        question = str(row.get("question_stem") or "")
        kept += 1
        yield {
            "instruction": _format_mc_instruction(question, texts),
            "input": "",
            "output": answer,
            "raw_prompt": _format_mc_prompt(question, texts, prefix="Question:"),
        }

    logger.info("OpenBookQA: kept %d train multiple-choice pairs", kept)


def _load_sciq(max_samples: int | None, seed: int) -> Iterable[dict[str, str]]:
    """Yield SciQ train examples with deterministic A-D answer shuffling."""
    from datasets import load_dataset

    logger.info("Downloading allenai/sciq train from HuggingFace…")
    rows = [dict(row) for row in load_dataset("allenai/sciq", split="train")]
    rows = _maybe_cap_rows(rows, max_samples=max_samples, seed=seed, label="SciQ")

    kept = 0
    for idx, row in enumerate(rows):
        correct = str(row.get("correct_answer") or "").strip()
        options = [
            correct,
            str(row.get("distractor1") or "").strip(),
            str(row.get("distractor2") or "").strip(),
            str(row.get("distractor3") or "").strip(),
        ]
        if not correct or any(not option for option in options):
            continue
        rng = random.Random(f"{seed}:sciq:{idx}:{row.get('question', '')}")
        shuffled = list(options)
        rng.shuffle(shuffled)
        question = str(row.get("question") or "")
        answer = _MC_LETTERS[shuffled.index(correct)]
        kept += 1
        yield {
            "instruction": _format_mc_instruction(question, shuffled),
            "input": "",
            "output": answer,
            "raw_prompt": _format_mc_prompt(question, shuffled, prefix="Question:"),
        }

    logger.info("SciQ: kept %d train multiple-choice pairs", kept)


def _load_winogrande(max_samples: int | None, seed: int) -> Iterable[dict[str, str]]:
    """Yield WinoGrande debiased train examples as A/B completion choices."""
    from datasets import load_dataset

    logger.info("Downloading allenai/winogrande winogrande_debiased train from HuggingFace…")
    rows = [dict(row) for row in load_dataset("allenai/winogrande", "winogrande_debiased", split="train")]
    rows = _maybe_cap_rows(rows, max_samples=max_samples, seed=seed, label="WinoGrande")

    kept = 0
    for row in rows:
        answer = str(row.get("answer") or "").strip()
        if answer not in {"1", "2"}:
            continue
        sentence = str(row.get("sentence") or "")
        options = [str(row.get("option1") or ""), str(row.get("option2") or "")]
        gold = "A" if answer == "1" else "B"
        kept += 1
        yield {
            "instruction": _format_mc_instruction(sentence, options, letters=("A", "B")),
            "input": "",
            "output": gold,
            "raw_prompt": _format_mc_prompt(
                sentence, options,
                prefix="Context:",
                letters=("A", "B"),
                instruction_line='Respond with only "A" or "B".',
            ),
        }

    logger.info("WinoGrande: kept %d train multiple-choice pairs", kept)


def _load_hellaswag(max_samples: int | None, seed: int) -> Iterable[dict[str, str]]:
    """Yield HellaSwag train examples as A-D MC rows with an internal default cap."""
    from datasets import load_dataset

    logger.info("Downloading Rowan/hellaswag train from HuggingFace…")
    rows = [dict(row) for row in load_dataset("Rowan/hellaswag", split="train")]
    rows = _maybe_cap_rows(
        rows,
        max_samples=max_samples,
        seed=seed,
        label="HellaSwag",
        internal_cap=6000,
    )

    kept = 0
    for row in rows:
        endings = [str(ending or "").strip() for ending in row.get("endings", [])]
        if len(endings) != 4 or any(not ending for ending in endings):
            continue
        label = str(row.get("label") or "").strip()
        if not label.isdigit() or int(label) not in range(4):
            continue
        ctx = " ".join(
            part
            for part in (
                str(row.get("ctx_a") or "").strip(),
                str(row.get("ctx_b") or "").strip(),
            )
            if part
        )
        kept += 1
        yield {
            "instruction": _format_mc_instruction(ctx, endings),
            "input": "",
            "output": _MC_LETTERS[int(label)],
            "raw_prompt": _format_mc_prompt(ctx, endings, prefix="Context:"),
        }

    logger.info("HellaSwag: kept %d train multiple-choice pairs", kept)


def _ascii_ratio(text: str) -> float:
    return sum(ord(ch) < 128 for ch in text) / max(len(text), 1)


def _lambada_like_candidate_from_sentences(
    sentences: list[str],
    *,
    rng: random.Random,
    drop_counts: dict[str, int],
) -> dict[str, str] | None:
    if not sentences:
        return None

    start_indices = list(range(len(sentences)))
    offset = rng.randrange(len(start_indices))
    start_indices = start_indices[offset:] + start_indices[:offset]

    for start in start_indices:
        passage_sentences: list[str] = []
        word_count = 0
        for sentence in sentences[start:]:
            sentence_words = _RE_ALPHA_WORD.findall(sentence)
            if not sentence_words:
                continue
            if word_count + len(sentence_words) > 150:
                break
            passage_sentences.append(sentence)
            word_count += len(sentence_words)
            if word_count >= 50:
                break

        if not 50 <= word_count <= 150 or not passage_sentences:
            continue

        passage = " ".join(passage_sentences).strip()
        target_match = _RE_TRAILING_WORD.search(passage)
        if not target_match:
            continue

        target_original = target_match.group(1)
        target = target_original.lower()
        if not target_original.isalpha():
            continue
        if len(target) < 4:
            drop_counts["too_short"] += 1
            continue
        if target in _LAMBADA_LIKE_STOPWORDS:
            drop_counts["stopword"] += 1
            continue

        prefix = passage[: target_match.start(1)]
        previous_words = [word.lower() for word in _RE_ALPHA_WORD.findall(prefix)]
        seen_before = target in previous_words
        content_word = (
            target.endswith(_LAMBADA_LIKE_CONTENT_SUFFIXES)
            or target_original[:1].isupper()
        )
        if not (seen_before or content_word):
            drop_counts["no_content_word"] += 1
            continue

        context = prefix.rstrip()
        context = context.rstrip(",;:.!?").rstrip()
        if not context or context[-1] in ",;:.!?":
            continue

        return {
            "instruction": _LAMBADA_LIKE_INSTRUCTION,
            "input": context,
            "output": target,
        }

    return None


def _load_lambada_like(max_samples: int | None, seed: int) -> Iterable[dict[str, str]]:
    """Yield OpenWebText long-context cloze rows with one-word lowercase targets."""
    from datasets import load_dataset

    cap = max_samples if max_samples is not None and max_samples > 0 else 5000
    cap = min(cap, 5000)
    rng = random.Random(seed)
    logger.info("Streaming Skylion007/openwebtext train from HuggingFace for lambada_like…")
    ds = load_dataset(
        "Skylion007/openwebtext",
        split="train",
        streaming=True,
        trust_remote_code=False,
    )
    if hasattr(ds, "shuffle"):
        ds = ds.shuffle(seed=seed, buffer_size=10_000)

    drop_counts = {"too_short": 0, "stopword": 0, "no_content_word": 0}
    kept_rows: list[dict[str, str]] = []

    for row in ds:
        raw_text = str(row.get("text") or "").strip()
        if not raw_text or _RE_ARTICLE_HEADER.match(raw_text):
            continue
        if _ascii_ratio(raw_text) <= 0.95:
            continue

        sentences = [
            sentence.strip()
            for sentence in _RE_SENTENCE_SPLIT.split(raw_text)
            if sentence.strip()
        ]
        candidate = _lambada_like_candidate_from_sentences(
            sentences,
            rng=rng,
            drop_counts=drop_counts,
        )
        if candidate is None:
            continue

        kept_rows.append(candidate)
        if len(kept_rows) >= cap:
            break

    logger.info(
        "lambada_like: filter drops from openwebtext: too_short=%d, stopword=%d, no_content_word=%d",
        drop_counts["too_short"],
        drop_counts["stopword"],
        drop_counts["no_content_word"],
    )
    logger.info("lambada_like: kept %d cloze examples from openwebtext", len(kept_rows))
    yield from kept_rows


# Mapping name → loader.  To add another dataset, drop in a loader with the
# same yield-shape and register it here.
DATASET_LOADERS: dict[str, Callable[[int | None, int], Iterable[dict[str, str]]]] = {
    "alpaca": _load_alpaca,
    "arc": _load_arc,
    "dolly": _load_dolly,
    "hellaswag": _load_hellaswag,
    "lambada_like": _load_lambada_like,
    "oasst1": _load_oasst1,
    "openbookqa": _load_openbookqa,
    "sciq": _load_sciq,
    "winogrande": _load_winogrande,
}


def _normalise_answer_text(text: str) -> str:
    """Return a compact comparison key for answer-text matching."""
    text = clean_text(text).lower()
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    text = re.sub(r"[.!,;:]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_mc_choice_options(text: str) -> dict[str, str]:
    """Extract labelled A-D choices from a prompt-like text block."""
    matches = list(_RE_MC_CHOICE_LABEL.finditer(text or ""))
    options: dict[str, str] = {}
    for idx, match in enumerate(matches):
        letter = (match.group(1) or match.group(2) or "").upper()
        if letter not in _MC_LETTERS or letter in options:
            continue
        next_start = (
            matches[idx + 1].start()
            if idx + 1 < len(matches)
            else len(text)
        )
        option_text = (text[match.end() : next_start] or "").strip()
        if option_text:
            options[letter] = option_text
    return options


def _mc_letters_for_prompt(instruction: str, input_text: str) -> set[str]:
    """Return the allowed answer letters if the row looks like an MC prompt."""
    prompt_text = "\n".join(part for part in (instruction, input_text) if part)
    options = _extract_mc_choice_options(prompt_text)
    if len(options) >= 2:
        return set(options)
    if _RE_EXPLICIT_MC_PROMPT.search(prompt_text):
        return set(_MC_LETTERS)
    return set()


def _decode_mc_answer_letter(
    output: str,
    valid_letters: set[str],
    choice_options: dict[str, str],
) -> str | None:
    """Decode a verbose MC answer into its A-D letter, if possible."""
    valid_letters = valid_letters or set(_MC_LETTERS)

    def _valid(match: re.Match[str]) -> str | None:
        letter = match.group(1).upper()
        return letter if letter in valid_letters else None

    exact = _RE_LETTER_ONLY_ANSWER.match(output)
    if exact:
        return _valid(exact)

    for pattern in _MC_ANSWER_LETTER_PATTERNS:
        match = pattern.search(output)
        if match:
            letter = _valid(match)
            if letter:
                return letter

    leading = _RE_LEADING_MC_LETTER.match(output)
    if leading:
        return _valid(leading)

    answer_text = _normalise_answer_text(output)
    normalised_options = {
        letter: _normalise_answer_text(option)
        for letter, option in choice_options.items()
    }
    for letter, option_text in sorted(
        normalised_options.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    ):
        if (
            letter in valid_letters
            and option_text
            and (answer_text == option_text or answer_text.startswith(f"{option_text} "))
        ):
            return letter

    return None


def _clean_and_filter_sft_row(
    row: dict[str, str],
    *,
    filter_code_and_math: bool = True,
) -> dict[str, str] | None:
    """Return a cleaned SFT row, or ``None`` if it is outside the target domain."""
    instruction = clean_text(
        str(row.get("instruction") or ""),
        strip_code_and_math=filter_code_and_math,
    )
    input_text = clean_text(
        str(row.get("input") or ""),
        strip_code_and_math=filter_code_and_math,
    )
    output = clean_text(
        str(row.get("output") or ""),
        strip_code_and_math=filter_code_and_math,
    )

    choice_options = _extract_mc_choice_options(f"{instruction}\n{input_text}")
    valid_letters = _mc_letters_for_prompt(instruction, input_text)
    if valid_letters:
        output = _decode_mc_answer_letter(output, valid_letters, choice_options) or ""

    if not is_clean_instruction_example(
        instruction,
        input_text,
        output,
        filter_code_and_math=filter_code_and_math,
    ):
        return None

    if decontam.is_contaminated(instruction, input_text):
        decontam.record_contamination(instruction, input_text)
        return None

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }


# ---------------------------------------------------------------------------
# Build + cache the tokenised SFT dataset
# ---------------------------------------------------------------------------


def _cache_paths(cache_dir: Path, dataset_key: str) -> tuple[Path, Path]:
    base = cache_dir / dataset_key
    return base / "train.pt", base / "val.pt"


def _normalise_dataset_weights(
    datasets: list[str],
    dataset_weights: dict[str, float] | None,
) -> dict[str, float]:
    """Return validated per-dataset weights with defaults filled to 1.0."""
    weights = {name: 1.0 for name in datasets}
    if not dataset_weights:
        return weights

    unknown = sorted(set(dataset_weights) - set(datasets))
    if unknown:
        raise ValueError(
            f"dataset_weights contains unknown dataset keys: {unknown}. "
            f"Configured datasets: {datasets}"
        )

    for name, value in dataset_weights.items():
        weight = float(value)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(
                f"dataset_weights[{name!r}] must be a positive finite number, got {value!r}"
            )
        weights[name] = weight

    return weights


def _dataset_weights_cache_suffix(weights: dict[str, float]) -> str:
    """Return a deterministic, filesystem-safe cache suffix for dataset weights."""
    canonical = "+".join(f"{name}={weights[name]:.12g}" for name in sorted(weights))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"_W{digest}"


def _split_dataset_examples(
    examples: list[SFTExample],
    *,
    val_fraction: float,
    seed: int,
    dataset_name: str,
) -> tuple[list[SFTExample], list[SFTExample]]:
    """Split one dataset into train/val before any train-time weighting."""
    shuffled = list(examples)
    random.Random(f"{seed}:{dataset_name}:split").shuffle(shuffled)

    if not shuffled or val_fraction <= 0:
        return shuffled, []

    n_val = max(1, int(len(shuffled) * val_fraction))
    if len(shuffled) > 1:
        n_val = min(n_val, len(shuffled) - 1)

    return shuffled[n_val:], shuffled[:n_val]


def _apply_dataset_weight(
    examples: list[SFTExample],
    *,
    weight: float,
    seed: int,
    dataset_name: str,
) -> list[SFTExample]:
    """Scale one dataset's train split via deterministic over/under-sampling."""
    if not examples:
        return []

    target_count = max(1, int(round(len(examples) * weight)))
    full_copies, remainder = divmod(target_count, len(examples))

    weighted: list[SFTExample] = list(examples) * full_copies
    if remainder:
        sampled = list(examples)
        random.Random(f"{seed}:{dataset_name}:weight").shuffle(sampled)
        weighted.extend(sampled[:remainder])

    return weighted


def prepare_sft_dataset(
    datasets: list[str],
    cache_dir: str | Path,
    *,
    dataset_weights: dict[str, float] | None = None,
    filter_code_and_math: bool = True,
    max_seq_len: int = 1024,
    val_fraction: float = 0.02,
    max_samples: int | None = None,
    seed: int = 42,
    force: bool = False,
) -> tuple[list[SFTExample], list[SFTExample]]:
    """Prepare (or load cached) tokenised SFT train/val splits.

    The cache key encodes the dataset list, weights, max_seq_len and
    max_samples so that changing any of these invalidates the cache.
    """
    cache_dir = Path(cache_dir)
    weights = _normalise_dataset_weights(datasets, dataset_weights)
    filter_suffix = "codefilter_on" if filter_code_and_math else "codefilter_off"
    dataset_key = (
        f"v5_clean_mc_letters_{filter_suffix}_{'+'.join(sorted(datasets))}"
        f"{_dataset_weights_cache_suffix(weights)}"
        f"_L{max_seq_len}"
        f"_N{max_samples if max_samples else 'all'}"
    )
    train_path, val_path = _cache_paths(cache_dir, dataset_key)

    if not force and train_path.exists() and val_path.exists():
        logger.info(
            "[sft] Using cached tokenised SFT data at %s",
            train_path.parent,
        )
        train = [
            SFTExample(**d) for d in torch.load(train_path, weights_only=False)
        ]
        val = [
            SFTExample(**d) for d in torch.load(val_path, weights_only=False)
        ]
        logger.info(
            "[sft] Loaded %d train + %d val examples from cache",
            len(train),
            len(val),
        )
        return train, val

    enc = _get_tokenizer()
    train_examples: list[SFTExample] = []
    val_examples: list[SFTExample] = []
    dropped = 0
    decontam.reset_decontam_counters()

    for name in datasets:
        if name not in DATASET_LOADERS:
            raise ValueError(
                f"Unknown SFT dataset: {name!r}. "
                f"Registered: {sorted(DATASET_LOADERS)}"
            )
        logger.info("[sft] Tokenising %s …", name)
        raw_examples: list[SFTExample] = []
        source_dropped = 0
        source_truncated = 0
        source_filtered = 0
        for row in DATASET_LOADERS[name](max_samples, seed):
            row = _clean_and_filter_sft_row(
                row,
                filter_code_and_math=filter_code_and_math,
            )
            if row is None:
                source_filtered += 1
                continue
            ex = tokenise_pair(
                enc,
                row["instruction"],
                row.get("input", ""),
                row["output"],
                max_seq_len=max_seq_len,
                raw_prompt=row.get("raw_prompt"),
            )
            if ex is None:
                dropped += 1
                source_dropped += 1
                continue
            ex.source = name
            if ex.truncated:
                source_truncated += 1
            raw_examples.append(ex)

        base_train, base_val = _split_dataset_examples(
            raw_examples,
            val_fraction=val_fraction,
            seed=seed,
            dataset_name=name,
        )
        weighted_train = _apply_dataset_weight(
            base_train,
            weight=weights[name],
            seed=seed,
            dataset_name=name,
        )
        train_examples.extend(weighted_train)
        val_examples.extend(base_val)
        logger.info(
            "[sft] %s: %d tokenised -> %d train / %d val before weighting -> %d train after weight %.3g",
            name,
            len(raw_examples),
            len(base_train),
            len(base_val),
            len(weighted_train),
            weights[name],
        )
        total_seen = len(raw_examples) + source_dropped
        trunc_rate = source_truncated / max(len(raw_examples), 1)
        drop_rate = source_dropped / max(total_seen, 1)
        logger.info(
            "[sft] %s quality stats: filtered=%d, truncated=%d/%d (%.2f%%), dropped=%d/%d (%.2f%%)",
            name,
            source_filtered,
            source_truncated,
            len(raw_examples),
            trunc_rate * 100,
            source_dropped,
            total_seen,
            drop_rate * 100,
        )

    if dropped:
        logger.warning(
            "[sft] Dropped %d examples that didn't fit in max_seq_len=%d or had empty output",
            dropped,
            max_seq_len,
        )

    decontam_counts = decontam.get_decontam_counters()
    if decontam_counts:
        logger.info("[sft] Public-eval decontamination drops: %s", decontam_counts)
        lambada_drops = decontam_counts.get("lambada_test", 0)
        if lambada_drops:
            logger.warning(
                "[sft] LAMBADA decontamination dropped %d rows; expected near zero for lambada_like/openwebtext",
                lambada_drops,
            )

    rng = random.Random(seed)
    rng.shuffle(train_examples)
    rng.shuffle(val_examples)
    logger.info(
        "[sft] Built %d train + %d val examples (val_fraction=%.2f%%; val is unweighted)",
        len(train_examples),
        len(val_examples),
        val_fraction * 100,
    )

    train_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save([ex.__dict__ for ex in train_examples], train_path)
    torch.save([ex.__dict__ for ex in val_examples], val_path)
    logger.info("[sft] Cached tokenised data to %s", train_path.parent)

    return train_examples, val_examples


# ---------------------------------------------------------------------------
# torch Dataset + collator
# ---------------------------------------------------------------------------


class SFTDataset(Dataset):
    """In-memory dataset of tokenised SFT examples.

    ``__getitem__`` returns the raw ``SFTExample``; padding and target-mask
    construction are done by :func:`sft_collate_fn` so batches can have
    varying prompt lengths without wasteful pre-padding to ``max_seq_len``.
    """

    def __init__(self, examples: list[SFTExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SFTExample:
        return self.examples[idx]


def sft_collate_fn(
    batch: list[SFTExample],
    pad_token_id: int = EOS_TOKEN_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a batch and build input/target tensors.

    The loss is computed via ``F.cross_entropy(..., ignore_index=-100)`` so
    both prompt tokens and pad tokens carry ``-100`` in the target.

    Given a tokenised sequence ``tokens`` of length ``L`` with prompt length
    ``k``:
      - ``input  = tokens[:-1]`` (length ``L-1``)
      - ``target = tokens[1:]``  (length ``L-1``) with ``target[:k-1] = -100``

    Batches are right-padded to the max ``L-1`` in the batch; pad positions in
    target are ``-100``.
    """
    # We predict ``L-1`` positions for a sequence of length ``L``.
    lengths = [len(ex.input_ids) - 1 for ex in batch]
    max_len = max(lengths)

    B = len(batch)
    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    targets = torch.full((B, max_len), -100, dtype=torch.long)

    for i, ex in enumerate(batch):
        ids = ex.input_ids
        L = len(ids)
        k = ex.prompt_len  # prompt length in tokens

        inp = ids[:-1]  # length L-1
        tgt = ids[1:]   # length L-1

        input_ids[i, : L - 1] = torch.tensor(inp, dtype=torch.long)
        # Mask prompt positions in target (first k-1 predictions are of prompt tokens).
        # The position at index ``k-1`` predicts the first *response* token — keep it.
        tgt_tensor = torch.tensor(tgt, dtype=torch.long)
        if k - 1 > 0:
            tgt_tensor[: k - 1] = -100
        targets[i, : L - 1] = tgt_tensor
        # Positions L-1 … max_len-1 remain ``-100`` (pad).

    return input_ids, targets


__all__ = [
    "DATASET_LOADERS",
    "EOS_TOKEN_ID",
    "SFTDataset",
    "SFTExample",
    "format_alpaca_prompt",
    "prepare_sft_dataset",
    "sft_collate_fn",
    "tokenise_pair",
]
