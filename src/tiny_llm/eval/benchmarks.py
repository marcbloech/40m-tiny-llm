"""Standard LM benchmarks: LAMBADA, HellaSwag, WinoGrande, OpenBookQA."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from tiny_llm.eval._scoring import (
    compute_choice_log_probs,
    generation_mc_score,
    get_encoder,
    score_winogrande_pair,
)

if TYPE_CHECKING:
    from tiny_llm.model.transformer import GPTModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HellaSwag
# ---------------------------------------------------------------------------


def evaluate_hellaswag(
    model: GPTModel,
    device: torch.device,
) -> dict:
    """Run HellaSwag (commonsense sentence completion).

    Returns:
        Dict with 'accuracy_ll', 'accuracy_gen', and 'n_examples' keys.
    """
    from datasets import load_dataset

    logger.info("Loading HellaSwag validation split...")
    ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=False)
    enc = get_encoder()

    correct_ll = 0
    correct_gen = 0
    n = len(ds)

    model.eval()
    for i, row in enumerate(ds):
        context_text = row["activity_label"] + " " + row["ctx_a"] + " " + row["ctx_b"]
        context_ids = enc.encode(context_text)

        endings = row["endings"]
        choice_ids_list = [enc.encode(" " + e) for e in endings]
        label = int(row["label"])

        log_probs = compute_choice_log_probs(
            model, context_ids, choice_ids_list, device
        )
        if max(range(len(log_probs)), key=lambda j: log_probs[j]) == label:
            correct_ll += 1

        if generation_mc_score(
            model, context_text, endings, label, device,
            constrained=True,
            context_prefix="Context:",
        ):
            correct_gen += 1

        if (i + 1) % 500 == 0:
            logger.info(
                "  HellaSwag: %d/%d (ll_acc=%.3f, gen_acc=%.3f)",
                i + 1,
                n,
                correct_ll / (i + 1),
                correct_gen / (i + 1),
            )

    return {
        "accuracy_ll": correct_ll / n,
        "accuracy_gen": correct_gen / n,
        "n_examples": n,
    }


# ---------------------------------------------------------------------------
# WinoGrande
# ---------------------------------------------------------------------------


def evaluate_winogrande(
    model: GPTModel,
    device: torch.device,
) -> dict:
    """Run WinoGrande (coreference resolution, binary choice).

    Returns:
        Dict with 'accuracy_ll', 'accuracy_gen', and 'n_examples' keys.
    """
    from datasets import load_dataset

    logger.info("Loading WinoGrande validation split...")
    ds = load_dataset(
        "allenai/winogrande",
        "winogrande_xl",
        split="validation",
        trust_remote_code=False,
    )

    correct_ll = 0
    correct_gen = 0
    n = len(ds)

    model.eval()
    for i, row in enumerate(ds):
        sentence = row["sentence"]
        option1 = row["option1"]
        option2 = row["option2"]
        label = int(row["answer"]) - 1

        log_probs = score_winogrande_pair(model, sentence, option1, option2, device)
        if max(range(2), key=lambda j: log_probs[j]) == label:
            correct_ll += 1

        if generation_mc_score(
            model, sentence.replace("_", "___"), [option1, option2], label, device,
            constrained=True,
            context_prefix="Context:",
            extra_instruction='Respond with only "A" or "B".',
        ):
            correct_gen += 1

        if (i + 1) % 500 == 0:
            logger.info(
                "  WinoGrande: %d/%d (ll_acc=%.3f, gen_acc=%.3f)",
                i + 1,
                n,
                correct_ll / (i + 1),
                correct_gen / (i + 1),
            )

    return {
        "accuracy_ll": correct_ll / n,
        "accuracy_gen": correct_gen / n,
        "n_examples": n,
    }


# ---------------------------------------------------------------------------
# OpenBookQA
# ---------------------------------------------------------------------------


def evaluate_openbookqa(
    model: GPTModel,
    device: torch.device,
) -> dict:
    """Run OpenBookQA (science QA, 4-choice).

    Returns:
        Dict with 'accuracy_ll', 'accuracy_gen', and 'n_examples' keys.
    """
    from datasets import load_dataset

    logger.info("Loading OpenBookQA test split...")
    ds = load_dataset(
        "allenai/openbookqa", "main", split="test", trust_remote_code=False
    )
    enc = get_encoder()

    correct_ll = 0
    correct_gen = 0
    n = len(ds)
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    model.eval()
    for i, row in enumerate(ds):
        question = row["question_stem"]
        choices_data = row["choices"]
        choices_text = choices_data["text"]
        label = label_map[row["answerKey"]]

        context_text = "Question: " + question
        context_ids = enc.encode(context_text)
        choice_ids_list = [enc.encode(" " + c) for c in choices_text]

        log_probs = compute_choice_log_probs(
            model, context_ids, choice_ids_list, device
        )
        if max(range(len(log_probs)), key=lambda j: log_probs[j]) == label:
            correct_ll += 1

        if generation_mc_score(
            model, question, choices_text, label, device,
            constrained=True,
            context_prefix="Question:",
        ):
            correct_gen += 1

        if (i + 1) % 200 == 0:
            logger.info(
                "  OpenBookQA: %d/%d (ll_acc=%.3f, gen_acc=%.3f)",
                i + 1,
                n,
                correct_ll / (i + 1),
                correct_gen / (i + 1),
            )

    return {
        "accuracy_ll": correct_ll / n,
        "accuracy_gen": correct_gen / n,
        "n_examples": n,
    }


# ---------------------------------------------------------------------------
# LAMBADA
# ---------------------------------------------------------------------------


def evaluate_lambada(model: GPTModel, device: torch.device) -> dict:
    """Evaluate on LAMBADA (last-word prediction accuracy).

    Returns:
        Dict with 'accuracy_ll', 'accuracy_gen', and 'n_examples' keys.
    """
    from datasets import load_dataset

    logger.info("Loading LAMBADA (OpenAI version) test split...")
    ds = load_dataset(
        "EleutherAI/lambada_openai", "default", split="test", trust_remote_code=False
    )
    enc = get_encoder()

    correct_ll = 0
    correct_gen = 0
    n = len(ds)

    model.eval()
    for i, row in enumerate(ds):
        text = row["text"]
        # Split: context = all but last word, target = last word
        words = text.rsplit(" ", 1)
        if len(words) < 2:
            n -= 1
            continue

        context_text = words[0]
        target_word = words[1]

        context_ids = enc.encode(context_text)
        target_ids = enc.encode(" " + target_word)

        # Log-likelihood: check if model assigns highest probability to the target
        # For LAMBADA, "accuracy" = does the model predict the exact last word?
        # We check by greedy-generating len(target_ids) tokens and comparing
        max_seq = model.context_length
        if len(context_ids) > max_seq:
            context_ids = context_ids[-max_seq:]

        input_tensor = torch.tensor([context_ids], dtype=torch.long, device=device)

        # Greedy prediction of next token(s)
        predicted_ids = []
        current_input = input_tensor
        with torch.no_grad():
            for _ in range(len(target_ids)):
                if current_input.size(1) > max_seq:
                    current_input = current_input[:, -max_seq:]
                out_logits, _ = model(current_input)
                next_id = torch.argmax(out_logits[0, -1, :]).item()
                predicted_ids.append(next_id)
                current_input = torch.cat(
                    [current_input, torch.tensor([[next_id]], device=device)], dim=1
                )

        # LL accuracy: does greedy generation match the target?
        if predicted_ids == target_ids:
            correct_ll += 1

        # Generation accuracy: same as LL for LAMBADA (both check exact match)
        predicted_text = enc.decode(predicted_ids).strip()
        if predicted_text == target_word:
            correct_gen += 1

        if (i + 1) % 500 == 0:
            logger.info(
                "  LAMBADA: %d/%d (ll_acc=%.3f, gen_acc=%.3f)",
                i + 1,
                n,
                correct_ll / (i + 1),
                correct_gen / (i + 1),
            )

    return {
        "accuracy_ll": correct_ll / n if n > 0 else 0.0,
        "accuracy_gen": correct_gen / n if n > 0 else 0.0,
        "n_examples": n,
    }


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------


def run_all_benchmarks(model: GPTModel, device: torch.device) -> dict:
    """Run all benchmarks and return aggregated results.

    Returns:
        Dict mapping benchmark_name → result_dict.
    """
    results = {}

    benchmarks = [
        ("hellaswag", evaluate_hellaswag),
        ("winogrande", evaluate_winogrande),
        ("openbookqa", evaluate_openbookqa),
        ("lambada", evaluate_lambada),
    ]

    for name, fn in benchmarks:
        logger.info("--- Running %s ---", name)
        results[name] = fn(model, device)
        logger.info("  %s results: %s", name, results[name])

    return results
