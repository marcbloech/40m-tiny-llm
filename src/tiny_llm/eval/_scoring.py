"""Shared multiple-choice scoring helpers for the eval pipeline.

Both `tiny_llm.eval.benchmarks` (full uncapped eval) and
`scripts/quick_eval.py` (capped quick eval) route through here, so the
two paths always produce identical numbers on the same checkpoint.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from tiny_llm.tokenizer import (
    decode_tokens,
    encode_text,
    get_gpt2_tokenizer_adapter,
)

if TYPE_CHECKING:
    from tiny_llm.model.transformer import GPTModel


def get_encoder():
    return get_gpt2_tokenizer_adapter()


def compute_choice_log_probs(
    model: "GPTModel",
    context_ids: list[int],
    choice_ids_list: list[list[int]],
    device: torch.device,
) -> list[float]:
    """Return the mean per-token log-prob of each choice given the context.

    Length-normalized: summing raw log-probs over a choice span biases
    multiple-choice tasks toward shorter endings, which is a tokenization
    artifact rather than a real quality signal. Taking the mean is the
    `acc_norm` convention used by lm-evaluation-harness and cited HellaSwag
    baselines.
    """
    context_len = len(context_ids)
    max_seq = model.context_length
    results: list[float] = []

    for choice_ids in choice_ids_list:
        full_ids = context_ids + choice_ids
        if len(full_ids) > max_seq:
            trim = len(full_ids) - max_seq
            full_ids = full_ids[trim:]
            adj_context_len = context_len - trim
        else:
            adj_context_len = context_len

        input_tensor = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
        target_tensor = torch.tensor(full_ids[1:], dtype=torch.long, device=device)

        with torch.no_grad():
            logits, _ = model(input_tensor)  # (1, S, V)

        log_probs = F.log_softmax(logits[0], dim=-1)  # (S, V)

        start = max(adj_context_len - 1, 0)
        scored = log_probs[start:].gather(1, target_tensor[start:].unsqueeze(1))
        if scored.numel() == 0:
            results.append(float("-inf"))
            continue
        results.append(scored.mean().item())

    return results


@lru_cache(maxsize=16)
def get_label_token_ids(labels_key: str) -> tuple[int, ...]:
    """Return token IDs whose single-token decode maps to allowed labels.

    ``labels_key`` is a compact string like "AB" or "ABCD".
    """
    labels = {c.upper() for c in labels_key}
    enc = get_encoder()
    # Faster than 50k decode() calls: convert all vocab ids to token strings once.
    token_strs = enc.tokenizer.convert_ids_to_tokens(list(range(enc.vocab_size)))

    ids: list[int] = []
    for tok_id, tok in enumerate(token_strs):
        txt = str(tok).replace("Ġ", "").strip().upper()
        if txt in labels:
            ids.append(tok_id)
    return tuple(ids)


def argmax_with_token_constraint(
    logits_1d: torch.Tensor,
    allowed_token_ids: tuple[int, ...] | list[int] | None,
) -> int:
    """Argmax under an optional token-id allowlist constraint.

    If all constrained logits are ``-inf`` (pathological/corrupt case), falls
    back to unconstrained argmax instead of silently returning token 0.
    """
    if not allowed_token_ids:
        return int(torch.argmax(logits_1d).item())

    mask = torch.full_like(logits_1d, float("-inf"))
    idx = torch.tensor(list(allowed_token_ids), device=logits_1d.device, dtype=torch.long)
    mask[idx] = logits_1d[idx]

    best = int(torch.argmax(mask).item())
    if torch.isneginf(mask[best]):
        return int(torch.argmax(logits_1d).item())
    return best


def generation_mc_score(
    model: "GPTModel",
    context: str,
    choices: list[str],
    correct_idx: int,
    device: torch.device,
    *,
    constrained: bool = True,
    context_prefix: str = "",
    extra_instruction: str = "",
) -> bool:
    """Format as a multiple-choice prompt and check whether the model's
    next greedy token decodes to the correct letter.

    When ``constrained=True``, next-token argmax is restricted to valid answer
    letters (A/B for binary, A-D for 4-way, etc.) via logit masking.

    ``context_prefix`` (e.g. ``"Context:"`` or ``"Question:"``) is prepended
    to the context line to match the official multiple-choice prompt format.

    ``extra_instruction`` (e.g. ``'Respond with only "A" or "B".'``) is
    inserted before ``Answer:``.
    """
    letters = "ABCDEFGHIJ"

    ctx_line = f"{context_prefix} {context}" if context_prefix else context
    prompt_parts = [ctx_line]
    for i, choice in enumerate(choices):
        prompt_parts.append(f"{letters[i]}) {choice}")
    if extra_instruction:
        prompt_parts.append(extra_instruction)
    prompt_parts.append("Answer:")
    prompt_text = "\n".join(prompt_parts)

    prompt_ids = encode_text(prompt_text)

    max_seq = model.context_length
    if len(prompt_ids) >= max_seq:
        prompt_ids = prompt_ids[-(max_seq - 1) :]

    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(input_tensor)

    next_logits = logits[0, -1, :]
    allowed_ids: tuple[int, ...] | None = None
    if constrained:
        label_set = letters[: len(choices)]
        allowed_ids = get_label_token_ids(label_set)

    predicted_token = argmax_with_token_constraint(next_logits, allowed_ids)
    predicted_text = decode_tokens([predicted_token]).strip()

    correct_letter = letters[correct_idx]
    return predicted_text.upper().startswith(correct_letter)


def score_winogrande_pair(
    model: "GPTModel",
    sentence: str,
    option1: str,
    option2: str,
    device: torch.device,
) -> list[float]:
    """Score a WinoGrande pair by length-normalized log-prob.

    Splits the sentence at ``_``: the prefix becomes the context and each
    ``option + suffix`` becomes the scored target. Mirrors the
    lm-evaluation-harness WinoGrande scoring — the shared prefix is not
    scored, avoiding the BPE-boundary bias that comes from re-tokenizing
    two full sentences and comparing raw log-prob sums.
    """
    blank_idx = sentence.index("_")
    prefix = sentence[:blank_idx]
    suffix = sentence[blank_idx + 1 :]

    context_ids = encode_text(prefix)
    target1 = encode_text(option1 + suffix)
    target2 = encode_text(option2 + suffix)

    return compute_choice_log_probs(model, context_ids, [target1, target2], device)
