"""Text generation utilities for inference and evaluation."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from tiny_llm.model.transformer import GPTModel
from tiny_llm.tokenizer import decode_tokens, encode_text
from tiny_llm.training.checkpoint import load_checkpoint
from tiny_llm.utils.seed import set_seed

logger = logging.getLogger(__name__)


def generate(
    checkpoint_path: str | Path,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    device: str = "cpu",
    seed: int = 42,
    config_path: str | Path | None = None,
    chat_template: str = "raw",
    auto_detect_template: bool = True,
    return_full_prompt: bool = True,
) -> str:
    """Generate text from a prompt using a saved checkpoint."""
    set_seed(seed)
    dev = torch.device(device)

    meta = load_checkpoint(checkpoint_path, model=None, device=dev)
    model_config = meta["model_config"]
    if model_config is None:
        if config_path is None:
            raise ValueError(
                "Checkpoint missing model_config and no config_path provided. "
                "Pass config_path to specify the model architecture."
            )
        from tiny_llm.utils.config import load_config

        cfg = load_config(str(config_path))
        model_config = cfg.model.model_dump()
        logger.info("Using model config from %s (older checkpoint format)", config_path)

    model = GPTModel(model_config)
    load_checkpoint(checkpoint_path, model, device=dev)
    model.to(dev)
    model.eval()

    template = (chat_template or "raw").lower()
    should_auto_detect = auto_detect_template

    if template == "raw" and should_auto_detect:
        saved_template = (model_config or {}).get("chat_template", "")
        if saved_template:
            template = saved_template.lower()
            logger.debug("Auto-detected chat_template=%r from checkpoint", template)

    eos_id: int | None = None
    if template == "alpaca":
        # Match the exact SFT training format (see sft_dataset.format_alpaca_prompt).
        from tiny_llm.posttraining.sft_dataset import (
            EOS_TOKEN_ID,
            format_alpaca_prompt,
        )

        prompt_text = format_alpaca_prompt(prompt, "")
        eos_id = EOS_TOKEN_ID
    else:
        prompt_text = prompt

    prompt_ids = encode_text(prompt_text)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=dev)

    # 4) Generate (stops early on EOS when using the Alpaca template)
    output_ids = model.generate(
        idx,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_id,
    )

    # 5) Decode back to text
    output_list = output_ids[0].tolist()
    new_ids = output_list[len(prompt_ids):]
    # Strip trailing EOS if the model emitted one.
    if eos_id is not None and new_ids and new_ids[-1] == eos_id:
        new_ids = new_ids[:-1]

    if template == "alpaca":
        return decode_tokens(new_ids).strip()

    if not return_full_prompt:
        return decode_tokens(new_ids)

    return decode_tokens(output_list)
