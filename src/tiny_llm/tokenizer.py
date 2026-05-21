"""Shared GPT-2 tokenizer helpers.

The project standardizes on ``transformers.GPT2TokenizerFast`` so all
tokenization paths (pre-training, SFT, eval, demo, tests) use the same
implementation and cached vocabulary files.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from transformers import GPT2TokenizerFast


@lru_cache(maxsize=1)
def get_gpt2_tokenizer() -> GPT2TokenizerFast:
    """Return the shared GPT-2 tokenizer instance."""
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.eos_token_id is None:
        raise ValueError("GPT-2 tokenizer must define eos_token_id.")
    return tokenizer


@dataclass(frozen=True)
class GPT2TokenizerAdapter:
    """Small adapter that preserves the repo's previous encode/decode semantics."""

    tokenizer: GPT2TokenizerFast

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: Iterable[int]) -> str:
        return self.tokenizer.decode(
            list(token_ids),
            clean_up_tokenization_spaces=False,
        )


@lru_cache(maxsize=1)
def get_gpt2_tokenizer_adapter() -> GPT2TokenizerAdapter:
    return GPT2TokenizerAdapter(get_gpt2_tokenizer())


def encode_text(text: str) -> list[int]:
    """Encode text without adding special tokens."""
    return get_gpt2_tokenizer().encode(text, add_special_tokens=False)


def decode_tokens(token_ids: Iterable[int]) -> str:
    """Decode token IDs without cleanup that would alter whitespace."""
    return get_gpt2_tokenizer().decode(
        list(token_ids),
        clean_up_tokenization_spaces=False,
    )


__all__ = [
    "GPT2TokenizerAdapter",
    "decode_tokens",
    "encode_text",
    "get_gpt2_tokenizer",
    "get_gpt2_tokenizer_adapter",
]
