"""Positional encoding: RoPE and learned embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Applies rotation to query/key pairs to encode relative position information.
    The rotation encodes *relative* position, enabling better length generalisation
    than learned absolute embeddings.

    See: https://arxiv.org/abs/2104.09864

    Args:
        d_model: Model dimension (must be divisible by n_heads)
        n_heads: Number of attention heads
        max_seq_len: Maximum sequence length (cache is extended on-the-fly if exceeded)
        theta: Base frequency for the rotation (default: 10 000)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 2048,
        theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.head_dim = d_model // n_heads

        # Inverse frequencies: θ_i = 1 / (theta^(2i / d_head)) for i in [0, d_head/2)
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-build cos/sin cache for the initial max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Pre-compute cos and sin tables for positions [0, seq_len)."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, head_dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of the last dimension: [x1, x2] → [-x2, x1]."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, position_offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor, shape (batch, n_heads, seq_len, head_dim)
            k: Key tensor, shape (batch, n_heads, seq_len, head_dim)
            position_offset: First sequence position represented by ``q``/``k``.

        Returns:
            Rotated (q, k) with same shapes.
        """
        seq_len = q.shape[2]
        end_pos = position_offset + seq_len

        # Extend cache on the fly if the sequence is longer than what was pre-built
        if end_pos > self.cos_cached.shape[0]:
            self._build_cache(end_pos)

        # Broadcast: (1, 1, seq_len, head_dim)
        cos = self.cos_cached[position_offset:end_pos].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[position_offset:end_pos].unsqueeze(0).unsqueeze(0)

        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)

        return q_rotated, k_rotated


class LearnedPositionalEmbedding(nn.Module):
    """Learned absolute positional embeddings (GPT-2 style).

    Each position gets its own trainable vector that is *added* to the token
    embeddings before the transformer blocks.

    Args:
        context_length: Maximum sequence length
        d_model: Embedding dimension
    """

    def __init__(self, context_length: int, d_model: int) -> None:
        super().__init__()
        self.pos_embedding = nn.Embedding(context_length, d_model)

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        """Add positional embeddings to input.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
            position_offset: First absolute position to embed.

        Returns:
            Tensor with positional information added, same shape.
        """
        seq_len = x.shape[1]
        end_pos = position_offset + seq_len
        if end_pos > self.pos_embedding.num_embeddings:
            raise ValueError(
                f"sequence position {end_pos} exceeds learned positional "
                f"embedding length {self.pos_embedding.num_embeddings}"
            )
        positions = torch.arange(position_offset, end_pos, device=x.device)
        return x + self.pos_embedding(positions)
