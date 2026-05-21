"""Multi-head causal self-attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

KVCache = tuple[torch.Tensor, torch.Tensor]


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention with optional RoPE and QK-Norm.

    Uses ``torch.nn.functional.scaled_dot_product_attention`` for an efficient
    fused kernel (FlashAttention-2 on supported hardware).

    Args:
        d_model: Model embedding dimension.
        n_heads: Number of attention heads.
        dropout: Attention dropout probability (applied only during training).
        bias: Whether to include bias in Q/K/V/O projections.
        qk_norm: Whether to apply per-head RMSNorm to Q and K before scoring.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_dropout = dropout

        # Q, K, V, Output projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias) # = 1 fully connected NN layer (weight matrix of shape (d_model, d_model) and optional bias of shape (d_model,))
        self.k_proj = nn.Linear(d_model, d_model, bias=bias) # = 1 fully connected NN layer (weight matrix of shape (d_model, d_model) and optional bias of shape (d_model,))
        self.v_proj = nn.Linear(d_model, d_model, bias=bias) # = 1 fully connected NN layer (weight matrix of shape (d_model, d_model) and optional bias of shape (d_model,))
        self.out_proj = nn.Linear(d_model, d_model, bias=bias) # = 1 fully connected NN layer (weight matrix of shape (d_model, d_model) and optional bias of shape (d_model,))

        # Optional QK-Norm — prevents attention logit explosion in deep models
        # (used by OLMo 2, Gemma 2/3). Negligible parameter cost: 2 × head_dim.
        if qk_norm:
            from tiny_llm.model.layers import RMSNorm

            self.q_norm: nn.Module | None = RMSNorm(self.head_dim)
            self.k_norm: nn.Module | None = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        rope: nn.Module | None = None,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
        max_cache_len: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        """Apply multi-head causal self-attention.

        Args:
            x: Input tensor, shape ``(batch, seq_len, d_model)``.
            rope: Optional :class:`RotaryPositionalEmbedding` module.
            past_key_value: Previously cached key/value tensors for this layer.
            use_cache: Whether to return the updated key/value cache.
            position_offset: First sequence position of the uncached ``x`` tokens.
            max_cache_len: If set, retain only the most recent cached positions.

        Returns:
            Output tensor, shape ``(batch, seq_len, d_model)``. If
            ``use_cache=True``, also returns the updated ``(key, value)`` cache.
        """
        B, S, _ = x.shape

        # Project to Q, K, V and reshape → (B, n_heads, S, head_dim)
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Optional QK-Norm (applied before RoPE, per OLMo 2 convention)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)  # type: ignore[union-attr]

        # Inject positional information via RoPE rotation
        if rope is not None:
            q, k = rope(q, k, position_offset=position_offset)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        if (
            past_key_value is not None
            and max_cache_len is not None
            and k.size(2) > max_cache_len
        ):
            k = k[:, :, -max_cache_len:, :]
            v = v[:, :, -max_cache_len:, :]

        present_key_value = (k, v)

        attn_mask = None
        is_causal = past_key_value is None
        if past_key_value is not None and S > 1:
            key_len = k.size(2)
            past_len = key_len - S
            q_pos = torch.arange(S, device=x.device).unsqueeze(1)
            k_pos = torch.arange(key_len, device=x.device).unsqueeze(0)
            attn_mask = k_pos <= (past_len + q_pos)

        # Scaled dot-product attention with causal mask (fused kernel where available)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape back → (B, S, d_model)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)

        out = self.out_proj(out)
        if use_cache:
            return out, present_key_value
        return out
