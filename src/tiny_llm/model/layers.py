"""Transformer building blocks: RMSNorm, FeedForward, SwiGLU, TransformerBlock."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

KVCache = tuple[torch.Tensor, torch.Tensor]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Simpler and cheaper than LayerNorm — skips the mean-centering step.

    Args:
        d_model: Feature dimension.
        eps: Epsilon for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


# ---------------------------------------------------------------------------
# Feed-forward variants
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation.

    Standard GPT-2 style: ``Linear → GELU → Linear``.

    Args:
        d_model: Input / output dimension.
        d_ff: Hidden dimension (typically ``4 × d_model``).
        dropout: Dropout rate applied after the second linear.
        bias: Whether to include bias terms in the linear layers.
    """

    def __init__(
        self, d_model: int, d_ff: int, dropout: float = 0.0, bias: bool = False
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.gelu(self.fc1(x))))


class SwiGLUFeedForward(nn.Module):
    """Position-wise feed-forward with SwiGLU activation.

    ``SiLU(gate(x)) ⊙ up(x) → down → output``

    Uses 3 weight matrices instead of 2, so ``d_ff`` should be ~⅔ of GELU ``d_ff``
    for parameter parity.  Used by Llama, Gemma, Mistral, and most 2024+ LLMs.

    See: Shazeer (2020) — *GLU Variants Improve Transformer*
         https://arxiv.org/abs/2002.05202

    Args:
        d_model: Input / output dimension.
        d_ff: Inner hidden dimension (gate / up projection size).
        dropout: Dropout rate.
        bias: Whether to include bias terms.
    """

    def __init__(
        self, d_model: int, d_ff: int, dropout: float = 0.0, bias: bool = False
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=bias)
        self.up = nn.Linear(d_model, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Single transformer decoder block with configurable normalization placement.

    Supports three normalization modes (all using RMSNorm):

    - ``"pre"``:      ``x = x + Sublayer(Norm(x))``            — GPT-2, Llama
    - ``"post"``:     ``x = Norm(x + Sublayer(x))``            — Original Transformer
    - ``"pre_post"``: ``x = PostNorm(x + Sublayer(PreNorm(x)))`` — OLMo 2

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout rate for residual paths.
        bias: Whether to use bias in linear layers.
        qk_norm: Whether to apply QK-Norm in attention.
        activation: ``"gelu"`` or ``"swiglu"``.
        norm_position: ``"pre"``, ``"post"``, or ``"pre_post"``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        bias: bool = False,
        qk_norm: bool = False,
        activation: str = "gelu",
        norm_position: str = "pre",
    ) -> None:
        super().__init__()

        if norm_position not in ("pre", "post", "pre_post"):
            raise ValueError(
                f"norm_position must be 'pre', 'post', or 'pre_post', got '{norm_position}'"
            )

        self.norm_position = norm_position

        # ---- Self-attention sublayer (lazy import to avoid circular deps) ----
        from tiny_llm.model.attention import MultiHeadAttention

        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            bias=bias,
            qk_norm=qk_norm,
        )

        # ---- Feed-forward sublayer ----
        if activation == "swiglu":
            self.ff: nn.Module = SwiGLUFeedForward(
                d_model, d_ff, dropout=dropout, bias=bias
            )
        elif activation == "gelu":
            self.ff = FeedForward(d_model, d_ff, dropout=dropout, bias=bias)
        else:
            raise ValueError(
                f"activation must be 'gelu' or 'swiglu', got '{activation}'"
            )

        # ---- Norms ----
        # Pre-norms (always created — used in "pre" and "pre_post" modes,
        # and repurposed as the sole norms in "post" mode)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # Extra post-norms for "pre_post" mode
        self.post_norm1: RMSNorm | None = None
        self.post_norm2: RMSNorm | None = None
        if norm_position == "pre_post":
            self.post_norm1 = RMSNorm(d_model)
            self.post_norm2 = RMSNorm(d_model)

        # Residual dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: nn.Module | None = None,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
        max_cache_len: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        """Apply transformer block.

        Args:
            x: Input tensor, shape ``(batch, seq_len, d_model)``.
            rope: Optional :class:`RotaryPositionalEmbedding`.
            past_key_value: Previously cached attention key/value tensors.
            use_cache: Whether to return the updated key/value cache.
            position_offset: First sequence position of the uncached ``x`` tokens.
            max_cache_len: If set, retain only the most recent cached positions.

        Returns:
            Output tensor, shape ``(batch, seq_len, d_model)``. If
            ``use_cache=True``, also returns the updated ``(key, value)`` cache.
        """
        present_key_value: KVCache | None = None

        def run_attn(attn_input: torch.Tensor) -> torch.Tensor:
            nonlocal present_key_value
            if use_cache:
                attn_out, present_key_value = self.attn(
                    attn_input,
                    rope=rope,
                    past_key_value=past_key_value,
                    use_cache=True,
                    position_offset=position_offset,
                    max_cache_len=max_cache_len,
                )
                return attn_out
            return self.attn(attn_input, rope=rope)

        if self.norm_position == "pre":
            # Pre-norm: x = x + Dropout(Sublayer(Norm(x)))
            x = x + self.resid_dropout(run_attn(self.norm1(x)))
            x = x + self.resid_dropout(self.ff(self.norm2(x)))

        elif self.norm_position == "post":
            # Post-norm: x = Norm(x + Dropout(Sublayer(x)))
            x = self.norm1(x + self.resid_dropout(run_attn(x)))
            x = self.norm2(x + self.resid_dropout(self.ff(x)))

        elif self.norm_position == "pre_post":
            # Pre-post: x = PostNorm(x + Dropout(Sublayer(PreNorm(x))))
            assert self.post_norm1 is not None and self.post_norm2 is not None
            x = self.post_norm1(
                x + self.resid_dropout(run_attn(self.norm1(x)))
            )
            x = self.post_norm2(x + self.resid_dropout(self.ff(self.norm2(x))))

        if use_cache:
            assert present_key_value is not None
            return x, present_key_value
        return x
