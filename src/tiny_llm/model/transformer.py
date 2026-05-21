"""GPT decoder-only transformer model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tiny_llm.model.layers import RMSNorm, TransformerBlock
from tiny_llm.model.positional import LearnedPositionalEmbedding, RotaryPositionalEmbedding

KVCache = tuple[torch.Tensor, torch.Tensor]
KVCacheList = tuple[KVCache, ...]


def _apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    repetition_penalty: float,
) -> None:
    """Apply repetition penalty in-place to logits for previously seen tokens."""
    for batch_idx in range(token_ids.size(0)):
        repeated_ids = torch.unique(token_ids[batch_idx])
        repeated_logits = logits[batch_idx, repeated_ids]
        logits[batch_idx, repeated_ids] = torch.where(
            repeated_logits < 0,
            repeated_logits * repetition_penalty,
            repeated_logits / repetition_penalty,
        )


def _apply_top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply nucleus sampling by keeping the smallest token set with mass top_p."""
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_remove = cumulative_probs > top_p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False

    remove = torch.zeros_like(sorted_remove)
    remove.scatter_(dim=-1, index=sorted_indices, src=sorted_remove)
    return logits.masked_fill(remove, float("-inf"))


class GPTModel(nn.Module):
    """Decoder-only GPT model with configurable architecture.

    Supports RoPE or learned absolute positional embeddings, GELU or SwiGLU
    feed-forward, pre-/post-/pre-post-normalization, optional QK-Norm, optional
    weight tying, and configurable bias.

    Args:
        config: Dictionary with model hyperparameters. Expected keys:

            ========== ====== ==========================================
            Key        Type   Description
            ========== ====== ==========================================
            vocab_size  int    Vocabulary size (default 50 257 for GPT-2)
            context_length int Maximum sequence length
            n_layers    int    Number of transformer blocks
            n_heads     int    Number of attention heads
            d_model     int    Model embedding dimension
            d_ff        int    Feed-forward hidden dimension
            dropout     float  Dropout rate
            activation  str    ``"gelu"`` or ``"swiglu"``
            norm_position str  ``"pre"``, ``"post"``, ``"pre_post"``
            weight_tying bool  Share embedding & output projection
            bias        bool   Use bias in linear layers
            qk_norm     bool   Apply RMSNorm to Q/K in attention
            positional_encoding str ``"rope"`` or ``"learned"``
            rope_theta  float  RoPE base frequency
            ========== ====== ==========================================
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        # ---- Unpack config with sensible defaults ----
        vocab_size: int = config["vocab_size"]
        context_length: int = config["context_length"]
        n_layers: int = config["n_layers"]
        n_heads: int = config["n_heads"]
        d_model: int = config["d_model"]
        d_ff: int = config["d_ff"]
        dropout: float = config.get("dropout", 0.1)
        activation: str = config.get("activation", "gelu")
        norm_position: str = config.get("norm_position", "pre")
        weight_tying: bool = config.get("weight_tying", True)
        bias: bool = config.get("bias", False)
        qk_norm: bool = config.get("qk_norm", False)
        pos_enc: str = config.get("positional_encoding", "rope")
        rope_theta: float = config.get("rope_theta", 10_000.0)

        self.context_length = context_length
        self.d_model = d_model

        # ---- Token embedding ----
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # ---- Positional encoding ----
        self.rope: RotaryPositionalEmbedding | None = None
        self.pos_embedding: LearnedPositionalEmbedding | None = None

        if pos_enc == "rope":
            self.rope = RotaryPositionalEmbedding(
                d_model, n_heads, max_seq_len=context_length, theta=rope_theta,
            )
        elif pos_enc == "learned":
            self.pos_embedding = LearnedPositionalEmbedding(context_length, d_model)
        else:
            raise ValueError(
                f"positional_encoding must be 'rope' or 'learned', got '{pos_enc}'"
            )

        # ---- Dropout after input representation ----
        self.emb_dropout = nn.Dropout(dropout)

        # ---- Transformer blocks (stacked n_layers times) ----
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    bias=bias,
                    qk_norm=qk_norm,
                    activation=activation,
                    norm_position=norm_position,
                )
                for _ in range(n_layers)
            ]
        )

        # ---- Final RMSNorm before output projection ----
        # Only needed for pre-norm and pre_post-norm: the residual stream exits
        # the last block un-normalised and must be normed before the lm_head.
        # Post-norm blocks already end with a norm inside the block, so ln_f
        # would be redundant (normalising an already-normalised tensor).
        self.ln_f: RMSNorm | None = (
            RMSNorm(d_model) if norm_position in ("pre", "pre_post") else None
        )

        # ---- Language-model head ----
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # ---- Weight tying (mandatory at small scale — saves V × d_model params) ----
        if weight_tying:
            self.lm_head.weight = self.token_embedding.weight

        # ---- Weight initialization (GPT-2 convention) ----
        self.apply(self._init_weights)

        # Residual projection scaling: σ = 0.02 / √(2 × n_layers)
        # Applied to attention output projection and FFN down/fc2 projection.
        residual_std = 0.02 / math.sqrt(2 * n_layers)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc2.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-2-style weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # --------------------------------------------------------------------- #
    # Forward pass
    # --------------------------------------------------------------------- #

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_key_values: KVCacheList | None = None,
        use_cache: bool = False,
        position_offset: int = 0,
        max_cache_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[
        torch.Tensor, torch.Tensor | None, KVCacheList
    ]:
        """Forward pass.

        Args:
            idx: Input token indices, shape ``(batch, seq_len)``.
            targets: Target token indices for loss computation,
                     shape ``(batch, seq_len)``.  If *None*, loss is not computed.
            past_key_values: Optional per-layer KV cache from a previous forward.
            use_cache: Whether to return an updated per-layer KV cache.
            position_offset: First position represented by ``idx`` when caching.
            max_cache_len: If set, retain only this many positions in each cache.

        Returns:
            ``(logits, loss)`` where *logits* has shape
            ``(batch, seq_len, vocab_size)`` and *loss* is a scalar tensor or
            *None*. If ``use_cache=True``, also returns the updated cache.
        """
        # 1) Token embedding
        x = self.token_embedding(idx)  # (B, S, d_model)

        # 2) Positional signal — learned absolute adds here; RoPE applied inside attention
        if self.pos_embedding is not None:
            x = self.pos_embedding(x, position_offset=position_offset)

        x = self.emb_dropout(x)

        # 3) Transformer blocks
        present_key_values: list[KVCache] = []
        for layer_idx, block in enumerate(self.blocks):
            past = past_key_values[layer_idx] if past_key_values is not None else None
            if use_cache:
                x, present = block(
                    x,
                    rope=self.rope,
                    past_key_value=past,
                    use_cache=True,
                    position_offset=position_offset,
                    max_cache_len=max_cache_len,
                )
                present_key_values.append(present)
            else:
                x = block(x, rope=self.rope)

        # 4) Final normalization (pre-norm only — post-norm blocks normalise internally)
        if self.ln_f is not None:
            x = self.ln_f(x)

        # 5) Project to vocab logits
        logits = self.lm_head(x)  # (B, S, vocab_size)

        # 6) Optional loss computation (targets are already shifted by the dataloader)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100, #-100 is standard ignore index for language modeling loss in PyTorch (tokens with this index are ignored in loss computation
            )

        if use_cache:
            return logits, loss, tuple(present_key_values)

        return logits, loss

    # --------------------------------------------------------------------- #
    # Autoregressive generation
    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = None,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        eos_token_id: int | None = None,
        suppress_first_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation.

        Args:
            idx: Context token indices, shape ``(batch, seq_len)``.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature (0 → greedy).
            top_k: If set, only sample from the top-*k* logits.
            top_p: Nucleus sampling threshold. Values below 1.0 keep the
                smallest token set whose probability mass exceeds top_p.
            repetition_penalty: Penalty applied to tokens that already appear
                in the context. Values above 1.0 reduce repeated-token logits.
            eos_token_id: If set and batch size is 1, stop as soon as this
                token is produced. The EOS token is included in the output.
            suppress_first_token_id: If set, this token is masked out for only
                the first generated position.

        Returns:
            Token indices including generated tokens, shape
            ``(batch, seq_len + k)`` with ``k <= max_new_tokens``.
        """
        for idx_next in self.generate_stream(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
            suppress_first_token_id=suppress_first_token_id,
        ):
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

    def generate_stream(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = None,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        eos_token_id: int | None = None,
        suppress_first_token_id: int | None = None,
    ):
        """Yield one generated token tensor at a time."""
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if top_p <= 0.0 or top_p > 1.0:
            raise ValueError("top_p must be in the interval (0, 1]")

        past_key_values: KVCacheList | None = None
        cache_position = 0

        with torch.no_grad():
            for step in range(max_new_tokens):
                # Crop to context window if necessary
                idx_cond = (
                    idx
                    if idx.size(1) <= self.context_length
                    else idx[:, -self.context_length :]
                )

                if (
                    self.pos_embedding is not None
                    and past_key_values is not None
                    and cache_position >= self.context_length
                ):
                    past_key_values = None
                    cache_position = 0

                if past_key_values is None:
                    model_input = idx_cond
                    position_offset = 0
                else:
                    model_input = idx[:, -1:]
                    position_offset = cache_position

                logits, _, past_key_values = self(
                    model_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    position_offset=position_offset,
                    max_cache_len=self.context_length,
                )
                cache_position = position_offset + model_input.size(1)
                logits = logits[:, -1, :]  # last position only

                if repetition_penalty != 1.0:
                    _apply_repetition_penalty(logits, idx, repetition_penalty)

                if (
                    step == 0
                    and suppress_first_token_id is not None
                    and 0 <= int(suppress_first_token_id) < logits.size(-1)
                ):
                    logits[:, int(suppress_first_token_id)] = float("-inf")

                # Temperature scaling
                if temperature == 0.0:
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    logits = logits / temperature

                    # Optional top-k filtering
                    if top_k is not None and top_k > 0:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = float("-inf")
                    logits = _apply_top_p_filter(logits, top_p)

                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)

                yield idx_next

                # Early stop on EOS (single-sample case only).
                if (
                    eos_token_id is not None
                    and idx.size(0) == 1
                    and int(idx_next.item()) == int(eos_token_id)
                ):
                    break

                idx = torch.cat([idx, idx_next], dim=1)
