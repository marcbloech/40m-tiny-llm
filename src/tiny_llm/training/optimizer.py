"""Optimizer and learning-rate scheduler factories.

Includes:
- **AdamW** — universal default (PyTorch built-in).
- **Lion** — memory-efficient sign-based optimizer (Chen et al., 2023).
- **Muon** — momentum with Newton-Schulz orthogonalization (experimental).
- **Cosine** scheduler with linear warmup.
- **WSD** (Warmup-Stable-Decay) scheduler.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lion optimizer (Chen et al., 2023 — "Symbolic Discovery of Optimization Algorithms")
# ---------------------------------------------------------------------------


class Lion(torch.optim.Optimizer):
    r"""Lion optimizer — uses the *sign* of an interpolated momentum update.

    Significantly more memory-efficient than Adam-family optimizers because it
    only stores one momentum buffer (vs. two for Adam).

    Note: Lion typically needs **lower learning rates** and **higher weight
    decay** than AdamW.  A good starting point is ``lr = adamw_lr / 3`` and
    ``weight_decay = adamw_wd * 3``.

    Reference: https://arxiv.org/abs/2302.06675

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (default: 1e-4).
        betas: Momentum interpolation coefficients (default: (0.9, 0.99)).
        weight_decay: Decoupled weight decay (default: 0.0).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                # Decoupled weight decay
                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                # Update = sign(beta1 * momentum + (1 - beta1) * grad)
                update = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1)
                p.add_(update.sign_(), alpha=-lr)

                # Update momentum buffer
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss


# ---------------------------------------------------------------------------
# Muon optimizer (experimental — Newton-Schulz orthogonalization)
# ---------------------------------------------------------------------------


class Muon(torch.optim.Optimizer):
    r"""Muon optimizer — momentum with Newton-Schulz orthogonalization.

    Applies orthogonalized momentum updates to 2-D weight matrices (the
    Newton-Schulz iteration approximates the orthogonal polar factor).
    Non-2-D parameters (biases, norms, embeddings) receive plain momentum SGD.

    **Experimental.** At our 40 M scale the benefits are uncertain.  For
    production Muon training one would typically pair Muon (2-D weights) with
    AdamW (everything else) at different learning rates.  This simplified
    implementation uses a single LR for convenience.

    Based on: https://github.com/KellerJordan/Muon

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (default: 0.02 — much higher than AdamW).
        momentum: Momentum coefficient (default: 0.95).
        nesterov: Use Nesterov-style update (default: True).
        ns_steps: Number of Newton-Schulz iterations (default: 5).
        weight_decay: Decoupled weight decay (default: 0.0).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    # ---- Newton-Schulz orthogonalization ----

    @staticmethod
    def _orthogonalize(M: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """Approximate orthogonal polar factor via cubic Newton-Schulz iteration.

        Given matrix M, iteratively converges toward the closest orthogonal
        matrix (in Frobenius norm).  Only applied to 2-D tensors.
        """
        if M.dim() != 2:
            return M

        a, b = M.shape
        transposed = a > b
        if transposed:
            M = M.T

        X = M.float()
        norm = X.norm()
        X = X / (norm + 1e-7)

        for _ in range(steps):
            A = X @ X.T
            X = (3.0 * X - A @ X) / 2.0

        if transposed:
            X = X.T

        return (X * norm).to(M.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(grad)

                if nesterov:
                    update = grad + mu * buf
                else:
                    update = buf.clone()

                # Orthogonalize for 2-D weight matrices
                if p.dim() == 2 and min(p.shape) > 1:
                    update = self._orthogonalize(update, ns_steps)

                # Decoupled weight decay
                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                p.add_(update, alpha=-lr)

        return loss


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] | list[float] = (0.9, 0.95),
) -> torch.optim.Optimizer:
    """Create an optimizer with proper weight-decay parameter grouping.

    Weight decay is applied only to parameters with ``dim >= 2`` (weight
    matrices).  Biases, norm weights, and embeddings are excluded.

    Args:
        model: The model whose parameters to optimize.
        optimizer_name: ``"adamw"``, ``"lion"``, or ``"muon"``.
        learning_rate: Peak learning rate.
        weight_decay: Weight decay coefficient.
        betas: Beta / momentum coefficients (interpretation depends on optimizer).

    Returns:
        Configured optimizer instance.
    """
    betas_tuple = tuple(betas) if isinstance(betas, list) else betas

    # Separate params into decay / no-decay groups
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    logger.info(
        "Optimizer '%s': %d decay params (%.2fM), %d no-decay params (%.2fM)",
        optimizer_name, len(decay_params), n_decay / 1e6,
        len(no_decay_params), n_no_decay / 1e6,
    )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_groups, lr=learning_rate, betas=betas_tuple)

    if optimizer_name == "lion":
        return Lion(param_groups, lr=learning_rate, betas=betas_tuple)

    if optimizer_name == "muon":
        momentum = betas_tuple[0] if betas_tuple else 0.95
        return Muon(param_groups, lr=learning_rate, momentum=momentum)

    raise ValueError(
        f"Unknown optimizer '{optimizer_name}'. Choose from: adamw, lion, muon"
    )


# ---------------------------------------------------------------------------
# LR scheduler factory
# ---------------------------------------------------------------------------


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    lr_schedule: str = "cosine",
    warmup_steps: int = 200,
    max_steps: int = 10000,
    min_lr_ratio: float = 0.1,
    wsd_decay_fraction: float = 0.2,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a learning-rate scheduler.

    Supported schedules:

    - ``"cosine"``: Linear warmup → cosine decay to ``min_lr_ratio × peak``.
    - ``"wsd"``:    Linear warmup → stable constant → cosine decay.

    Args:
        optimizer: The optimizer to schedule.
        lr_schedule: ``"cosine"`` or ``"wsd"``.
        warmup_steps: Number of linear warmup steps.
        max_steps: Total training steps.
        min_lr_ratio: Minimum LR as a fraction of the peak LR.
        wsd_decay_fraction: Fraction of *max_steps* used for the WSD decay phase.

    Returns:
        A :class:`torch.optim.lr_scheduler.LambdaLR` scheduler.
    """

    if lr_schedule == "cosine":

        def _cosine_lambda(step: int) -> float:
            # Linear warmup
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            # Cosine decay
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            progress = min(1.0, progress)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, _cosine_lambda)

    if lr_schedule == "wsd":
        decay_steps = int(max_steps * wsd_decay_fraction)
        stable_end = max_steps - decay_steps  # where stable phase ends

        def _wsd_lambda(step: int) -> float:
            # Phase 1: Linear warmup
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            # Phase 2: Stable (constant LR)
            if step < stable_end:
                return 1.0
            # Phase 3: Cosine decay
            decay_progress = (step - stable_end) / max(1, decay_steps)
            decay_progress = min(1.0, decay_progress)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * decay_progress)
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, _wsd_lambda)

    raise ValueError(
        f"Unknown lr_schedule '{lr_schedule}'. Choose from: cosine, wsd"
    )
