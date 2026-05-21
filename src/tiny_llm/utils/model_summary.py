"""Structurally correct model summary — drop-in replacement for torchinfo.

torchinfo traces the *forward pass*, so any module called more than once
(e.g. a shared RotaryPositionalEmbedding called once per layer, or
weight-tied embeddings) appears and is counted multiple times.

This implementation walks the *module tree* once via named_children(),
tracking parameter data pointers to guarantee each physical weight tensor
is counted exactly once regardless of how many times it is used.
"""

from __future__ import annotations

import sys
from typing import TextIO

import torch
import torch.nn as nn


# ── Shape extraction helpers ──────────────────────────────────────────────────

def _to_shape(x: object) -> str:
    """Return a compact shape string for tensors, tuples, or None."""
    if isinstance(x, torch.Tensor):
        return str(list(x.shape))
    if isinstance(x, (tuple, list)):
        parts = [_to_shape(v) for v in x if v is not None]
        return parts[0] if len(parts) == 1 else f"[{', '.join(parts)}]"
    return "—"


# ── Parameter counting ────────────────────────────────────────────────────────

def _own_params(module: nn.Module, seen: set[int]) -> tuple[int, int, int]:
    """Count parameters *directly owned* by this module (not children).

    Returns (trainable, frozen, shared_skipped).
    shared_skipped counts tensors whose data_ptr was already in `seen`.
    """
    trainable = frozen = skipped = 0
    for param in module.parameters(recurse=False):
        ptr = param.data_ptr()
        if ptr in seen:
            skipped += param.numel()
            continue
        seen.add(ptr)
        if param.requires_grad:
            trainable += param.numel()
        else:
            frozen += param.numel()
    return trainable, frozen, skipped


# ── Tree walker ───────────────────────────────────────────────────────────────

def _walk(
    module: nn.Module,
    seen: set[int],
    shapes: dict[int, dict],
    depth: int,
    max_depth: int,
    rows: list[dict],
    branch: str = "",
    is_last: bool = True,
) -> tuple[int, int]:
    """Recursively walk the module tree and collect row data.

    Returns (trainable_total, frozen_total) for this subtree.
    """
    own_t, own_f, own_skip = _own_params(module, seen)

    connector = "└─ " if is_last else "├─ "
    label = branch + connector + module.__class__.__name__

    mod_shapes = shapes.get(id(module), {})
    in_shape  = mod_shapes.get("input",  "—")
    out_shape = mod_shapes.get("output", "—")

    rows.append({
        "label":     label,
        "in_shape":  in_shape,
        "out_shape": out_shape,
        "own_params": own_t + own_f,
        "shared":    own_skip > 0,
        "depth":     depth,
        # subtree totals filled in below
        "sub_t": 0,
        "sub_f": 0,
    })
    row_idx = len(rows) - 1

    child_branch = branch + ("   " if is_last else "│  ")
    children = list(module.named_children())
    sub_t = own_t
    sub_f = own_f

    if depth < max_depth:
        for i, (_, child) in enumerate(children):
            ct, cf = _walk(
                child, seen, shapes,
                depth + 1, max_depth, rows,
                branch=child_branch,
                is_last=(i == len(children) - 1),
            )
            sub_t += ct
            sub_f += cf
    else:
        # Subtree collapsed — still need to count params correctly
        child_seen_ptrs: set[int] = set()
        for param in module.parameters():
            ptr = param.data_ptr()
            if ptr in seen or ptr in child_seen_ptrs:
                continue
            child_seen_ptrs.add(ptr)
            seen.add(ptr)
            if param.requires_grad:
                sub_t += param.numel()
            else:
                sub_f += param.numel()

    rows[row_idx]["sub_t"] = sub_t
    rows[row_idx]["sub_f"] = sub_f
    return sub_t, sub_f


# ── Shape inference via forward hooks ────────────────────────────────────────

def _collect_shapes(
    model: nn.Module,
    input_data: tuple,
    device: torch.device | None,
) -> dict[int, dict]:
    """Run one forward pass with hooks to collect first-call shapes only."""
    shapes: dict[int, dict] = {}
    hooks = []

    def make_hook(mod_id: int):
        def hook(module, inp, out):
            if mod_id not in shapes:   # first call only — ignore repeats
                shapes[mod_id] = {
                    "input":  _to_shape(inp),
                    "output": _to_shape(out),
                }
        return hook

    for mod in model.modules():
        hooks.append(mod.register_forward_hook(make_hook(id(mod))))

    try:
        tensors = []
        for t in input_data:
            if isinstance(t, torch.Tensor) and device is not None:
                t = t.to(device)
            tensors.append(t)
        with torch.no_grad():
            model(*tensors)
    finally:
        for h in hooks:
            h.remove()

    return shapes


# ── Formatting ────────────────────────────────────────────────────────────────

def _fmt(n: int) -> str:
    return f"{n:,}" if n else "—"


def _print_table(
    model_name: str,
    rows: list[dict],
    total_t: int,
    total_f: int,
    file: TextIO,
) -> None:
    W_LABEL    = 52
    W_IN       = 20
    W_OUT      = 20
    W_PARAMS   = 14
    W_TRAIN    = 10
    SEP = "─" * (W_LABEL + W_IN + W_OUT + W_PARAMS + W_TRAIN + 8)

    def row(label, in_s, out_s, params, trainable):
        return (
            f"{label:<{W_LABEL}} "
            f"{in_s:<{W_IN}} "
            f"{out_s:<{W_OUT}} "
            f"{params:>{W_PARAMS}} "
            f"{trainable:>{W_TRAIN}}"
        )

    header = row("Layer (type)", "Input shape", "Output shape", "Params", "Trainable")

    print(SEP, file=file)
    print(f"Model: {model_name}", file=file)
    print(SEP, file=file)
    print(header, file=file)
    print(SEP, file=file)

    for r in rows:
        label  = r["label"]
        params = r["sub_t"] + r["sub_f"]
        shared_tag = " *" if r["shared"] else ""
        print(
            row(
                label[:W_LABEL] + shared_tag,
                r["in_shape"][:W_IN],
                r["out_shape"][:W_OUT],
                _fmt(params) if params else "—",
                "✓" if r["sub_t"] > 0 else "✗",
            ),
            file=file,
        )

    print(SEP, file=file)
    total = total_t + total_f
    print(f"Total params:      {total:>15,}", file=file)
    print(f"Trainable params:  {total_t:>15,}", file=file)
    print(f"Non-trainable:     {total_f:>15,}", file=file)
    if total_t < total:
        shared = total - total_t - total_f
        # shared params (weight tying) reduce the unique count
        unique = total_t + total_f
        print(f"Unique params:     {unique:>15,}  (shared weights marked with *)", file=file)
    print(SEP, file=file)


# ── Public API ────────────────────────────────────────────────────────────────

def model_summary(
    model: nn.Module,
    input_data: tuple | None = None,
    max_depth: int = 4,
    device: torch.device | None = None,
    file: TextIO | None = None,
) -> dict[str, int]:
    """Print a structurally correct model summary.

    Unlike torchinfo, this walks the module tree once, so shared modules
    (RoPE, weight-tied embeddings) are never double-counted.

    Args:
        model:      The model to summarise (unwrapped).
        input_data: Optional tuple of example inputs for shape inference.
                    If None, shapes are omitted.
        max_depth:  Maximum nesting depth to display (default 4).
        device:     Device to move input_data to (default: None = no move).
        file:       Output stream (default: sys.stdout).

    Returns:
        {'total_params', 'trainable_params', 'non_trainable_params'}
    """
    if file is None:
        file = sys.stdout

    was_training = model.training
    model.eval()

    shapes: dict[int, dict] = {}
    if input_data is not None:
        shapes = _collect_shapes(model, input_data, device)

    seen: set[int] = set()
    rows: list[dict] = []

    # Walk from the root (shown as the model class name)
    root_branch = ""
    children = list(model.named_children())
    total_t = total_f = 0
    for i, (_, child) in enumerate(children):
        ct, cf = _walk(
            child, seen, shapes,
            depth=1, max_depth=max_depth, rows=rows,
            branch=root_branch,
            is_last=(i == len(children) - 1),
        )
        total_t += ct
        total_f += cf

    # Also count any direct params on the root itself
    for param in model.parameters(recurse=False):
        ptr = param.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        if param.requires_grad:
            total_t += param.numel()
        else:
            total_f += param.numel()

    _print_table(
        model_name=model.__class__.__name__,
        rows=rows,
        total_t=total_t,
        total_f=total_f,
        file=file,
    )

    if was_training:
        model.train()

    return {
        "total_params":         total_t + total_f,
        "trainable_params":     total_t,
        "non_trainable_params": total_f,
    }
