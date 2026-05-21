"""Gradio chat interface for interactive text generation."""

from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from tiny_llm.model.transformer import GPTModel
from tiny_llm.tokenizer import decode_tokens, encode_text

logger = logging.getLogger(__name__)


_ALPACA_PREAMBLE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
)
_SUPPORTED_CHAT_TEMPLATES = {"raw", "alpaca", "simple"}
_DEMO_BENCHMARK_PROMPT = "What is the color of the sky?"
_NO_OUTPUT_SENTINEL = "(no output)"
_TOKENS_PER_SECOND_PREFIX = "tokens/s:"


@dataclass(frozen=True)
class LoadedModel:
    """In-memory checkpoint + generation metadata for demo use."""

    name: str
    checkpoint_path: Path
    model: GPTModel
    model_config: dict
    chat_template: str
    eos_id: int | None
    device: torch.device


def _history_turns(history: list) -> list[tuple[str, str]]:
    """Normalise Gradio history into a flat list of (user, assistant) pairs."""
    turns: list[tuple[str, str]] = []
    pending_user: str | None = None
    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            if role == "user":
                pending_user = content
            elif role == "assistant" and pending_user is not None:
                content = _strip_generation_stats(content)
                if content == _NO_OUTPUT_SENTINEL:
                    pending_user = None
                    continue
                turns.append((pending_user, content))
                pending_user = None
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            u = str(item[0]).strip()
            a = _strip_generation_stats(str(item[1]))
            if u and a != _NO_OUTPUT_SENTINEL:
                turns.append((u, a))
    return turns


def _strip_generation_stats(text: str) -> str:
    """Remove demo-only throughput footer from an assistant message."""
    lines = text.strip().splitlines()
    while lines and lines[-1].strip().lower().startswith(_TOKENS_PER_SECOND_PREFIX):
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
    return "\n".join(lines).strip()


def _format_tokens_per_second(generated_tokens: int, elapsed_seconds: float) -> str:
    if generated_tokens <= 0 or elapsed_seconds <= 0:
        return f"{_TOKENS_PER_SECOND_PREFIX} n/a"
    return f"{_TOKENS_PER_SECOND_PREFIX} {generated_tokens / elapsed_seconds:.2f}"


def _append_generation_stats(
    text: str,
    generated_tokens: int,
    elapsed_seconds: float,
) -> str:
    body = text.strip() or _NO_OUTPUT_SENTINEL
    return f"{body}\n\n{_format_tokens_per_second(generated_tokens, elapsed_seconds)}"


def _build_prompt(
    message: str,
    history: list,
    chat_template: str = "alpaca",
) -> str:
    """Build a chat-style prompt."""
    template = (chat_template or "alpaca").lower()

    if template == "raw":
        return message.strip()

    if template == "alpaca":
        prior_turns = _history_turns(history)
        parts: list[str] = [_ALPACA_PREAMBLE, ""]
        for user_text, asst_text in prior_turns:
            parts.append(f"### Instruction:\n{user_text}")
            parts.append("")
            parts.append(f"### Response:\n{asst_text}")
            parts.append("")
        parts.append(f"### Instruction:\n{message.strip()}")
        parts.append("")
        parts.append("### Response:\n")
        return "\n".join(parts).rstrip(" ")

    # Legacy "simple" template (kept for pre-SFT checkpoints).
    lines: list[str] = []
    for user_text, asst_text in _history_turns(history):
        lines.append(f"User: {user_text}")
        if asst_text:
            lines.append(f"Assistant: {asst_text}")
    lines.append(f"User: {message.strip()}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _model_name_from_path(path: Path) -> str:
    """Derive a compact model display name from a checkpoint path."""
    try:
        for parent in path.parents:
            if parent.name == "checkpoints":
                rel = path.relative_to(parent)
                return str(rel.with_suffix("")).replace("/", "_")
    except Exception:
        pass
    return path.stem


def _dedupe_paths(paths: list[str | Path]) -> list[Path]:
    seen: set[Path] = set()
    deduped: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        resolved = p.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _load_single_model(
    path: str | Path,
    device: torch.device,
    chat_template: str | None,
) -> LoadedModel:
    """Load one checkpoint into a ``LoadedModel`` record."""
    checkpoint_path = Path(path).expanduser().resolve()

    try:
        # Load checkpoint payload on CPU first to avoid temporary duplicate
        # allocations on accelerator memory when loading multiple models.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg:
            raise RuntimeError(
                "Out of memory while loading checkpoint "
                f"{checkpoint_path}. Try fewer arena models or --device cpu."
            ) from exc
        raise

    model_config = checkpoint.get("model_config")
    if model_config is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is missing model_config. "
            "The demo requires checkpoints with embedded model_config metadata."
        )

    model = GPTModel(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    try:
        model.to(device)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg:
            raise RuntimeError(
                "Out of memory while materializing checkpoint "
                f"{checkpoint_path} on {device}. Try fewer arena models or --device cpu."
            ) from exc
        raise
    model.eval()

    resolved_template = _resolve_checkpoint_template(model_config, chat_template)

    eos_id: int | None = None
    if resolved_template == "alpaca":
        from tiny_llm.posttraining.sft_dataset import EOS_TOKEN_ID

        eos_id = EOS_TOKEN_ID

    return LoadedModel(
        name=_model_name_from_path(checkpoint_path),
        checkpoint_path=checkpoint_path,
        model=model,
        model_config=model_config,
        chat_template=resolved_template,
        eos_id=eos_id,
        device=device,
    )


def _load_model_registry(
    paths: list[str | Path],
    device: torch.device,
    chat_template: str | None,
    labels: list[str] | None = None,
) -> list[LoadedModel]:
    """Load and return models for all unique checkpoint paths."""
    deduped = _dedupe_paths(paths)
    models: list[LoadedModel] = []
    for p in deduped:
        models.append(_load_single_model(p, device=device, chat_template=chat_template))

    if labels and len(labels) == len(models):
        models = [
            LoadedModel(
                name=label,
                checkpoint_path=m.checkpoint_path,
                model=m.model,
                model_config=m.model_config,
                chat_template=m.chat_template,
                eos_id=m.eos_id,
                device=m.device,
            )
            for m, label in zip(models, labels)
        ]

    # Ensure display names are unique for UI controls.
    used: dict[str, int] = {}
    unique_models: list[LoadedModel] = []
    for model in models:
        idx = used.get(model.name, 0)
        used[model.name] = idx + 1
        if idx == 0:
            unique_models.append(model)
            continue
        unique_models.append(
            LoadedModel(
                name=f"{model.name} ({idx + 1})",
                checkpoint_path=model.checkpoint_path,
                model=model.model,
                model_config=model.model_config,
                chat_template=model.chat_template,
                eos_id=model.eos_id,
                device=model.device,
            )
        )
    return unique_models


def _available_demo_benchmark_devices() -> list[torch.device]:
    """Return devices worth benchmarking for local demo inference."""
    devices = [torch.device("cpu")]
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif (
        device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "synchronize")
    ):
        torch.mps.synchronize()


def _cleanup_benchmark_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif (
        device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "empty_cache")
    ):
        torch.mps.empty_cache()


def _select_fastest_device(timings_ms: dict[str, float]) -> torch.device:
    if not timings_ms:
        raise RuntimeError("Unable to benchmark any demo device.")
    return torch.device(min(timings_ms, key=timings_ms.get))


@torch.no_grad()
def benchmark_demo_device(
    checkpoint_path: str | Path,
    chat_template: str | None,
    benchmark_tokens: int = 20,
) -> torch.device:
    """Benchmark short demo generation and return the fastest available device."""
    candidates = _available_demo_benchmark_devices()
    if len(candidates) == 1:
        return candidates[0]

    timings_ms: dict[str, float] = {}
    for device in candidates:
        loaded_model: LoadedModel | None = None
        try:
            loaded_model = _load_single_model(
                checkpoint_path,
                device=device,
                chat_template=chat_template,
            )
            # Disable EOS for benchmarking so each device performs the same
            # number of autoregressive decode steps.
            loaded_model = LoadedModel(
                name=loaded_model.name,
                checkpoint_path=loaded_model.checkpoint_path,
                model=loaded_model.model,
                model_config=loaded_model.model_config,
                chat_template=loaded_model.chat_template,
                eos_id=None,
                device=loaded_model.device,
            )
            prompt = _build_prompt(
                _DEMO_BENCHMARK_PROMPT,
                history=[],
                chat_template=loaded_model.chat_template,
            )

            torch.manual_seed(0)
            for _ in _generate_single_stream(
                loaded_model,
                prompt,
                temperature=0.8,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.1,
                max_new_tokens=4,
            ):
                pass

            _synchronize_device(device)
            started = time.perf_counter()
            torch.manual_seed(0)
            for _ in _generate_single_stream(
                loaded_model,
                prompt,
                temperature=0.8,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.1,
                max_new_tokens=max(1, int(benchmark_tokens)),
            ):
                pass
            _synchronize_device(device)
            timings_ms[str(device)] = round(
                (time.perf_counter() - started) * 1000.0,
                2,
            )
        except RuntimeError as exc:
            logger.warning("Skipping demo device %s during benchmark: %s", device, exc)
        finally:
            del loaded_model
            _cleanup_benchmark_device(device)

    selected = _select_fastest_device(timings_ms)
    logger.info(
        "Demo device benchmark (%d tokens): %s; selected %s",
        benchmark_tokens,
        timings_ms,
        selected,
    )
    return selected


def _resolve_checkpoint_template(
    model_config: dict,
    requested_template: str | None,
) -> str:
    """Resolve the prompt template for one checkpoint.

    ``None`` means auto mode: use checkpoint metadata when present, otherwise
    the legacy simple chat layout that works better for pre-SFT checkpoints.
    """
    if requested_template is not None:
        template = requested_template.lower()
        if template not in _SUPPORTED_CHAT_TEMPLATES:
            raise ValueError(f"Unsupported chat template: {requested_template!r}")
        return template

    saved_template = str(model_config.get("chat_template", "")).lower().strip()
    if not saved_template:
        return "simple"
    if saved_template not in _SUPPORTED_CHAT_TEMPLATES:
        raise ValueError(f"Unsupported checkpoint chat_template metadata: {saved_template!r}")
    return saved_template


def validate_template_compatibility(
    models: list[LoadedModel],
    chat_template: str | None,
) -> None:
    """Validate that selected checkpoints are compatible with the chosen template."""
    if chat_template is None:
        return

    template = (chat_template or "alpaca").lower()
    if template in {"raw", "simple"}:
        return
    if template != "alpaca":
        raise ValueError(f"Unsupported chat template: {chat_template!r}")

    incompatible: list[str] = []
    for m in models:
        saved_template = str(m.model_config.get("chat_template", "")).lower().strip()
        if saved_template != "alpaca":
            incompatible.append(m.name)

    if incompatible:
        raise ValueError(
            "Selected template 'alpaca' is incompatible with checkpoint(s): "
            f"{', '.join(incompatible)}. "
            "Use --chat-template simple/raw or remove pre-SFT checkpoints."
        )


def validate_compare_request(
    prompt: str,
    selected_model_names: list[str] | None,
    max_models: int,
) -> str | None:
    """Return a validation error message for compare requests, else ``None``."""
    if not (prompt or "").strip():
        return "Please enter a prompt before running compare."
    selected = list(selected_model_names or [])
    if len(selected) < 2:
        return "Select at least 2 models to compare."
    if len(selected) > max_models:
        return f"Select at most {max_models} models."
    return None


def _generate_single_stream(
    loaded_model: LoadedModel,
    prompt_text: str,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
):
    """Yield progressively decoded response text from one loaded model."""
    prompt_ids = encode_text(prompt_text)
    max_input_tokens = max(1, loaded_model.model.context_length - 1)
    if len(prompt_ids) > max_input_tokens:
        prompt_ids = prompt_ids[-max_input_tokens:]

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=loaded_model.device)
    k = int(top_k) if int(top_k) > 0 else None
    t = max(0.0, float(temperature))
    p = float(top_p)
    rp = float(repetition_penalty)
    n_new = max(1, int(max_new_tokens))

    started = time.perf_counter()
    for attempt in range(2):
        new_ids: list[int] = []
        yielded = False
        retry_immediate_eos = False

        for idx_next in loaded_model.model.generate_stream(
            idx,
            max_new_tokens=n_new,
            temperature=t,
            top_k=k,
            top_p=p,
            repetition_penalty=rp,
            eos_token_id=loaded_model.eos_id,
            suppress_first_token_id=loaded_model.eos_id,
        ):
            token_id = int(idx_next[0, 0].item())
            if loaded_model.eos_id is not None and token_id == int(loaded_model.eos_id):
                if not new_ids and attempt == 0:
                    retry_immediate_eos = True
                break
            new_ids.append(token_id)
            text = decode_tokens(new_ids).strip()
            yielded = True
            yield text

        if retry_immediate_eos:
            continue

        elapsed = time.perf_counter() - started
        yield _append_generation_stats(
            decode_tokens(new_ids).strip() if yielded else _NO_OUTPUT_SENTINEL,
            generated_tokens=len(new_ids),
            elapsed_seconds=elapsed,
        )
        return


def append_compare_log(log_path: str | Path, payload: dict) -> None:
    """Append one compare-run payload as a JSONL line."""
    p = Path(log_path)
    if p.suffix.lower() != ".jsonl":
        raise ValueError(f"arena log path must end with .jsonl, got: {p}")

    resolved = p.resolve()
    cwd = Path.cwd().resolve()
    if not resolved.is_relative_to(cwd):
        raise ValueError(
            f"arena log path must stay within the current workspace ({cwd}), got: {resolved}"
        )

    if resolved.exists() and resolved.is_dir():
        raise ValueError(f"arena log path points to a directory, expected a file: {resolved}")

    resolved.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with resolved.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def launch_demo(
    checkpoint_path: str | Path,
    device: str = "cpu",
    share: bool = False,
    chat_template: str | None = None,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    arena_checkpoints: list[str] | None = None,
    arena_labels: list[str] | None = None,
    arena_max_models: int = 4,
    arena_log_path: str | Path = "eval_outputs/arena_compare.jsonl",
) -> None:
    """Launch a Gradio web interface for interacting with one or many models."""
    dev = torch.device(device)
    template = chat_template.lower() if chat_template is not None else None

    registry_paths = [checkpoint_path, *(arena_checkpoints or [])]
    models = _load_model_registry(
        registry_paths, device=dev, chat_template=template, labels=arena_labels,
    )
    validate_template_compatibility(models, template)

    if not models:
        raise ValueError("No checkpoints provided for demo launch.")

    model_by_name = {m.name: m for m in models}
    primary_model = next((m for m in models if m.checkpoint_path == Path(checkpoint_path).resolve()), models[0])

    model_mem_mb = sum(
        p.numel() * p.element_size() for m in models for p in m.model.parameters()
    ) / (1024 * 1024)
    logger.info(
        "Loaded %d demo model(s) on %s (approx %.1f MB params).",
        len(models),
        dev,
        model_mem_mb,
    )

    @torch.no_grad()
    def chat_fn(
        message: str,
        history: list,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p_value: float = top_p,
        repetition_penalty_value: float = repetition_penalty,
        max_new_tokens: int = 100,
    ):
        prompt = _build_prompt(message, history, chat_template=primary_model.chat_template)
        yield from _generate_single_stream(
            primary_model,
            prompt,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p_value,
            repetition_penalty=repetition_penalty_value,
            max_new_tokens=max_new_tokens,
        )

    def _blank_output_updates():
        import gradio as gr

        return [gr.update(value="", label=f"Model {i + 1}", visible=False) for i in range(arena_max_models)]

    def _render_output_updates(slot_state: list[dict[str, object]]):
        import gradio as gr

        return [
            gr.update(
                value=str(slot["value"]),
                label=str(slot["label"]),
                visible=bool(slot["visible"]),
            )
            for slot in slot_state
        ]

    @torch.no_grad()
    def compare_fn(
        prompt: str,
        selected_names: list[str],
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty_value: float,
        max_new_tokens: int,
    ):
        err = validate_compare_request(prompt, selected_names, arena_max_models)
        if err:
            yield [f"Warning: {err}", *_blank_output_updates()]
            return

        selected_models: list[LoadedModel] = [
            model_by_name[name] for name in selected_names if name in model_by_name
        ]
        if len(selected_models) != len(selected_names):
            missing = [n for n in selected_names if n not in model_by_name]
            yield [f"Warning: Unknown selected model(s): {', '.join(missing)}", *_blank_output_updates()]
            return

        slot_state: list[dict[str, object]] = [
            {"value": "", "label": f"Model {i + 1}", "visible": False}
            for i in range(arena_max_models)
        ]
        timings_ms: dict[str, float] = {}
        outputs: dict[str, str] = {}

        total_work = len(selected_models) * int(max_new_tokens)
        if total_work > 1024:
            yield [
                "Info: large compare request — this may take a while "
                f"({len(selected_models)} models × {int(max_new_tokens)} tokens).",
                *_render_output_updates(slot_state),
            ]

        yield [
            f"Generating with model 1/{len(selected_models)}...",
            *_render_output_updates(slot_state),
        ]

        for idx, loaded_model in enumerate(selected_models, start=1):
            status = f"Generating with model {idx}/{len(selected_models)}: {loaded_model.name}..."
            yield [status, *_render_output_updates(slot_state)]

            started = time.perf_counter()
            model_prompt = _build_prompt(prompt, history=[], chat_template=loaded_model.chat_template)
            slot_state[idx - 1] = {
                "label": loaded_model.name,
                "value": "",
                "visible": True,
            }
            yield [status, *_render_output_updates(slot_state)]

            text = ""
            for text in _generate_single_stream(
                loaded_model,
                model_prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty_value,
                max_new_tokens=max_new_tokens,
            ):
                slot_state[idx - 1] = {
                    "label": loaded_model.name,
                    "value": text.strip() or "(no output)",
                    "visible": True,
                }
                yield [status, *_render_output_updates(slot_state)]

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            text = text.strip() or "(no output)"
            outputs[loaded_model.name] = text
            timings_ms[loaded_model.name] = round(elapsed_ms, 2)
            slot_state[idx - 1] = {
                "label": loaded_model.name,
                "value": text,
                "visible": True,
            }
            yield [status, *_render_output_updates(slot_state)]

        append_compare_log(
            arena_log_path,
            {
                "prompt": prompt,
                "decode": {
                    "temperature": float(temperature),
                    "top_k": int(top_k),
                    "top_p": float(top_p),
                    "repetition_penalty": float(repetition_penalty_value),
                    "max_new_tokens": int(max_new_tokens),
                },
                "models": [
                    {
                        "name": m.name,
                        "checkpoint": str(m.checkpoint_path),
                    }
                    for m in selected_models
                ],
                "outputs": outputs,
                "timings_ms": timings_ms,
            },
        )

        done = f"Done. Compared {len(selected_models)} model(s)."
        yield [done, *_render_output_updates(slot_state)]

    try:
        import gradio as gr
    except ImportError:
        logger.warning(
            "gradio is not installed. Falling back to terminal chat loop. "
            "Install with: uv sync --extra demo"
        )
        print("Tiny LLM CLI chat. Type /exit to quit.")
        print(
            f"Loaded {len(models)} model(s). Compare tab requires Gradio; "
            f"CLI fallback uses primary model: {primary_model.name}."
        )
        history: list[tuple[str, str]] = []
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"/exit", "exit", "quit"}:
                break
            if not user_input:
                continue
            reply = ""
            for chunk in chat_fn(user_input, history):
                reply = chunk
            print(f"Model: {reply}\n")
            history.append((user_input, reply))
        return

    default_compare = [m.name for m in models[: min(2, len(models))]]

    with gr.Blocks(title="Tiny LLM Demo") as demo:
        gr.Markdown(
            f"**Loaded {len(models)} model(s) on `{dev}`** · "
            f"Approx model memory: **{model_mem_mb:.1f} MB**"
        )

        with gr.Tabs():
            with gr.Tab("Chat"):
                with gr.Row():
                    chat_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.8,
                        step=0.05,
                        label="Temperature",
                    )
                    chat_top_k = gr.Slider(
                        minimum=0,
                        maximum=200,
                        value=40,
                        step=1,
                        label="Top-k (0 = off)",
                    )
                    chat_top_p = gr.Slider(
                        minimum=0.05,
                        maximum=1.0,
                        value=top_p,
                        step=0.01,
                        label="Top-p",
                    )
                    chat_repetition_penalty = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=repetition_penalty,
                        step=0.01,
                        label="Repetition penalty",
                    )
                    chat_max_tokens = gr.Slider(
                        minimum=1,
                        maximum=512,
                        value=100,
                        step=1,
                        label="Max new tokens",
                    )
                gr.ChatInterface(
                    fn=chat_fn,
                    additional_inputs=[
                        chat_temperature,
                        chat_top_k,
                        chat_top_p,
                        chat_repetition_penalty,
                        chat_max_tokens,
                    ],
                    title="Tiny LLM Chat",
                    description=f"Primary checkpoint: {primary_model.checkpoint_path}",
                )

            with gr.Tab("Compare"):
                gr.Markdown(
                    "Ask one question and compare outputs side-by-side. "
                    f"Select 2..{arena_max_models} models."
                )
                model_selector = gr.CheckboxGroup(
                    choices=[m.name for m in models],
                    value=default_compare,
                    label="Models",
                )
                compare_prompt = gr.Textbox(
                    lines=4,
                    placeholder="Enter the question to send to all selected models...",
                    label="Prompt",
                )
                with gr.Row():
                    compare_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.8,
                        step=0.05,
                        label="Temperature",
                    )
                    compare_top_k = gr.Slider(
                        minimum=0,
                        maximum=200,
                        value=40,
                        step=1,
                        label="Top-k (0 = off)",
                    )
                    compare_top_p = gr.Slider(
                        minimum=0.05,
                        maximum=1.0,
                        value=top_p,
                        step=0.01,
                        label="Top-p",
                    )
                    compare_repetition_penalty = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=repetition_penalty,
                        step=0.01,
                        label="Repetition penalty",
                    )
                    compare_max_tokens = gr.Slider(
                        minimum=1,
                        maximum=512,
                        value=100,
                        step=1,
                        label="Max new tokens",
                    )
                run_compare = gr.Button("Run Compare", variant="primary")
                compare_status = gr.Markdown("Ready.")

                output_boxes: list = []
                cols = 2
                rows = (arena_max_models + cols - 1) // cols
                for r in range(rows):
                    with gr.Row():
                        for c in range(cols):
                            idx = r * cols + c
                            if idx >= arena_max_models:
                                continue
                            output_boxes.append(
                                gr.Textbox(
                                    label=f"Model {idx + 1}",
                                    lines=12,
                                    interactive=False,
                                    visible=False,
                                )
                            )

                run_compare.click(
                    fn=compare_fn,
                    inputs=[
                        compare_prompt,
                        model_selector,
                        compare_temperature,
                        compare_top_k,
                        compare_top_p,
                        compare_repetition_penalty,
                        compare_max_tokens,
                    ],
                    outputs=[compare_status, *output_boxes],
                )
    demo.launch(share=share)
