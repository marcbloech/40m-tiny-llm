# Tiny LLM (40M)

A ~40M-parameter decoder-only transformer, written from scratch in PyTorch. Everything here is `torch.nn` code: the multi-head attention, RoPE, RMSNorm, SwiGLU FFN, KV cache, training loop, data pipeline. I didn't use HuggingFace Trainer, Lightning, or any pre-built model classes.

I trained it on ~800M tokens from six sources (OpenWebText, Wikipedia, FineWeb-Edu, C4, BookCorpus, WikiHow), then ran SFT and DPO post-training. Architecture and hyperparameters came out of Optuna sweeps and multi-seed ablations, all documented in the notebooks.

## Architecture

| Component | Choice | Why |
|-----------|--------|-----|
| Positional encoding | RoPE | Relative positions, extrapolates to unseen lengths |
| Normalization | RMSNorm (pre-norm) | Simpler than LayerNorm, more stable training |
| Activation | SwiGLU | More expressive per parameter than GELU |
| Weight tying | Embedding = LM head | Halves embedding params at this scale |
| Attention | Multi-head causal | 8 heads, 512-dim, with KV cache for inference |
| Context length | 1024 tokens | |
| Tokenizer | GPT-2 BPE (50,257 vocab) | Via tiktoken |

Final config: 7 layers, 512 hidden dim, 640 FFN dim (SwiGLU), 8 heads = ~40M parameters.

## What's in the pipeline

1. Data collection, cleaning, deduplication, 13-gram decontamination against eval sets, tokenization to memory-mapped binaries
2. Pre-training with AdamW, cosine LR schedule, gradient accumulation/clipping, checkpointing with top-K pruning
3. SFT with Alpaca prompt template, response-only loss masking, multi-dataset mixing (Alpaca, OASST1, Dolly, benchmark MC data), catastrophic forgetting probes
4. DPO with reference model freezing and preference pair training
5. Evaluation: perplexity (OpenWebText, WikiText-103), LAMBADA, HellaSwag, WinoGrande, OpenBookQA
6. Gradio chat demo with streaming, side-by-side model comparison, multiple chat templates

## Quick Start

```bash
# Requires Python 3.11+ and uv
uv sync

# Run the data pipeline
uv run python main.py --stage data --config configs/final.toml

# Train the model
uv run python main.py --stage train --config configs/final.toml

# SFT post-training
uv run python main.py --stage posttraining --config configs/final.toml --checkpoint checkpoints/exp_c/best.pt

# Run inference
uv run python main.py --stage inference --checkpoint <path> --prompt "Hello" --max-tokens 100

# Launch the chat demo
uv run python main.py --stage demo --checkpoint <path> --share

# Evaluate
uv run python main.py --stage evaluate --checkpoint <path>
```

Use `--overrides key=value` to override any config parameter from the CLI:
```bash
uv run python main.py --stage train --config configs/final.toml \
  --overrides training.learning_rate=1e-4 training.max_steps=5000
```

## Project Structure

```
main.py                     # CLI entry point (stage-based routing)
configs/                    # TOML configs (exp_a/b/c/d variants + final)
notebooks/                  # Experiment notebooks (EDA, ablations, sweeps)
src/tiny_llm/
  model/                    # GPTModel, MultiHeadAttention, RoPE, RMSNorm
  data/                     # Download, preprocess, decontaminate, tokenize
  training/                 # Trainer, AdamW + cosine scheduler, checkpointing
  posttraining/             # SFT and DPO pipelines
  eval/                     # Perplexity, LAMBADA, HellaSwag, generation
  demo/                     # Gradio chat interface with model arena
  utils/                    # Config system, device detection, seeding, logging
```

## Configs

| Config | Architecture | Purpose |
|--------|-------------|---------|
| `final.toml` | d=512, L=7, H=8 (~40M) | Final model (Config C) |
| `exp_a.toml` | d=288, L=8, H=4 | Optuna experiment A |
| `exp_b.toml` | d=448, L=10, H=8 | Optuna experiment B |
| `exp_c.toml` | d=512, L=7, H=8 | Optuna experiment C (winner) |
| `exp_d.toml` | — | Experimental variant D |

## License

MIT
