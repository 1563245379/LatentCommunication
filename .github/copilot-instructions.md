# Copilot Instructions — LatentCommunication

## Project Overview

Latent-space multi-agent reasoning system for LLMs. Agents (Planner → Critic → Refiner → Judger) collaborate through latent KV-cache representations rather than text, enabling intermediate reasoning without token generation. Supports math (GSM8K, AIME), multiple-choice (ARC, GPQA, MedQA), and coding (MBPP+, HumanEval+) benchmarks.

## Architecture

```
run.py            → CLI entry point, arg parsing, batch evaluation loop
models.py         → ModelWrapper: HuggingFace model loading, text/latent generation, alignment
data.py           → Task-specific data loaders (HF datasets + local JSON)
prompts.py        → Sequential & hierarchical prompt templates per task type
train_alignment.py→ Ridge regression alignment matrix training (standalone CLI)
utils.py          → Device/seed setup, answer extraction, safe code execution
methods/
  latent_mas.py   → Core multi-agent orchestration, KV cache threading, evaluation
  latent_mas_dd.py→ Data-driven subclass: loads pre-trained alignment from disk
```

**Key flow:** Load model → Load data → Train/load alignment matrix → Multi-agent latent loop → Evaluate → Save JSONL results.

## Build & Run

```bash
# Install
pip install -r requirements.txt

# Train alignment matrix (once per model)
python train_alignment.py --model_name Qwen/Qwen3-4B --n_train 500

# Run evaluation
python run.py --model_name Qwen/Qwen3-4B --task gsm8k --prompt sequential --latent_space_realign

# Results saved to results/ as JSONL
```

## Coding Conventions

- **Style:** snake_case for functions/variables, CamelCase for classes
- **Type hints:** Full annotations everywhere (`Dict`, `List`, `Optional`, `Tuple`)
- **Inference:** HuggingFace Transformers only (no vLLM); bfloat16 on CUDA, float32 on CPU
- **Batch processing:** All generation methods support batched inputs (`generate_bs` controls batch size)
- **State threading:** Past KV caches are passed through the agent chain with truncation for memory management
- **Prompt structure:** Task-aware templates — math uses `\boxed{}`, MC uses `A/B/C/D`, coding uses ` ```python ` blocks

## Key Patterns

- `ModelWrapper` encapsulates all model interactions. Never call `model.generate()` directly outside it.
- Alignment matrices live at `alignment_matrices/{model_name}/` and are auto-trained if missing on first run.
- `LatentMASMethod` is the base agent orchestrator; `LatentMASDDMethod` only overrides alignment loading.
- Answer extraction uses task-specific regex in `utils.py` — keep extractors consistent with prompt format.

## Pitfalls

- **Qwen enforcement:** Code asserts model name contains "qwen" by default. Use `--do_not_enforce_qwen` for other models.
- **GPU memory:** Large models (7B+) may exceed VRAM in bfloat16. Reduce batch size or use CPU fallback.
- **Alignment matrix:** First run auto-trains from 500 samples (slow). Pre-train with `train_alignment.py` separately.
- **MedQA data:** Requires local `data/medqa.json` — not auto-downloaded from HF.
- **Latent realignment:** Requires both `--latent_space_realign` flag AND a trained alignment matrix to be effective.
