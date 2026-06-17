# Research Pipeline

Chameleon measures how an LLM's coding accuracy degrades as the problem
statement is semantically distorted (paraphrased) at increasing intensity μ.

Entry points: `cli.py` (commands) → `chameleon/workflow.py` (`ChameleonWorkflow`).
Each project lives in `Projects/<name>/` with a `config.yaml` driving everything.

## Flow

```
Input JSONL (HumanEval format, original_data/)
  → 1. DISTORT    → distorted_data/distortions_complete.jsonl
  → 2. GENERATE   → distorted_data/samples.jsonl
  → 3. EVALUATE   → distorted_data/samples.jsonl_results.jsonl
  → 4. ANALYZE    → analysis/analysis_summary.csv + per-μ Pass@1
```

Stages are independently skippable (`--skip-distortion`, etc.) and resumable
(generate skips `task_id`s already in `samples.jsonl`).

## 1. Distortion

`DistortionRunner` (`chameleon/distortion/runner.py`) takes each original prompt
and produces `distortions_per_question` variants at every μ in `miu_values`.

- **μ (intensity):** `0.0` = identity (prompt returned unchanged), `1.0` = fully
  paraphrased. Per-μ rewrite rules + temperature scaling live in
  `chameleon/distortion/constants.py`.
- The distortion model (default Mistral `mistral-large-latest`) rewrites the
  prompt; raw output is cleaned by `_extract_single_distortion` (tag → numbered
  list → quoted → trailing `?` → fallback).
- Each output row carries `distortion_id` (unique per variant), `question_id`
  (maps back to the original), `miu`, `question_text`, `distorted_question`.

**Engine abstraction** (`chameleon/distortion/engine.py`): `BaseDistortionEngine`
with three backends — `LocalHuggingFaceEngine` (offline GPU), `APIDistortionEngine`
(OpenAI/Anthropic/Mistral), `OllamaDistortionEngine` (local server) — built via
the `DistortionEngine` factory. NOTE: `runner.py` currently uses its own optimized
Batch-API path; `engine.py` is the modular/offline reference, not on the hot path.

## 2. Generation (target model inference)

`_stage_generate` loads `distortions_complete.jsonl`, builds a `Task` per row, and
asks the benchmark for the final prompt (`benchmark.get_generation_prompt`). The
**target model** (a `ModelBackend` from `chameleon/models/registry.py`) completes
each prompt in parallel (`ThreadPoolExecutor`, temp 0.2, max 1024 tokens).
Output: `{task_id=distortion_id, completion}` lines in `samples.jsonl`.

## 3. Evaluation (functional correctness)

`_stage_evaluate` joins each sample with its distortion row (`distortion_id` →
`question_id` → original task with unit tests), then `benchmark.evaluate(task,
completion)` executes the code against the test cases in a sandbox. Output rows:
`{task_id, passed, miu, ...}` in `samples.jsonl_results.jsonl`.

## 4. Analysis

`_stage_analyze` maps `passed` back onto the distortion metadata and reports
overall Pass@1 plus Pass@1 broken down by μ — the degradation curve. Saved to
`analysis/analysis_summary.csv`. Richer metrics (CRI, elasticity, McNemar) live
in `chameleon/analysis/`.

## Key config (config.yaml)

- `distortion.miu_values`, `distortion.distortions_per_question`
- `distortion.engine.{vendor,model_name}` — distortion model
- `target_model.{vendor,name}` — model under test
- `benchmark.{type,data_path}` — which benchmark drives prompts/eval/metrics
