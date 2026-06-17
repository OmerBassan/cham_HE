# VQAv2 Benchmark

Robustness evaluation of LLMs on Visual Question Answering using the VQAv2 dataset.

## What it tests

Measures how much a model's answer accuracy degrades when the input question is semantically distorted at varying μ (miu) intensity levels. The distorted question preserves the correct answer but uses different vocabulary or sentence structure.

## Dataset

- **Source**: VQAv2 (COCO val2014), 300 sampled images
- **Format**: JSON list — each entry has `question_id`, `question`, `image_id`, `answers` (10 annotator answers), `image_path`
- **Location**: `benchmarks/vqav2/data/vqav2_sampled_optimized.json`
- **Images**: `benchmarks/vqav2/data/sampled_images_300/`

## Evaluation method

**VQA soft accuracy** (official VQAv2 metric):

```text
score = min(count(predicted_answer in ground_truth_answers) / 3, 1.0)
```

A score > 0 counts as a correct answer (`is_correct = True`). The benchmark reports mean VQA score across all tasks.

## Pipeline steps

1. Load tasks from JSON
2. Distort questions with Mistral (at μ level)
3. Validate distortions are semantically equivalent (YES/NO judge)
4. Generate answers for original questions
5. Generate answers for distorted questions
6. Evaluate both with VQA soft accuracy
7. Report original vs distorted scores (robustness gap)

## Quick usage

```python
from benchmarks.vqav2 import VQAV2Benchmark

benchmark = VQAV2Benchmark(config={
    "distortion_model": "mistral-large-latest",
    "generation_model": "mistral-large-latest",
    "validation_model": "mistral-large-latest",
})

summary = benchmark.run_full_pipeline(
    data_path="benchmarks/vqav2/data/vqav2_sampled_optimized.json",
    output_dir="benchmarks/vqav2/test_consolidated",
    miu=0.6,
    limit=10,
)
```

## Via the Chameleon CLI

VQAv2 is a **self-contained** benchmark: when a project's `config.yaml` selects it, the
unified workflow delegates the whole run to `VQAV2Benchmark.run_full_pipeline()` (single μ,
original vs distorted), because generation needs images that the generic text stages can't send.

Add this `benchmark:` block to the project's `config.yaml` (the `init` wizard does not emit it):

```yaml
benchmark:
  type: vqav2
  data_path: benchmarks/vqav2/data/vqav2_sampled_optimized.json
  miu: 0.5                                 # single distortion intensity
  limit: 150                               # optional; omit to run all rows
  generation_model: gemini-2.5-flash       # MUST be a key in ALLOWED_MODELS (config.py)
  distortion_model: mistral-large-latest
  validation_model: mistral-large-latest
  inter_request_delay: 4
  tasks_filename: tasks_complete.jsonl
```

The image directory (`sampled_images_300/`) is resolved automatically as a sibling of
`data_path`. Required env keys: `MISTRAL_API_KEY` (distortion + validation) and the generation
model's key (e.g. `GEMINI_API_KEY` for Gemini, `MISTRAL_API_KEY` for Pixtral) in the project `.env`.

Then run:

```bash
python cli.py workflow --project <name>
```

Outputs land in the project's `results/` dir: `summary.json`, `tasks_complete.jsonl`, and a
resumable `checkpoint.jsonl`.

## Expected output

```json
{
  "total_tasks": 10,
  "distortion_successful": 9,
  "validation_passed": 8,
  "original_metrics": { "vqa_accuracy": 0.65, "exact_match": 0.50 },
  "distorted_metrics": { "vqa_accuracy": 0.58, "exact_match": 0.43 }
}
```

## Run smoke test

```bash
python benchmarks/vqav2/examples/test_consolidated.py
```

## Run unit tests

```bash
pytest tests/test_vqav2.py -v
```
