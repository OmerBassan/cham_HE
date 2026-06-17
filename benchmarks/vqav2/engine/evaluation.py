"""
CLI entry point for stand-alone VQA evaluation (fire-based).

Usage:
    python -m benchmarks.vqav2.engine.evaluation \\
        --results_file path/to/tasks_complete.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

try:
    import fire
except ImportError:
    fire = None  # fire is optional; class can still be used programmatically

from benchmarks.vqav2.engine.execution import evaluate_answer
from benchmarks.vqav2.engine.evaluate_functional_correctness import compute_vqa_metrics


def entry_point(results_file: str, output_file: Optional[str] = None) -> None:
    path = Path(results_file)
    tasks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))

    original_results = []
    distorted_results = []
    for task in tasks:
        gt = task.get("answers", [])
        if task.get("original_answer"):
            original_results.append(
                evaluate_answer(task["question_id"], task["original_answer"], gt)
            )
        if task.get("distorted_answer"):
            distorted_results.append(
                evaluate_answer(task["question_id"], task["distorted_answer"], gt)
            )

    report = {
        "original": compute_vqa_metrics(original_results),
        "distorted": compute_vqa_metrics(distorted_results),
    }

    print(json.dumps(report, indent=2))

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    if fire:
        fire.Fire(entry_point)
    else:
        import sys
        print("Install 'fire' to use CLI: pip install fire", file=sys.stderr)
