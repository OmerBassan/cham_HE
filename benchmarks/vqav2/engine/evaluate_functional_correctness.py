"""
Aggregate VQA evaluation results into dataset-level metrics.
"""

from __future__ import annotations

from typing import Dict, List


def compute_vqa_accuracy(results: List[Dict]) -> float:
    """Mean VQA soft score across all results."""
    if not results:
        return 0.0
    return sum(r["vqa_score"] for r in results) / len(results)


def compute_vqa_metrics(results: List[Dict]) -> Dict:
    if not results:
        return {"vqa_accuracy": 0.0, "exact_match": 0.0, "total": 0}

    vqa_acc = compute_vqa_accuracy(results)
    exact = sum(1 for r in results if r.get("passed", False)) / len(results)

    return {
        "vqa_accuracy": round(vqa_acc, 4),
        "exact_match": round(exact, 4),
        "total": len(results),
    }
