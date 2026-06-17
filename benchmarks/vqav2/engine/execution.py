"""
VQA answer evaluation.

Uses the official VQA soft accuracy metric:
    score = min(number_of_annotators_who_gave_the_same_answer / 3, 1.0)

This rewards answers that appear in the ground-truth list without requiring
a perfect match, while capping at 1.0.
"""

from __future__ import annotations

import re
import string
from typing import Dict, List


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


def vqa_score(predicted: str, ground_truth_answers: List[str]) -> float:
    """Return VQA soft accuracy in [0.0, 1.0]."""
    if not ground_truth_answers:
        return 0.0
    pred_norm = _normalize(predicted)
    count = sum(1 for a in ground_truth_answers if _normalize(a) == pred_norm)
    return min(count / 3.0, 1.0)


def evaluate_answer(
    question_id: str,
    predicted: str,
    ground_truth_answers: List[str],
) -> Dict:
    score = vqa_score(predicted, ground_truth_answers)
    return {
        "question_id": question_id,
        "predicted": predicted,
        "ground_truth": ground_truth_answers,
        "vqa_score": score,
        "passed": score > 0.0,
    }
