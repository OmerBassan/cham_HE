"""
VQAv2 dataset loader for the vaq benchmark.

Expected format: a JSON file containing a top-level list of objects with at
minimum the fields ``question_id`` and ``question``.  Optional fields:
``image_id``, ``image``, ``answers`` (list of str or dicts with 'answer' key),
``multiple_choice_answer``, ``answer_type``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

REQUIRED_FIELDS = ("question_id", "question")


def load_entries(data_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"VQAv2 data file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("VQAv2 data must be a JSON list of entries.")

    if limit is not None:
        data = data[:limit]

    cleaned: List[Dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Entry at index {idx} is not a JSON object.")
        missing = [
            field for field in REQUIRED_FIELDS
            if field not in item or item[field] in (None, "")
        ]
        if missing:
            raise ValueError(
                f"Entry at index {idx} missing required fields: {', '.join(missing)}"
            )
        cleaned.append(item)

    return cleaned


def normalize_answers(raw_answers: List[Any]) -> List[str]:
    """Flatten answer list: accept both plain strings and dicts with 'answer' key."""
    result = []
    for a in raw_answers:
        if isinstance(a, str):
            result.append(a)
        elif isinstance(a, dict):
            result.append(str(a.get("answer", "")))
    return result
