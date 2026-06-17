from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VAQTask:
    question_id: str
    question: str
    image_id: Optional[int]
    image_path: str
    answers: List[str]
    multiple_choice_answer: Optional[str] = None
    answer_type: Optional[str] = None

    # Distortion fields
    distorted_question: Optional[str] = None
    distortion_success: bool = False
    distortion_miu: Optional[float] = None
    distortion_error: Optional[str] = None

    # Validation fields
    validation_passed: Optional[bool] = None
    validation_reason: str = ""

    # Generation fields
    original_answer: Optional[str] = None
    distorted_answer: Optional[str] = None
    original_generation_success: bool = False
    distorted_generation_success: bool = False

    # Evaluation fields
    original_vqa_score: Optional[float] = None
    distorted_vqa_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "image_id": self.image_id,
            "image_path": self.image_path,
            "answers": self.answers,
            "multiple_choice_answer": self.multiple_choice_answer,
            "answer_type": self.answer_type,
            "distorted_question": self.distorted_question,
            "distortion_success": self.distortion_success,
            "distortion_miu": self.distortion_miu,
            "distortion_error": self.distortion_error,
            "validation_passed": self.validation_passed,
            "validation_reason": self.validation_reason,
            "original_answer": self.original_answer,
            "distorted_answer": self.distorted_answer,
            "original_generation_success": self.original_generation_success,
            "distorted_generation_success": self.distorted_generation_success,
            "original_vqa_score": self.original_vqa_score,
            "distorted_vqa_score": self.distorted_vqa_score,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> VAQTask:
        return cls(
            question_id=d["question_id"],
            question=d["question"],
            image_id=d.get("image_id"),
            image_path=d.get("image_path", ""),
            answers=d.get("answers", []),
            multiple_choice_answer=d.get("multiple_choice_answer"),
            answer_type=d.get("answer_type"),
            distorted_question=d.get("distorted_question"),
            distortion_success=d.get("distortion_success", False),
            distortion_miu=d.get("distortion_miu"),
            distortion_error=d.get("distortion_error"),
            validation_passed=d.get("validation_passed"),
            validation_reason=d.get("validation_reason", ""),
            original_answer=d.get("original_answer"),
            distorted_answer=d.get("distorted_answer"),
            original_generation_success=d.get("original_generation_success", False),
            distorted_generation_success=d.get("distorted_generation_success", False),
            original_vqa_score=d.get("original_vqa_score"),
            distorted_vqa_score=d.get("distorted_vqa_score"),
        )


@dataclass
class PipelineConfig:
    distortion_model: str = "mistral-large-latest"
    validation_model: str = "mistral-large-latest"
    generation_model: str = "gemini-2.0-flash"
    miu: float = 0.6
    project_path: Optional[str] = None
    generation_temperature: float = 0.0
    generation_max_tokens: int = 64


@dataclass
class PipelineResult:
    total_tasks: int = 0
    distortion_successful: int = 0
    validation_passed_count: int = 0
    original_answers_generated: int = 0
    distorted_answers_generated: int = 0
    original_vqa_score: float = 0.0
    distorted_vqa_score: float = 0.0
    miu: float = 0.6
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "distortion_successful": self.distortion_successful,
            "validation_passed": self.validation_passed_count,
            "original_answers_generated": self.original_answers_generated,
            "distorted_answers_generated": self.distorted_answers_generated,
            "original_vqa_score": self.original_vqa_score,
            "distorted_vqa_score": self.distorted_vqa_score,
            "miu": self.miu,
            **self.metadata,
        }
