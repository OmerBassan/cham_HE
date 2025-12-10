"""
Evaluation module for target model testing.

Handles batch submission, monitoring, and result collection for 
evaluating model performance on distorted questions.
"""

from chameleon.evaluation.batch_processor import (
    EvaluationConfig,
    BatchProcessor,
    run_evaluation,
)

__all__ = [
    "EvaluationConfig",
    "BatchProcessor",
    "run_evaluation",
]

