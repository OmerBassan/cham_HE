"""
Chameleon - LLM Robustness Testing Framework

A framework for evaluating model performance under various 
lexical distortion conditions.

Main modules:
- chameleon.distortion: Generate distorted questions
- chameleon.evaluation: Evaluate target models
- chameleon.analysis: Analyze results
- chameleon.workflow: Complete end-to-end workflow
- chameleon.cli: Command-line interface
"""

__version__ = "2.0.0"

from chameleon.workflow import ChameleonWorkflow, WorkflowConfig, run_workflow
from chameleon.distortion import run_distortions, DistortionRunner

__all__ = [
    "__version__",
    "ChameleonWorkflow",
    "WorkflowConfig",
    "run_workflow",
    "run_distortions",
    "DistortionRunner",
]
