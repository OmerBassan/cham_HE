"""
Core module for Chameleon framework.

Provides:
- Project management
- Configuration handling
- Data schemas
"""

from chameleon.core.schemas import (
    Modality,
    BackendType,
    DistortionEngineType,
    DistortionEngineConfig,
    DistortionConfig,
    DataSchema,
    ProjectConfig,
    ProjectMetadata,
    EvaluationRecord,
    BatchResult,
)

from chameleon.core.project import (
    Project,
    list_projects,
)

from chameleon.core.config import (
    ChameleonConfig,
)

__all__ = [
    # Schemas
    "Modality",
    "BackendType",
    "DistortionConfig",
    "DataSchema",
    "ProjectConfig",
    "ProjectMetadata",
    "EvaluationRecord",
    "BatchResult",
    # Project
    "Project",
    "list_projects",
    # Config
    "ChameleonConfig",
]
