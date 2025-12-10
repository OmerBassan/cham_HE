"""
CLI module for Chameleon framework.

Provides:
- Command-line interface commands
- Input validation utilities
- File handling utilities
"""

from chameleon.cli.commands import main
from chameleon.cli.validators import (
    get_validated_input,
    get_validated_input_with_default,
    validate_project_name,
    validate_miu_values,
    validate_model_name,
    validate_vendor,
    validate_api_key,
    validate_file_path,
    get_yes_no,
    get_choice,
    get_choice_with_skip,
    ValidationError,
)
from chameleon.cli.file_utils import (
    convert_excel_to_csv,
    load_data_file,
    copy_file_to_project,
    standardize_distortion_data,
)

__all__ = [
    "main",
    "get_validated_input",
    "get_validated_input_with_default",
    "validate_project_name",
    "validate_miu_values",
    "validate_model_name",
    "validate_vendor",
    "validate_api_key",
    "validate_file_path",
    "get_yes_no",
    "get_choice",
    "get_choice_with_skip",
    "ValidationError",
    "convert_excel_to_csv",
    "load_data_file",
    "copy_file_to_project",
    "standardize_distortion_data",
]
