"""
File utilities for CLI operations.

Provides:
- XLS/XLSX to CSV conversion
- File format detection
- Data loading utilities
"""

import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd


def convert_excel_to_csv(
    excel_path: Path,
    output_path: Optional[Path] = None,
    sheet_name: Optional[str] = None
) -> Path:
    """
    Convert Excel file (XLS/XLSX) to CSV.
    
    Args:
        excel_path: Path to Excel file
        output_path: Output CSV path (defaults to same name with .csv)
        sheet_name: Specific sheet to convert (defaults to first sheet)
    
    Returns:
        Path to created CSV file
    """
    excel_path = Path(excel_path)
    
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    if excel_path.suffix.lower() not in ['.xls', '.xlsx', '.xlsm']:
        raise ValueError(f"Not an Excel file: {excel_path}")
    
    # Determine output path
    if output_path is None:
        output_path = excel_path.with_suffix('.csv')
    else:
        output_path = Path(output_path)
    
    # Read Excel file
    try:
        if sheet_name:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(excel_path)
    except ImportError:
        raise ImportError(
            "openpyxl or xlrd package required for Excel files. "
            "Install with: pip install openpyxl xlrd"
        )
    
    # Save as CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    return output_path


def list_excel_sheets(excel_path: Path) -> List[str]:
    """List all sheet names in an Excel file."""
    try:
        xl = pd.ExcelFile(excel_path)
        return xl.sheet_names
    except ImportError:
        raise ImportError("openpyxl or xlrd package required for Excel files.")


def detect_file_format(file_path: Path) -> str:
    """
    Detect file format based on extension and content.
    
    Returns:
        Format string: 'csv', 'jsonl', 'json', 'excel', 'unknown'
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    
    format_map = {
        '.csv': 'csv',
        '.jsonl': 'jsonl',
        '.json': 'json',
        '.xls': 'excel',
        '.xlsx': 'excel',
        '.xlsm': 'excel',
        '.tsv': 'tsv',
    }
    
    return format_map.get(ext, 'unknown')


def load_data_file(file_path: Path, auto_convert: bool = True) -> pd.DataFrame:
    """
    Load data from various file formats.
    
    Args:
        file_path: Path to data file
        auto_convert: Automatically convert Excel to CSV
    
    Returns:
        DataFrame with loaded data
    """
    file_path = Path(file_path)
    format_type = detect_file_format(file_path)
    
    if format_type == 'csv':
        return pd.read_csv(file_path, encoding='utf-8')
    
    elif format_type == 'jsonl':
        return pd.read_json(file_path, lines=True)
    
    elif format_type == 'json':
        return pd.read_json(file_path)
    
    elif format_type == 'tsv':
        return pd.read_csv(file_path, sep='\t', encoding='utf-8')
    
    elif format_type == 'excel':
        if auto_convert:
            # Convert to CSV first
            csv_path = convert_excel_to_csv(file_path)
            print(f"   📄 Converted Excel to CSV: {csv_path}")
            return pd.read_csv(csv_path, encoding='utf-8')
        else:
            return pd.read_excel(file_path)
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def copy_file_to_project(
    source_path: Path,
    project_dir: Path,
    target_subdir: str = "original_data",
    convert_excel: bool = True
) -> Path:
    """
    Copy a file to a project directory, converting if necessary.
    
    Args:
        source_path: Source file path
        project_dir: Project directory
        target_subdir: Subdirectory within project (original_data, distorted_data, etc.)
        convert_excel: Convert Excel files to CSV
    
    Returns:
        Path to the copied/converted file
    """
    source_path = Path(source_path)
    target_dir = project_dir / target_subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    
    format_type = detect_file_format(source_path)
    
    if format_type == 'excel' and convert_excel:
        # Convert Excel to CSV
        target_path = target_dir / source_path.with_suffix('.csv').name
        df = pd.read_excel(source_path)
        df.to_csv(target_path, index=False, encoding='utf-8')
        print(f"   📄 Converted and copied: {source_path.name} -> {target_path.name}")
    else:
        # Simple copy
        target_path = target_dir / source_path.name
        shutil.copy2(source_path, target_path)
        print(f"   📄 Copied: {source_path.name}")
    
    return target_path


def validate_data_schema(
    df: pd.DataFrame,
    required_columns: List[str],
    optional_columns: List[str] = None
) -> Dict[str, Any]:
    """
    Validate DataFrame schema against requirements.
    
    Args:
        df: DataFrame to validate
        required_columns: Columns that must be present
        optional_columns: Columns that are nice to have
    
    Returns:
        Dictionary with validation results
    """
    optional_columns = optional_columns or []
    
    result = {
        "valid": True,
        "missing_required": [],
        "missing_optional": [],
        "extra_columns": [],
        "column_count": len(df.columns),
        "row_count": len(df),
    }
    
    df_columns = set(df.columns)
    required_set = set(required_columns)
    optional_set = set(optional_columns)
    expected_set = required_set | optional_set
    
    # Check required columns
    missing_required = required_set - df_columns
    if missing_required:
        result["valid"] = False
        result["missing_required"] = list(missing_required)
    
    # Check optional columns
    missing_optional = optional_set - df_columns
    result["missing_optional"] = list(missing_optional)
    
    # Extra columns
    result["extra_columns"] = list(df_columns - expected_set)
    
    return result


def standardize_distortion_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize distortion data to expected column format.
    
    Expected columns:
    - question_id (or q_id)
    - original_question (or original_q)
    - distorted_question (or distorted_q)
    - miu
    - distortion_index
    - answer_options (or ans_options)
    - correct_answer (or correct_ans)
    - subject (optional)
    - topic (optional)
    
    Returns:
        DataFrame with standardized column names
    """
    # Column name mappings
    column_mappings = {
        'q_id': 'question_id',
        'original_q': 'original_question',
        'distorted_q': 'distorted_question',
        'ans_options': 'answer_options',
        'correct_ans': 'correct_answer',
        'target_model_ans': 'model_answer',
    }
    
    df = df.copy()
    
    # Apply mappings
    for old_name, new_name in column_mappings.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    return df


