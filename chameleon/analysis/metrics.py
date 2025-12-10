"""
Basic metrics calculation for model evaluation.

Provides:
- Accuracy calculation
- Group-wise accuracy
- Performance degradation metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple


def calculate_accuracy(
    df: pd.DataFrame,
    answer_col: str = "model_answer",
    correct_col: str = "correct_answer",
    is_correct_col: Optional[str] = "is_correct"
) -> float:
    """
    Calculate overall accuracy from a DataFrame.
    
    Args:
        df: DataFrame with results
        answer_col: Column with model answers
        correct_col: Column with correct answers
        is_correct_col: Column with pre-computed correctness (optional)
    
    Returns:
        Accuracy as a float (0.0 to 1.0)
    """
    if is_correct_col and is_correct_col in df.columns:
        correct = df[is_correct_col].sum()
        total = df[is_correct_col].notna().sum()
    else:
        correct = (df[answer_col] == df[correct_col]).sum()
        total = len(df)
    
    return correct / total if total > 0 else 0.0


def calculate_accuracy_by_group(
    df: pd.DataFrame,
    group_col: str,
    answer_col: str = "model_answer",
    correct_col: str = "correct_answer",
    is_correct_col: Optional[str] = "is_correct"
) -> pd.DataFrame:
    """
    Calculate accuracy grouped by a column (e.g., subject, distortion level).
    
    Args:
        df: DataFrame with results
        group_col: Column to group by
        answer_col: Column with model answers
        correct_col: Column with correct answers
        is_correct_col: Column with pre-computed correctness
    
    Returns:
        DataFrame with accuracy per group
    """
    results = []
    
    for group_value in df[group_col].unique():
        group_df = df[df[group_col] == group_value]
        accuracy = calculate_accuracy(group_df, answer_col, correct_col, is_correct_col)
        
        results.append({
            group_col: group_value,
            "accuracy": accuracy,
            "correct_count": int(group_df[is_correct_col].sum()) if is_correct_col and is_correct_col in df.columns else int((group_df[answer_col] == group_df[correct_col]).sum()),
            "total_count": len(group_df),
        })
    
    return pd.DataFrame(results).sort_values(group_col)


def calculate_degradation(
    df: pd.DataFrame,
    baseline_filter: Dict[str, Any],
    comparison_col: str,
    answer_col: str = "model_answer",
    correct_col: str = "correct_answer",
    is_correct_col: Optional[str] = "is_correct"
) -> pd.DataFrame:
    """
    Calculate performance degradation from baseline for each value of comparison_col.
    
    Args:
        df: DataFrame with results
        baseline_filter: Dict of column:value to filter baseline (e.g., {"miu": 0.0})
        comparison_col: Column to compare across (e.g., "miu", "subject")
        answer_col: Column with model answers
        correct_col: Column with correct answers
        is_correct_col: Column with pre-computed correctness
    
    Returns:
        DataFrame with degradation metrics
    """
    # Calculate baseline accuracy
    baseline_mask = pd.Series([True] * len(df))
    for col, val in baseline_filter.items():
        baseline_mask &= (df[col] == val)
    
    baseline_df = df[baseline_mask]
    baseline_accuracy = calculate_accuracy(baseline_df, answer_col, correct_col, is_correct_col)
    
    # Calculate accuracy and degradation for each comparison value
    results = []
    
    for comp_value in df[comparison_col].unique():
        comp_df = df[df[comparison_col] == comp_value]
        comp_accuracy = calculate_accuracy(comp_df, answer_col, correct_col, is_correct_col)
        
        degradation = baseline_accuracy - comp_accuracy
        degradation_pct = (degradation / baseline_accuracy * 100) if baseline_accuracy > 0 else 0
        
        results.append({
            comparison_col: comp_value,
            "accuracy": comp_accuracy,
            "baseline_accuracy": baseline_accuracy,
            "degradation": degradation,
            "degradation_percent": degradation_pct,
            "sample_size": len(comp_df),
        })
    
    return pd.DataFrame(results).sort_values(comparison_col)


def calculate_cross_degradation(
    df: pd.DataFrame,
    group_col: str,
    level_col: str = "miu",
    baseline_level: float = 0.0,
    is_correct_col: str = "is_correct"
) -> pd.DataFrame:
    """
    Calculate degradation for each group across different levels.
    
    Args:
        df: DataFrame with results
        group_col: Column to group by (e.g., "subject")
        level_col: Column with distortion levels (e.g., "miu")
        baseline_level: Value of level_col to use as baseline
        is_correct_col: Column with pre-computed correctness
    
    Returns:
        DataFrame with rows=groups, columns=levels, values=degradation
    """
    results = []
    
    for group in df[group_col].unique():
        group_df = df[df[group_col] == group]
        
        # Get baseline accuracy for this group
        baseline_df = group_df[group_df[level_col] == baseline_level]
        baseline_acc = calculate_accuracy(baseline_df, is_correct_col=is_correct_col)
        
        row = {group_col: group, "baseline_accuracy": baseline_acc}
        
        # Calculate degradation at each level
        for level in sorted(df[level_col].unique()):
            if level == baseline_level:
                row[f"deg_{level}"] = 0.0
            else:
                level_df = group_df[group_df[level_col] == level]
                level_acc = calculate_accuracy(level_df, is_correct_col=is_correct_col)
                row[f"deg_{level}"] = (baseline_acc - level_acc) * 100
        
        results.append(row)
    
    return pd.DataFrame(results).sort_values(group_col)


def clean_model_answer(answer: Any, valid_answers: List[str] = ["A", "B", "C", "D"]) -> str:
    """
    Clean and extract model answer from raw response.
    
    Args:
        answer: Raw answer value
        valid_answers: List of valid answer options
    
    Returns:
        Cleaned answer string
    """
    import re
    
    if pd.isna(answer) or answer == '':
        return ''
    
    answer_str = str(answer).strip().upper()
    
    # If already a valid answer, return it
    if answer_str in valid_answers:
        return answer_str
    
    # Try to extract first valid answer character
    for char in answer_str:
        if char in valid_answers:
            return char
    
    # Try regex for patterns like "A)", "(A)", "Answer: A"
    pattern = r'[(\s]?([' + ''.join(valid_answers) + r'])[)\s.:,]?'
    match = re.search(pattern, answer_str)
    if match:
        return match.group(1)
    
    return answer_str


def prepare_results_dataframe(
    df: pd.DataFrame,
    answer_col: str = "gpt5_answer",
    correct_col: str = "correct_answer",
    clean_answers: bool = True
) -> pd.DataFrame:
    """
    Prepare results DataFrame with cleaned answers and correctness column.
    
    Args:
        df: Raw DataFrame with results
        answer_col: Column with model answers
        correct_col: Column with correct answers
        clean_answers: Whether to clean model answers
    
    Returns:
        DataFrame with 'is_correct' column added
    """
    df = df.copy()
    
    # Clean answers if requested
    if clean_answers:
        df['cleaned_answer'] = df[answer_col].apply(clean_model_answer)
        answer_col = 'cleaned_answer'
    
    # Calculate correctness
    df['is_correct'] = df.apply(
        lambda row: row[answer_col] == row[correct_col] 
        if pd.notna(row[answer_col]) and row[answer_col] != '' 
        else np.nan,
        axis=1
    )
    
    return df


