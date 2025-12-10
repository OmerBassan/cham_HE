"""
Report generation for analysis results.

Creates:
- Markdown reports
- Summary statistics
- Export to various formats
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List


def generate_statistical_report(
    distortion_results: pd.DataFrame,
    subject_results: pd.DataFrame,
    pairwise_results: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
    project_name: str = "Chameleon Analysis"
) -> str:
    """
    Generate a comprehensive statistical report in Markdown format.
    
    Args:
        distortion_results: Results from distortion level analysis
        subject_results: Results from subject-specific analysis
        pairwise_results: Optional pairwise level comparison results
        output_path: Path to save the report
        project_name: Name for the report header
    
    Returns:
        Report content as string
    """
    report_lines = []
    
    # Header
    report_lines.append(f"# {project_name}: Statistical Analysis Report\n")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("## McNemar's Test Results for Paired Comparisons\n")
    report_lines.append("This report presents comprehensive statistical analysis using McNemar's test ")
    report_lines.append("for paired binary outcomes (correct/incorrect answers).\n\n")
    
    # Executive Summary
    report_lines.append("## Executive Summary\n")
    
    if 'p_value' in distortion_results.columns:
        significant_distortions = distortion_results[distortion_results['p_value'] < 0.05]
        report_lines.append(f"- **Total Distortion Levels Tested**: {len(distortion_results)}\n")
        report_lines.append(f"- **Significant Distortion Effects**: {len(significant_distortions)} ")
        report_lines.append(f"({len(significant_distortions)/len(distortion_results)*100:.1f}%)\n")
    
    if 'is_significant' in subject_results.columns:
        significant_subjects = subject_results[subject_results['is_significant'] == True]
        report_lines.append(f"- **Subjects with Significant Degradation**: {len(significant_subjects)} ")
        report_lines.append(f"({len(significant_subjects)/len(subject_results)*100:.1f}%)\n\n")
    
    # Distortion Level Analysis
    report_lines.append("## 1. Distortion Level Analysis (vs Baseline)\n")
    report_lines.append("| Level | Baseline Acc | Distorted Acc | Difference | χ² | p-value | Significance |\n")
    report_lines.append("|-------|--------------|---------------|------------|-----|---------|-------------|\n")
    
    level_col = [c for c in distortion_results.columns if 'level' in c.lower()][0] if any('level' in c.lower() for c in distortion_results.columns) else distortion_results.columns[0]
    
    for _, row in distortion_results.iterrows():
        level = row.get(level_col, row.iloc[0])
        baseline = row.get('baseline_accuracy', 0) * 100
        distorted = row.get('comparison_accuracy', row.get('distorted_accuracy', 0)) * 100
        diff = row.get('accuracy_difference', 0) * 100
        stat = row.get('mcnemar_statistic', 0)
        p_val = row.get('p_value', 1)
        sig = row.get('significance', 'N/A')
        
        report_lines.append(f"| {level:.1f} | {baseline:.1f}% | {distorted:.1f}% | {diff:+.1f}% | ")
        report_lines.append(f"{stat:.2f} | {p_val:.6f} | {sig.split('(')[0].strip()} |\n")
    
    # Subject-Specific Analysis
    report_lines.append("\n## 2. Subject-Specific Analysis\n")
    report_lines.append("| Subject | Degradation | χ² | p-value | Significance |\n")
    report_lines.append("|---------|-------------|-----|---------|-------------|\n")
    
    for _, row in subject_results.iterrows():
        subject = row.get('subject_name', row.get('subject', 'Unknown'))
        deg = row.get('degradation_percent', 0)
        stat = row.get('mcnemar_statistic', 0)
        p_val = row.get('p_value', 1)
        sig = row.get('significance', 'N/A')
        
        report_lines.append(f"| {subject} | {deg:.1f}% | {stat:.2f} | {p_val:.6f} | ")
        report_lines.append(f"{sig.split('(')[0].strip()} |\n")
    
    # Pairwise Comparisons
    if pairwise_results is not None and len(pairwise_results) > 0:
        report_lines.append("\n## 3. Pairwise Level Comparisons\n")
        report_lines.append("| Comparison | Diff | χ² | p-value | Significance |\n")
        report_lines.append("|------------|------|-----|---------|-------------|\n")
        
        for _, row in pairwise_results.iterrows():
            comp = row.get('comparison', 'N/A')
            diff = row.get('accuracy_difference', 0) * 100
            stat = row.get('mcnemar_statistic', 0)
            p_val = row.get('p_value', 1)
            sig = row.get('significance', 'N/A')
            
            report_lines.append(f"| {comp} | {diff:+.1f}% | {stat:.2f} | {p_val:.6f} | ")
            report_lines.append(f"{sig.split('(')[0].strip()} |\n")
    
    # Interpretation
    report_lines.append("\n## Statistical Interpretation\n")
    report_lines.append("**McNemar's Test** is used for paired binary data to test whether the marginal ")
    report_lines.append("frequencies of correct/incorrect answers differ between two conditions.\n\n")
    report_lines.append("- **Null Hypothesis**: No difference in accuracy between conditions\n")
    report_lines.append("- **Alternative Hypothesis**: Significant difference in accuracy\n")
    report_lines.append("- **Significance Levels**: * p<0.05, ** p<0.01, *** p<0.001\n\n")
    report_lines.append("**Confidence Intervals** use Wilson score intervals for proportions.\n")
    
    report_content = "".join(report_lines)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    return report_content


def generate_summary_statistics(
    df: pd.DataFrame,
    group_cols: List[str] = ["subject", "miu"],
    value_col: str = "is_correct"
) -> Dict[str, Any]:
    """
    Generate summary statistics from results DataFrame.
    
    Args:
        df: DataFrame with results
        group_cols: Columns to generate statistics for
        value_col: Value column to summarize
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "total_records": len(df),
        "columns": list(df.columns),
        "generated_at": datetime.now().isoformat(),
    }
    
    # Overall statistics
    if value_col in df.columns:
        correct = df[value_col].sum()
        total = df[value_col].notna().sum()
        summary["overall_accuracy"] = correct / total if total > 0 else 0
        summary["total_correct"] = int(correct)
        summary["total_answered"] = int(total)
    
    # Group statistics
    for col in group_cols:
        if col in df.columns:
            unique_values = df[col].nunique()
            summary[f"{col}_count"] = unique_values
            summary[f"{col}_values"] = sorted(df[col].unique().tolist())
            
            if value_col in df.columns:
                group_stats = df.groupby(col)[value_col].agg(['sum', 'count', 'mean'])
                group_stats.columns = ['correct', 'total', 'accuracy']
                summary[f"{col}_statistics"] = group_stats.to_dict('index')
    
    return summary


def export_results_to_csv(
    results: Dict[str, pd.DataFrame],
    output_dir: Path,
    prefix: str = ""
) -> List[Path]:
    """
    Export multiple result DataFrames to CSV files.
    
    Args:
        results: Dictionary mapping names to DataFrames
        output_dir: Directory to save files
        prefix: Prefix for filenames
    
    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    for name, df in results.items():
        if isinstance(df, pd.DataFrame):
            filename = f"{prefix}{name}.csv" if prefix else f"{name}.csv"
            filepath = output_dir / filename
            df.to_csv(filepath, index=False)
            saved_files.append(filepath)
    
    return saved_files


def create_key_findings_summary(
    distortion_results: pd.DataFrame,
    subject_results: pd.DataFrame
) -> Dict[str, Any]:
    """
    Create a summary of key findings from the analysis.
    
    Args:
        distortion_results: Results from distortion analysis
        subject_results: Results from subject analysis
    
    Returns:
        Dictionary with key findings
    """
    findings = {
        "generated_at": datetime.now().isoformat(),
    }
    
    # Most significant distortion effects
    if 'p_value' in distortion_results.columns:
        most_sig = distortion_results.nsmallest(3, 'p_value')
        findings["most_significant_distortion_effects"] = most_sig.to_dict('records')
    
    # Most vulnerable subjects
    if 'degradation_percent' in subject_results.columns:
        most_vulnerable = subject_results.nlargest(5, 'degradation_percent')
        findings["most_vulnerable_subjects"] = most_vulnerable.to_dict('records')
    
    # Most resilient subjects
    if 'degradation_percent' in subject_results.columns:
        most_resilient = subject_results.nsmallest(5, 'degradation_percent')
        findings["most_resilient_subjects"] = most_resilient.to_dict('records')
    
    # Overall statistics
    if 'is_significant' in subject_results.columns:
        findings["subjects_significantly_affected"] = int(subject_results['is_significant'].sum())
        findings["subjects_not_significantly_affected"] = int((~subject_results['is_significant']).sum())
    
    return findings


