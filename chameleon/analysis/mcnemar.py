"""
McNemar's test implementation for paired comparisons.

McNemar's test is used for comparing paired binary outcomes, perfect for:
- Comparing model performance across distortion levels
- Testing if accuracy differences are statistically significant
- Analyzing subject-specific vulnerabilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import scipy.stats as stats

# Try to import statsmodels for McNemar test
try:
    from statsmodels.stats.contingency_tables import mcnemar as statsmodels_mcnemar
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


@dataclass
class McNemarResult:
    """Result from McNemar's test."""
    statistic: float
    p_value: float
    contingency_table: np.ndarray
    discordant_pairs: int
    odds_ratio: float
    group1_accuracy: float
    group2_accuracy: float
    accuracy_difference: float
    significance: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "discordant_pairs": self.discordant_pairs,
            "odds_ratio": self.odds_ratio,
            "group1_accuracy": self.group1_accuracy,
            "group2_accuracy": self.group2_accuracy,
            "accuracy_difference": self.accuracy_difference,
            "significance": self.significance,
        }


def mcnemar_test(
    group1_correct: np.ndarray,
    group2_correct: np.ndarray,
    exact: bool = False,
    correction: bool = True
) -> McNemarResult:
    """
    Perform McNemar's test for paired binary outcomes.
    
    Args:
        group1_correct: Boolean array of correct answers for condition 1
        group2_correct: Boolean array of correct answers for condition 2
        exact: Use exact binomial test (better for small samples)
        correction: Apply continuity correction
    
    Returns:
        McNemarResult with test statistics and interpretation
    """
    group1_correct = np.asarray(group1_correct, dtype=bool)
    group2_correct = np.asarray(group2_correct, dtype=bool)
    
    # Build contingency table
    # [[both_correct, g1_only], [g2_only, both_wrong]]
    both_correct = np.sum(group1_correct & group2_correct)     # a
    group1_only = np.sum(group1_correct & ~group2_correct)     # b
    group2_only = np.sum(~group1_correct & group2_correct)     # c
    both_wrong = np.sum(~group1_correct & ~group2_correct)     # d
    
    table = np.array([[both_correct, group1_only],
                      [group2_only, both_wrong]])
    
    # Perform McNemar's test
    discordant_pairs = group1_only + group2_only
    
    if HAS_STATSMODELS:
        try:
            if exact or discordant_pairs < 25:
                result = statsmodels_mcnemar(table, exact=True)
            else:
                result = statsmodels_mcnemar(table, exact=False, correction=correction)
            statistic = result.statistic
            p_value = result.pvalue
        except Exception:
            # Fallback to manual calculation
            statistic, p_value = _manual_mcnemar(group1_only, group2_only, correction)
    else:
        statistic, p_value = _manual_mcnemar(group1_only, group2_only, correction)
    
    # Calculate odds ratio
    if group2_only > 0:
        odds_ratio = group1_only / group2_only
    else:
        odds_ratio = np.inf if group1_only > 0 else 1.0
    
    # Calculate accuracies
    group1_acc = np.mean(group1_correct)
    group2_acc = np.mean(group2_correct)
    
    # Interpret significance
    significance = _interpret_p_value(p_value)
    
    return McNemarResult(
        statistic=statistic,
        p_value=p_value,
        contingency_table=table,
        discordant_pairs=discordant_pairs,
        odds_ratio=odds_ratio,
        group1_accuracy=group1_acc,
        group2_accuracy=group2_acc,
        accuracy_difference=group1_acc - group2_acc,
        significance=significance,
    )


def _manual_mcnemar(b: int, c: int, correction: bool = True) -> Tuple[float, float]:
    """Manual McNemar's test calculation."""
    if b + c == 0:
        return np.nan, np.nan
    
    if correction:
        # With continuity correction
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        statistic = (b - c) ** 2 / (b + c)
    
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    return statistic, p_value


def _interpret_p_value(p_value: float) -> str:
    """Convert p-value to significance interpretation."""
    if pd.isna(p_value):
        return "Unable to calculate"
    elif p_value < 0.001:
        return "Highly significant (p < 0.001) ***"
    elif p_value < 0.01:
        return "Very significant (p < 0.01) **"
    elif p_value < 0.05:
        return "Significant (p < 0.05) *"
    elif p_value < 0.1:
        return "Marginally significant (p < 0.1)"
    else:
        return "Not significant (p ≥ 0.1)"


def calculate_confidence_interval(
    accuracy: float,
    n: int,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate Wilson score confidence interval for a proportion.
    
    Args:
        accuracy: Proportion of correct answers
        n: Sample size
        confidence: Confidence level
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if n == 0:
        return (0.0, 0.0)
    
    z = stats.norm.ppf((1 + confidence) / 2)
    p = accuracy
    
    # Wilson score interval
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    
    return (max(0, center - margin), min(1, center + margin))


def analyze_distortion_significance(
    df: pd.DataFrame,
    baseline_col: str = "miu",
    baseline_value: float = 0.0,
    question_id_col: str = "question_id",
    subject_col: str = "subject",
    is_correct_col: str = "is_correct"
) -> pd.DataFrame:
    """
    Perform McNemar's tests comparing each distortion level to baseline.
    
    Handles multiple distortions per question by aggregating:
    - For baseline (miu=0): single question
    - For distorted (miu>0): aggregates by majority vote (>50% correct = correct)
    
    Args:
        df: DataFrame with results
        baseline_col: Column containing distortion level
        baseline_value: Baseline value to compare against
        question_id_col: Column for matching questions
        subject_col: Column for subject grouping
        is_correct_col: Column with correctness values
    
    Returns:
        DataFrame with test results for each distortion level
    """
    results = []
    
    # Get baseline data - one entry per question
    baseline_df = df[df[baseline_col] == baseline_value]
    
    # Create baseline index
    if subject_col in baseline_df.columns:
        baseline_correct = baseline_df.groupby([question_id_col, subject_col])[is_correct_col].first()
    else:
        baseline_correct = baseline_df.groupby(question_id_col)[is_correct_col].first()
    
    for level in sorted(df[baseline_col].unique()):
        if level == baseline_value:
            continue
        
        # Get comparison data - aggregate multiple distortions per question
        # Use majority vote: if >50% of distortions are correct, consider it correct
        comp_df = df[df[baseline_col] == level]
        
        if subject_col in comp_df.columns:
            comp_agg = comp_df.groupby([question_id_col, subject_col])[is_correct_col].mean()
            comp_correct = (comp_agg > 0.5)  # Majority vote
        else:
            comp_agg = comp_df.groupby(question_id_col)[is_correct_col].mean()
            comp_correct = (comp_agg > 0.5)
        
        # Find common questions
        common_idx = baseline_correct.index.intersection(comp_correct.index)
        
        if len(common_idx) == 0:
            continue
        
        # Perform McNemar's test
        baseline_paired = baseline_correct.loc[common_idx].astype(bool).values
        comp_paired = comp_correct.loc[common_idx].astype(bool).values
        
        result = mcnemar_test(baseline_paired, comp_paired)
        
        # Calculate confidence intervals
        n = len(common_idx)
        baseline_ci = calculate_confidence_interval(result.group1_accuracy, n)
        comp_ci = calculate_confidence_interval(result.group2_accuracy, n)
        
        # Also calculate raw accuracy (not majority vote)
        raw_comp_acc = comp_df[is_correct_col].mean()
        raw_baseline_acc = baseline_df[is_correct_col].mean()
        
        results.append({
            "comparison": f"{baseline_col}={baseline_value} vs {baseline_col}={level}",
            f"{baseline_col}_level": level,
            "baseline_accuracy": raw_baseline_acc,  # Use raw accuracy for display
            "comparison_accuracy": raw_comp_acc,
            "accuracy_difference": raw_baseline_acc - raw_comp_acc,
            "baseline_ci_lower": baseline_ci[0],
            "baseline_ci_upper": baseline_ci[1],
            "comparison_ci_lower": comp_ci[0],
            "comparison_ci_upper": comp_ci[1],
            "mcnemar_statistic": result.statistic,
            "p_value": result.p_value,
            "discordant_pairs": result.discordant_pairs,
            "significance": result.significance,
            "sample_size": n,
        })
    
    return pd.DataFrame(results)


def analyze_subject_significance(
    df: pd.DataFrame,
    subject_col: str = "subject",
    baseline_col: str = "miu",
    baseline_value: float = 0.0,
    comparison_value: float = 0.9,
    question_id_col: str = "question_id",
    is_correct_col: str = "is_correct"
) -> pd.DataFrame:
    """
    Perform McNemar's tests for each subject comparing baseline to high distortion.
    
    Handles multiple distortions per question by aggregating with majority vote.
    
    Args:
        df: DataFrame with results
        subject_col: Column with subject names
        baseline_col: Column containing distortion level
        baseline_value: Baseline distortion value
        comparison_value: Comparison distortion value (e.g., highest distortion)
        question_id_col: Column for matching questions
        is_correct_col: Column with correctness values
    
    Returns:
        DataFrame with test results per subject
    """
    results = []
    
    for subject in sorted(df[subject_col].unique()):
        subject_df = df[df[subject_col] == subject]
        
        baseline_df = subject_df[subject_df[baseline_col] == baseline_value]
        comp_df = subject_df[subject_df[baseline_col] == comparison_value]
        
        if len(baseline_df) == 0 or len(comp_df) == 0:
            continue
        
        # Aggregate baseline (should be one per question, but take first just in case)
        baseline_correct = baseline_df.groupby(question_id_col)[is_correct_col].first()
        
        # Aggregate comparison with majority vote
        comp_agg = comp_df.groupby(question_id_col)[is_correct_col].mean()
        comp_correct = (comp_agg > 0.5)  # Majority vote
        
        common_idx = baseline_correct.index.intersection(comp_correct.index)
        
        if len(common_idx) == 0:
            continue
        
        baseline_paired = baseline_correct.loc[common_idx].astype(bool).values
        comp_paired = comp_correct.loc[common_idx].astype(bool).values
        
        result = mcnemar_test(baseline_paired, comp_paired)
        
        # Calculate raw accuracies for display
        raw_baseline_acc = baseline_df[is_correct_col].mean()
        raw_comp_acc = comp_df[is_correct_col].mean()
        degradation_pct = (raw_baseline_acc - raw_comp_acc) * 100
        
        n = len(common_idx)
        baseline_ci = calculate_confidence_interval(raw_baseline_acc, len(baseline_df))
        comp_ci = calculate_confidence_interval(raw_comp_acc, len(comp_df))
        
        results.append({
            "subject": subject,
            "subject_name": subject.replace("_", " ").title(),
            "baseline_accuracy": raw_baseline_acc,
            "comparison_accuracy": raw_comp_acc,
            "degradation_percent": degradation_pct,
            "baseline_ci_lower": baseline_ci[0],
            "baseline_ci_upper": baseline_ci[1],
            "comparison_ci_lower": comp_ci[0],
            "comparison_ci_upper": comp_ci[1],
            "mcnemar_statistic": result.statistic,
            "p_value": result.p_value,
            "significance": result.significance,
            "sample_size": n,
            "is_significant": result.p_value < 0.05 if not pd.isna(result.p_value) else False,
        })
    
    return pd.DataFrame(results).sort_values("degradation_percent", ascending=False)


def analyze_pairwise_levels(
    df: pd.DataFrame,
    level_col: str = "miu",
    question_id_col: str = "question_id",
    subject_col: str = "subject",
    is_correct_col: str = "is_correct"
) -> pd.DataFrame:
    """
    Perform pairwise McNemar's tests between adjacent distortion levels.
    
    Handles multiple distortions per question by aggregating with majority vote.
    
    Args:
        df: DataFrame with results
        level_col: Column with distortion levels
        question_id_col: Column for matching questions
        subject_col: Column for subject grouping
        is_correct_col: Column with correctness values
    
    Returns:
        DataFrame with pairwise comparison results
    """
    results = []
    levels = sorted(df[level_col].unique())
    
    for i in range(len(levels) - 1):
        level1 = levels[i]
        level2 = levels[i + 1]
        
        df1 = df[df[level_col] == level1]
        df2 = df[df[level_col] == level2]
        
        # Aggregate by question with majority vote
        if subject_col in df.columns:
            agg1 = df1.groupby([question_id_col, subject_col])[is_correct_col].mean()
            agg2 = df2.groupby([question_id_col, subject_col])[is_correct_col].mean()
        else:
            agg1 = df1.groupby(question_id_col)[is_correct_col].mean()
            agg2 = df2.groupby(question_id_col)[is_correct_col].mean()
        
        correct1 = (agg1 > 0.5) if level1 > 0 else (agg1 > 0.5)  # Majority vote for distorted
        correct2 = (agg2 > 0.5)
        
        # For baseline (level1=0), use first value since there's only one
        if level1 == 0:
            if subject_col in df.columns:
                correct1 = df1.groupby([question_id_col, subject_col])[is_correct_col].first().astype(bool)
            else:
                correct1 = df1.groupby(question_id_col)[is_correct_col].first().astype(bool)
        
        common_idx = correct1.index.intersection(correct2.index)
        
        if len(common_idx) == 0:
            continue
        
        paired1 = correct1.loc[common_idx].values
        paired2 = correct2.loc[common_idx].values
        
        result = mcnemar_test(paired1, paired2)
        
        # Raw accuracies for display
        raw_acc1 = df1[is_correct_col].mean()
        raw_acc2 = df2[is_correct_col].mean()
        
        results.append({
            "comparison": f"{level_col}={level1} vs {level_col}={level2}",
            "level1": level1,
            "level2": level2,
            "level1_accuracy": raw_acc1,
            "level2_accuracy": raw_acc2,
            "accuracy_difference": raw_acc1 - raw_acc2,
            "mcnemar_statistic": result.statistic,
            "p_value": result.p_value,
            "significance": result.significance,
            "sample_size": len(common_idx),
            "is_significant": result.p_value < 0.05 if not pd.isna(result.p_value) else False,
        })
    
    return pd.DataFrame(results)


