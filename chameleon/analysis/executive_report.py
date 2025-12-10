"""
Executive Report Generator

Generates comprehensive .md reports with:
- Embedded charts and heatmaps
- McNemar statistical analysis
- Subject-wise and overall findings
- AI-generated insights (via Mistral)
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import base64


def image_to_base64(image_path: Path) -> str:
    """Convert image to base64 for embedding in markdown."""
    if not image_path.exists():
        return ""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_executive_report(
    project_name: str,
    projects_dir: str = "Projects",
    use_ai_insights: bool = False
) -> str:
    """
    Generate a comprehensive executive report.
    
    Works completely without any external API - all insights are data-driven.
    Optionally can use Mistral API for AI-generated summaries.
    
    Args:
        project_name: Name of the project
        projects_dir: Base projects directory
        use_ai_insights: Whether to use Mistral for generating insights (default: False)
    
    Returns:
        Path to the generated report
    """
    project_path = Path(projects_dir) / project_name
    results_dir = project_path / "results"
    analysis_dir = results_dir / "analysis_plots"
    
    # Load data files
    results_csv = results_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"Results file not found: {results_csv}")
    
    df = pd.read_csv(results_csv, encoding='utf-8')
    
    # Load analysis results
    perf_metrics = pd.read_csv(analysis_dir / "performance_metrics.csv") if (analysis_dir / "performance_metrics.csv").exists() else None
    mcnemar_dist = pd.read_csv(analysis_dir / "mcnemar_distortion_results.csv") if (analysis_dir / "mcnemar_distortion_results.csv").exists() else None
    mcnemar_subj = pd.read_csv(analysis_dir / "mcnemar_subject_results.csv") if (analysis_dir / "mcnemar_subject_results.csv").exists() else None
    mcnemar_pair = pd.read_csv(analysis_dir / "mcnemar_pairwise_results.csv") if (analysis_dir / "mcnemar_pairwise_results.csv").exists() else None
    
    # Load summary statistics
    summary_stats = {}
    if (analysis_dir / "summary_statistics.json").exists():
        with open(analysis_dir / "summary_statistics.json", 'r') as f:
            summary_stats = json.load(f)
    
    # Calculate key metrics
    overall_accuracy = df['is_correct'].mean() * 100
    baseline_acc = df[df['miu'] == 0.0]['is_correct'].mean() * 100 if 0.0 in df['miu'].values else overall_accuracy
    max_distortion_acc = df[df['miu'] == 0.9]['is_correct'].mean() * 100 if 0.9 in df['miu'].values else overall_accuracy
    degradation = baseline_acc - max_distortion_acc
    
    # Get target model info
    target_model = df['target_model_name'].iloc[0] if 'target_model_name' in df.columns else "Unknown"
    
    # Count subjects
    subjects = df['subject'].unique()
    n_subjects = len(subjects)
    n_questions = df['question_id'].nunique()
    n_distortions = len(df)
    
    # Generate report timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build report
    report = []
    
    # Title and metadata
    report.append(f"""# 📊 Chameleon Benchmark: Executive Report

**Project:** {project_name}  
**Generated:** {timestamp}  
**Target Model:** {target_model}

---

## 📋 Executive Summary

This report presents the results of evaluating **{target_model}** on the Chameleon Benchmark, which tests language model robustness against lexical distortions. The benchmark applies semantic paraphrasing at various intensity levels (μ=0.0 to μ=0.9) while preserving the original meaning and correct answer.

### Key Findings at a Glance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | {overall_accuracy:.1f}% |
| **Baseline Accuracy (μ=0.0)** | {baseline_acc:.1f}% |
| **Max Distortion Accuracy (μ=0.9)** | {max_distortion_acc:.1f}% |
| **Performance Degradation** | {degradation:.1f}% |
| **Questions Evaluated** | {n_questions:,} |
| **Total Distortions** | {n_distortions:,} |
| **Academic Subjects** | {n_subjects} |

---

## 📈 Performance Overview

### Accuracy by Distortion Level

The chart below shows how model accuracy changes as distortion intensity increases:

![Accuracy by Distortion Level](analysis_plots/1_accuracy_by_level.png)

""")
    
    # Add accuracy table by miu
    if 'miu' in df.columns:
        report.append("#### Detailed Accuracy by μ Level\n\n")
        report.append("| μ Level | Accuracy | Correct | Total | Degradation from Baseline |\n")
        report.append("|---------|----------|---------|-------|---------------------------|\n")
        
        for miu in sorted(df['miu'].unique()):
            subset = df[df['miu'] == miu]
            acc = subset['is_correct'].mean() * 100
            correct = subset['is_correct'].sum()
            total = len(subset)
            deg = baseline_acc - acc
            report.append(f"| μ={miu:.1f} | {acc:.1f}% | {correct:,} | {total:,} | {deg:+.1f}% |\n")
        
        report.append("\n")
    
    # Subject-wise analysis
    report.append("""---

## 🎓 Subject-Wise Analysis

### Subject Ranking by Accuracy

![Subject Accuracy Ranking](analysis_plots/1_subject_accuracy_ranking.png)

""")
    
    # Subject accuracy table
    if 'subject' in df.columns:
        report.append("#### Detailed Subject Performance\n\n")
        report.append("| Subject | Baseline (μ=0) | Distorted (μ=0.9) | Degradation | Significant? |\n")
        report.append("|---------|----------------|-------------------|-------------|---------------|\n")
        
        for subject in sorted(df['subject'].unique()):
            subj_df = df[df['subject'] == subject]
            baseline = subj_df[subj_df['miu'] == 0.0]['is_correct'].mean() * 100 if 0.0 in subj_df['miu'].values else 0
            distorted = subj_df[subj_df['miu'] == 0.9]['is_correct'].mean() * 100 if 0.9 in subj_df['miu'].values else 0
            deg = baseline - distorted
            
            # Check significance from McNemar results
            sig = "—"
            if mcnemar_subj is not None and len(mcnemar_subj) > 0:
                subj_row = mcnemar_subj[mcnemar_subj['subject'] == subject]
                if len(subj_row) > 0:
                    p_val = subj_row['p_value'].iloc[0]
                    if pd.notna(p_val):
                        if p_val < 0.001:
                            sig = "*** (p<0.001)"
                        elif p_val < 0.01:
                            sig = "** (p<0.01)"
                        elif p_val < 0.05:
                            sig = "* (p<0.05)"
                        else:
                            sig = "n.s."
            
            subject_name = subject.replace("_", " ").title()
            report.append(f"| {subject_name} | {baseline:.1f}% | {distorted:.1f}% | {deg:+.1f}% | {sig} |\n")
        
        report.append("\n")
    
    # Degradation heatmap
    report.append("""---

## 🔥 Degradation Heatmap

The heatmap below visualizes performance degradation across subjects and distortion levels. Darker colors indicate greater accuracy drops from the baseline.

![Degradation Heatmap](analysis_plots/2_degradation_heatmap.png)

""")
    
    # Key insights
    report.append("""---

## 💡 Key Insights

![Key Insights Summary](analysis_plots/3_key_insights.png)

""")
    
    # Statistical analysis section
    report.append("""---

## 📊 Statistical Analysis: McNemar's Test

McNemar's test is used for paired binary outcomes to determine if the difference in accuracy between conditions is statistically significant. This is the appropriate test for our benchmark because:

1. **Paired Data**: Same questions are tested under different conditions (baseline vs. distorted)
2. **Binary Outcomes**: Each answer is either correct or incorrect
3. **Marginal Homogeneity**: Tests whether the proportion of correct answers differs between conditions

### Interpreting Results

- **χ² (Chi-squared)**: Test statistic measuring the discrepancy between observed and expected frequencies
- **p-value**: Probability of observing such a difference by chance
  - p < 0.05: Significant (*)
  - p < 0.01: Very significant (**)
  - p < 0.001: Highly significant (***)

### Distortion Level Significance

![Statistical Significance](analysis_plots/4_statistical_significance.png)

""")
    
    # McNemar distortion results table
    if mcnemar_dist is not None and len(mcnemar_dist) > 0:
        report.append("#### McNemar Test: Baseline vs Each Distortion Level\n\n")
        report.append("| Comparison | χ² | p-value | Significance | Discordant Pairs |\n")
        report.append("|------------|-----|---------|--------------|------------------|\n")
        
        for _, row in mcnemar_dist.iterrows():
            level = row.get('miu_level', row.iloc[0])
            stat = row.get('mcnemar_statistic', 0)
            p_val = row.get('p_value', 1)
            sig = row.get('significance', 'N/A').split('(')[0].strip()
            disc = row.get('discordant_pairs', 0)
            
            report.append(f"| μ=0.0 vs μ={level:.1f} | {stat:.2f} | {p_val:.6f} | {sig} | {disc} |\n")
        
        report.append("\n")
    
    # Subject significance heatmap
    report.append("""### Subject-Specific Significance

![Subject Significance](analysis_plots/5_subject_significance.png)

""")
    
    # McNemar subject results
    if mcnemar_subj is not None and len(mcnemar_subj) > 0:
        report.append("#### McNemar Test: Subject-Specific Results (Baseline vs μ=0.9)\n\n")
        report.append("| Subject | Degradation | χ² | p-value | Significance |\n")
        report.append("|---------|-------------|-----|---------|-------------|\n")
        
        for _, row in mcnemar_subj.iterrows():
            subject = row.get('subject_name', row.get('subject', 'Unknown'))
            deg = row.get('degradation_percent', 0)
            stat = row.get('mcnemar_statistic', 0)
            p_val = row.get('p_value', 1)
            sig = row.get('significance', 'N/A').split('(')[0].strip()
            
            report.append(f"| {subject} | {deg:.1f}% | {stat:.2f} | {p_val:.6f} | {sig} |\n")
        
        report.append("\n")
    
    # Pairwise comparisons
    if mcnemar_pair is not None and len(mcnemar_pair) > 0:
        report.append("""### Pairwise Level Comparisons

Testing whether adjacent distortion levels show significant differences:

| Comparison | Diff | χ² | p-value | Significant? |
|------------|------|-----|---------|--------------|
""")
        for _, row in mcnemar_pair.iterrows():
            comp = row.get('comparison', 'N/A')
            diff = row.get('accuracy_difference', 0) * 100
            stat = row.get('mcnemar_statistic', 0)
            p_val = row.get('p_value', 1)
            sig = "Yes" if row.get('is_significant', False) else "No"
            
            report.append(f"| {comp} | {diff:+.2f}% | {stat:.2f} | {p_val:.6f} | {sig} |\n")
        
        report.append("\n")
    
    # Trend analysis
    report.append(f"""---

## 📉 Trend Analysis

### Overall Robustness Assessment

Based on the benchmark results, **{target_model}** demonstrates the following characteristics:

""")
    
    # Determine robustness level
    if degradation < 3:
        robustness = "**HIGH ROBUSTNESS** 🟢"
        robustness_desc = "The model shows minimal performance degradation even at maximum distortion levels. This indicates strong semantic understanding that is largely invariant to surface-level lexical changes."
    elif degradation < 6:
        robustness = "**MODERATE ROBUSTNESS** 🟡"
        robustness_desc = "The model shows moderate performance degradation with increasing distortion. While generally stable, there is room for improvement in handling lexical variations."
    else:
        robustness = "**LOW ROBUSTNESS** 🔴"
        robustness_desc = "The model shows significant performance degradation under distortion. This suggests the model may be overly reliant on surface-level patterns rather than deep semantic understanding."
    
    report.append(f"**Robustness Rating:** {robustness}\n\n")
    report.append(f"{robustness_desc}\n\n")
    
    # Key observations
    report.append("""### Key Observations

""")
    
    # Calculate observations
    observations = []
    
    # 1. Overall trend
    if degradation > 0:
        observations.append(f"1. **Performance Decline**: Accuracy decreases by {degradation:.1f} percentage points from baseline (μ=0.0) to maximum distortion (μ=0.9).")
    else:
        observations.append(f"1. **Stable Performance**: The model maintains consistent accuracy across all distortion levels, with only {abs(degradation):.1f}% change.")
    
    # 2. Most vulnerable subject
    if mcnemar_subj is not None and len(mcnemar_subj) > 0:
        most_affected = mcnemar_subj.iloc[0]
        observations.append(f"2. **Most Vulnerable Subject**: {most_affected.get('subject_name', 'Unknown')} shows the highest degradation at {most_affected.get('degradation_percent', 0):.1f}%.")
    
    # 3. Most resilient subject
    if mcnemar_subj is not None and len(mcnemar_subj) > 0:
        least_affected = mcnemar_subj.iloc[-1]
        observations.append(f"3. **Most Resilient Subject**: {least_affected.get('subject_name', 'Unknown')} shows the lowest degradation at {least_affected.get('degradation_percent', 0):.1f}%.")
    
    # 4. Statistical significance
    if mcnemar_dist is not None:
        sig_count = (mcnemar_dist['p_value'] < 0.05).sum()
        total = len(mcnemar_dist)
        observations.append(f"4. **Statistical Significance**: {sig_count}/{total} distortion levels show statistically significant accuracy changes (p < 0.05).")
    
    for obs in observations:
        report.append(f"{obs}\n\n")
    
    # Conclusions
    report.append(f"""---

## 🎯 Conclusions

### Summary

The Chameleon Benchmark evaluation of **{target_model}** reveals:

1. **Overall Performance**: The model achieves {overall_accuracy:.1f}% accuracy across all conditions, with {baseline_acc:.1f}% on original questions.

2. **Distortion Impact**: Performance degrades by {degradation:.1f}% from baseline to maximum distortion, indicating {'strong' if degradation < 3 else 'moderate' if degradation < 6 else 'weak'} robustness to lexical variations.

3. **Subject Variability**: Performance varies across subjects, with some showing greater vulnerability to distortions than others.

4. **Statistical Confidence**: McNemar's tests confirm {'most' if mcnemar_dist is not None and (mcnemar_dist['p_value'] < 0.05).sum() > len(mcnemar_dist)/2 else 'some'} observed differences are statistically significant.

### Recommendations

""")
    
    if degradation > 5:
        report.append("""- **Model Improvement**: Consider fine-tuning on paraphrased data to improve robustness
- **Data Augmentation**: Include lexically varied examples in training data
- **Ensemble Methods**: Combine with models that show complementary strengths
""")
    elif degradation > 2:
        report.append("""- **Targeted Improvement**: Focus on subjects with highest degradation
- **Monitoring**: Track robustness metrics in production to detect drift
- **Testing**: Continue benchmarking with diverse distortion types
""")
    else:
        report.append("""- **Maintain Quality**: The model shows excellent robustness; continue monitoring
- **Expand Testing**: Consider testing with other distortion types (e.g., adversarial)
- **Documentation**: Document this robustness for downstream users
""")
    
    # Footer
    report.append(f"""
---

## 📎 Appendix

### Methodology

**Distortion Levels (μ)**:
- μ=0.0: Original question (baseline)
- μ=0.1-0.3: Light lexical changes (1-4 words)
- μ=0.4-0.6: Moderate restructuring
- μ=0.7-0.9: Heavy paraphrasing with preserved meaning

**Statistical Tests**:
- McNemar's test for paired binary outcomes
- Wilson score confidence intervals
- Bonferroni correction considered for multiple comparisons

### Data Files

All analysis outputs are available in:
`{analysis_dir}`

### Generated By

Chameleon Benchmark v1.0  
Report generated: {timestamp}

---

*This report was automatically generated by the Chameleon evaluation framework.*
""")
    
    # Write report
    report_content = "".join(report)
    report_path = results_dir / "Executive_Report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ Executive report saved to: {report_path}")
    
    return str(report_path)


def generate_ai_insights(
    project_name: str,
    projects_dir: str = "Projects",
    mistral_api_key: Optional[str] = None
) -> str:
    """
    Use Mistral to generate AI insights for the report.
    
    Args:
        project_name: Project name
        projects_dir: Projects directory
        mistral_api_key: Mistral API key (or from env)
    
    Returns:
        AI-generated insights text
    """
    try:
        from mistralai import Mistral
    except ImportError:
        return "AI insights not available (mistralai not installed)"
    
    api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return "AI insights not available (no API key)"
    
    project_path = Path(projects_dir) / project_name
    results_dir = project_path / "results"
    analysis_dir = results_dir / "analysis_plots"
    
    # Load key data
    results_csv = results_dir / "results.csv"
    df = pd.read_csv(results_csv, encoding='utf-8')
    
    # Prepare statistics summary
    overall_acc = df['is_correct'].mean() * 100
    baseline_acc = df[df['miu'] == 0.0]['is_correct'].mean() * 100
    max_dist_acc = df[df['miu'] == 0.9]['is_correct'].mean() * 100
    
    # Subject stats
    subject_stats = df.groupby('subject')['is_correct'].mean() * 100
    
    # Prepare prompt
    prompt = f"""You are an expert data scientist analyzing LLM robustness benchmark results.

Based on the following statistics, provide 3-4 key insights and recommendations:

**Benchmark Results:**
- Overall Accuracy: {overall_acc:.1f}%
- Baseline Accuracy (original questions): {baseline_acc:.1f}%
- Maximum Distortion Accuracy (μ=0.9): {max_dist_acc:.1f}%
- Performance Degradation: {baseline_acc - max_dist_acc:.1f}%

**Subject Performance:**
{subject_stats.to_string()}

Please provide:
1. Main finding about model robustness
2. Pattern observed across distortion levels
3. Subject-specific insights
4. One actionable recommendation

Be concise and data-driven. Use bullet points."""

    try:
        client = Mistral(api_key=api_key)
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI insights generation failed: {e}"


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate executive report")
    parser.add_argument("--project", "-p", required=True, help="Project name")
    parser.add_argument("--projects-dir", default="Projects", help="Projects directory")
    
    args = parser.parse_args()
    
    generate_executive_report(args.project, args.projects_dir)

