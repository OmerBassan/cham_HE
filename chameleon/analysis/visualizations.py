"""
Visualization generation for analysis results.

Creates:
- Degradation heatmaps
- Accuracy plots
- Key insights summaries
- Statistical significance visualizations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import warnings

warnings.filterwarnings('ignore')

# Plotting imports with fallback
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


def _check_plotting():
    """Check if plotting libraries are available."""
    if not HAS_PLOTTING:
        raise ImportError(
            "Plotting requires matplotlib and seaborn. "
            "Install with: pip install matplotlib seaborn"
        )


def setup_plot_style():
    """Set up consistent plot styling."""
    _check_plotting()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
    })


def create_degradation_heatmap(
    df: pd.DataFrame,
    subject_col: str = "subject",
    level_col: str = "miu",
    value_col: str = "degradation",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 16),
    title: str = "Performance Degradation from Baseline",
    cmap: str = "Reds"
) -> plt.Figure:
    """
    Create a heatmap showing degradation by subject and distortion level.
    
    Args:
        df: DataFrame with performance data
        subject_col: Column with subject names
        level_col: Column with distortion levels
        value_col: Column with values to display
        output_path: Path to save the figure
        figsize: Figure size
        title: Plot title
        cmap: Color map
    
    Returns:
        matplotlib Figure
    """
    _check_plotting()
    setup_plot_style()
    
    # Create pivot table
    pivot = df.pivot(index=subject_col, columns=level_col, values=value_col)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    mask = pivot.isna()
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.1f',
        cmap=cmap,
        center=pivot.values[~np.isnan(pivot.values)].mean() if not pivot.isna().all().all() else 0,
        mask=mask,
        cbar_kws={'label': f'{value_col.replace("_", " ").title()} (%)'},
        linewidths=0.5,
        ax=ax
    )
    
    # Customize labels
    y_labels = [label.replace('_', ' ').title() for label in pivot.index]
    x_labels = [f'μ={x:.1f}' for x in pivot.columns]
    
    ax.set_yticklabels(y_labels, rotation=0, fontsize=9)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlabel('Distortion Level (μ)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Academic Subject', fontweight='bold', fontsize=12)
    ax.set_title(f'{title}\n(Darker = Higher Degradation)', fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_accuracy_plots(
    df: pd.DataFrame,
    level_col: str = "miu",
    accuracy_col: str = "accuracy",
    subject_col: str = "subject",
    output_dir: Optional[Path] = None,
    prefix: str = ""
) -> List[plt.Figure]:
    """
    Create a set of accuracy-related plots.
    
    Args:
        df: DataFrame with performance data
        level_col: Column with distortion levels
        accuracy_col: Column with accuracy values
        subject_col: Column with subject names
        output_dir: Directory to save figures
        prefix: Prefix for output filenames
    
    Returns:
        List of matplotlib Figures
    """
    _check_plotting()
    setup_plot_style()
    
    figures = []
    
    # 1. Accuracy by distortion level
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    level_accuracy = df.groupby(level_col)[accuracy_col].mean()
    bars = ax1.bar(
        range(len(level_accuracy)),
        level_accuracy.values * 100,
        color=['#4CAF50' if x > 0.85 else '#FFC107' if x > 0.75 else '#F44336' 
               for x in level_accuracy.values]
    )
    
    ax1.set_xticks(range(len(level_accuracy)))
    ax1.set_xticklabels([f'μ={x:.1f}' for x in level_accuracy.index], fontsize=10)
    ax1.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Distortion Level (μ)', fontweight='bold', fontsize=12)
    ax1.set_title('Model Accuracy by Distortion Level', fontweight='bold', fontsize=14, pad=15)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(level_accuracy.values * 100):
        ax1.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=10)
    
    ax1.set_ylim(0, 105)
    plt.tight_layout()
    figures.append(fig1)
    
    if output_dir:
        fig1.savefig(output_dir / f'{prefix}accuracy_by_level.png', dpi=300, bbox_inches='tight')
    
    # 2. Subject ranking by average accuracy
    if subject_col in df.columns:
        fig2, ax2 = plt.subplots(figsize=(12, 10))
        
        subject_accuracy = df.groupby(subject_col)[accuracy_col].mean().sort_values(ascending=True)
        
        colors = ['#4CAF50' if x > 0.9 else '#8BC34A' if x > 0.85 else 
                  '#FFC107' if x > 0.8 else '#FF9800' if x > 0.75 else '#F44336' 
                  for x in subject_accuracy.values]
        
        ax2.barh(range(len(subject_accuracy)), subject_accuracy.values * 100, color=colors)
        ax2.set_yticks(range(len(subject_accuracy)))
        ax2.set_yticklabels([s.replace('_', ' ').title() for s in subject_accuracy.index], fontsize=9)
        ax2.set_xlabel('Accuracy (%)', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Subject', fontweight='bold', fontsize=12)
        ax2.set_title('Subject Ranking by Average Accuracy', fontweight='bold', fontsize=14, pad=15)
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        figures.append(fig2)
        
        if output_dir:
            fig2.savefig(output_dir / f'{prefix}subject_accuracy_ranking.png', dpi=300, bbox_inches='tight')
    
    return figures


def create_statistical_significance_plot(
    df: pd.DataFrame,
    level_col: str = "miu_level",
    baseline_acc_col: str = "baseline_accuracy",
    comp_acc_col: str = "comparison_accuracy",
    p_value_col: str = "p_value",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create a plot showing statistical significance of distortion effects.
    
    Args:
        df: DataFrame with McNemar test results
        level_col: Column with distortion levels
        baseline_acc_col: Column with baseline accuracy
        comp_acc_col: Column with comparison accuracy
        p_value_col: Column with p-values
        output_path: Path to save the figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    _check_plotting()
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x_pos = range(len(df))
    
    # Plot baseline and comparison accuracy with error bars
    ax.plot(x_pos, df[baseline_acc_col] * 100, 'o-', label='Baseline (μ=0.0)', 
            linewidth=2, markersize=8, color='#2196F3')
    ax.plot(x_pos, df[comp_acc_col] * 100, 's-', label='Distorted', 
            linewidth=2, markersize=8, color='#F44336')
    
    # Mark significance levels
    for i, row in df.iterrows():
        p = row[p_value_col]
        y = row[comp_acc_col] * 100 - 2
        
        if pd.notna(p):
            if p < 0.001:
                marker = '***'
            elif p < 0.01:
                marker = '**'
            elif p < 0.05:
                marker = '*'
            else:
                marker = ''
            
            if marker:
                ax.text(list(df.index).index(i), y, marker, ha='center', 
                       fontweight='bold', fontsize=12, color='#F44336')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'μ={x:.1f}' for x in df[level_col]], fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Distortion Level', fontweight='bold', fontsize=12)
    ax.set_title('Statistical Significance of Distortion Effects\n(* p<0.05, ** p<0.01, *** p<0.001)', 
                fontweight='bold', fontsize=14, pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_key_insights_summary(
    performance_df: pd.DataFrame,
    subject_col: str = "subject",
    level_col: str = "miu",
    degradation_col: str = "degradation",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create a 4-panel summary highlighting key insights.
    
    Args:
        performance_df: DataFrame with performance metrics
        subject_col: Column with subject names
        level_col: Column with distortion levels
        degradation_col: Column with degradation values
        output_path: Path to save the figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    _check_plotting()
    setup_plot_style()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Calculate aggregated data
    subject_avg_deg = performance_df.groupby(subject_col)[degradation_col].mean()
    level_avg_deg = performance_df.groupby(level_col)[degradation_col].mean()
    
    # 1. Most vulnerable subjects (top 10)
    worst = subject_avg_deg.sort_values(ascending=False).head(10)
    colors1 = ['#D32F2F' if x > 20 else '#FF5722' if x > 10 else '#FFC107' for x in worst.values]
    ax1.barh(range(len(worst)), worst.values, color=colors1)
    ax1.set_yticks(range(len(worst)))
    ax1.set_yticklabels([s.replace('_', ' ').title() for s in worst.index], fontsize=9)
    ax1.set_xlabel('Avg Degradation (%)', fontweight='bold')
    ax1.set_title('MOST VULNERABLE: Top 10 Subjects', fontweight='bold', color='#D32F2F')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Most resilient subjects (bottom 10)
    best = subject_avg_deg.sort_values(ascending=True).head(10)
    colors2 = ['#4CAF50' if x < 5 else '#8BC34A' if x < 10 else '#FFC107' for x in best.values]
    ax2.barh(range(len(best)), best.values, color=colors2)
    ax2.set_yticks(range(len(best)))
    ax2.set_yticklabels([s.replace('_', ' ').title() for s in best.index], fontsize=9)
    ax2.set_xlabel('Avg Degradation (%)', fontweight='bold')
    ax2.set_title('MOST RESILIENT: Top 10 Subjects', fontweight='bold', color='#388E3C')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Degradation by level
    level_avg_filtered = level_avg_deg[level_avg_deg.index > 0]  # Exclude baseline
    colors3 = ['#4CAF50' if x < 8 else '#FFC107' if x < 12 else '#F44336' for x in level_avg_filtered.values]
    bars = ax3.bar(range(len(level_avg_filtered)), level_avg_filtered.values, color=colors3)
    ax3.set_xticks(range(len(level_avg_filtered)))
    ax3.set_xticklabels([f'μ={x:.1f}' for x in level_avg_filtered.index], fontsize=9)
    ax3.set_ylabel('Avg Degradation (%)', fontweight='bold')
    ax3.set_title('DEGRADATION BY LEVEL', fontweight='bold', color='#FF5722')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(level_avg_filtered.values):
        ax3.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9)
    
    # 4. Extreme cases scatter
    extreme = performance_df.nlargest(15, degradation_col)
    scatter = ax4.scatter(
        extreme[level_col],
        extreme[degradation_col],
        s=100,
        c=extreme[degradation_col],
        cmap='Reds',
        alpha=0.8,
        edgecolors='black'
    )
    ax4.set_xlabel('Distortion Level (μ)', fontweight='bold')
    ax4.set_ylabel('Degradation (%)', fontweight='bold')
    ax4.set_title('EXTREME CASES: Worst Individual Drops', fontweight='bold', color='#D32F2F')
    ax4.grid(True, alpha=0.3)
    
    # Annotate worst case
    if len(extreme) > 0:
        worst_case = extreme.iloc[0]
        ax4.annotate(
            f'{worst_case[subject_col].replace("_", " ").title()}\n{worst_case[degradation_col]:.1f}%',
            xy=(worst_case[level_col], worst_case[degradation_col]),
            xytext=(worst_case[level_col] + 0.1, worst_case[degradation_col] - 5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=9, fontweight='bold', ha='center'
        )
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_subject_significance_heatmap(
    df: pd.DataFrame,
    subject_col: str = "subject_name",
    degradation_col: str = "degradation_percent",
    p_value_col: str = "p_value",
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 14)
) -> plt.Figure:
    """
    Create heatmap showing subject degradation with significance levels.
    
    Args:
        df: DataFrame with subject significance results
        subject_col: Column with subject names
        degradation_col: Column with degradation percentages
        p_value_col: Column with p-values
        output_path: Path to save the figure
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    _check_plotting()
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by degradation
    df_sorted = df.sort_values(degradation_col, ascending=False)
    
    # Create heatmap data
    degradation_values = df_sorted[degradation_col].values.reshape(-1, 1)
    
    # Create heatmap
    im = ax.imshow(degradation_values, cmap='Reds', aspect='auto')
    plt.colorbar(im, ax=ax, label='Degradation (%)', shrink=0.8)
    
    # Add significance annotations
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        p = row[p_value_col]
        if pd.isna(p):
            sig = 'N/A'
        elif p < 0.001:
            sig = 'p < 0.001'
        elif p < 0.01:
            sig = 'p < 0.01'
        elif p < 0.05:
            sig = 'p < 0.05'
        else:
            sig = 'n.s.'
        
        color = 'white' if row[degradation_col] > 20 else 'black'
        ax.text(0, i, sig, ha='center', va='center', fontweight='bold', color=color, fontsize=9)
    
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted[subject_col].values, fontsize=9)
    ax.set_xticks([])
    ax.set_title('Subject Degradation with Statistical Significance\n(Baseline vs μ=0.9)', 
                fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


