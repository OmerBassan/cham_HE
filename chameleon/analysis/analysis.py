#!/usr/bin/env python
"""
Chameleon Analysis Engine
=========================
Single modular analysis script that generates ALL analyses.
All outputs to results/analysis/

Outputs (~20 files):
├── 01_accuracy_by_miu.csv/png         - Accuracy curve by distortion level
├── 02_accuracy_by_subject_miu.csv     - Subject × μ accuracy table
├── 02_subject_ranking.png             - Subject performance bar chart
├── 02_subject_miu_heatmap.png         - Subject × μ accuracy heatmap  
├── 02_degradation_heatmap.png         - Subject × μ degradation heatmap
├── 03_chameleon_robustness_index.csv  - CRI scores (global + per-subject)
├── 04_elasticity.csv/png              - Degradation slope analysis
├── 05_model_comparison.csv/png        - Model comparison table & scatter
├── 06_error_taxonomy.json             - Error type classification
├── 07_confusion_clusters.json         - TF-IDF clustering of failures
├── 08_bootstrap_intervals.csv         - 95% confidence intervals
├── 09_delta_accuracy_heatmap.csv/png  - Change from baseline heatmap
├── 10_question_difficulty_tiers.json  - Easy/Medium/Hard/Breakers
├── 11_mcnemar_distortion.csv/png      - McNemar test by μ level
├── 12_mcnemar_subject.csv/png         - McNemar test by subject
├── 13_key_insights.png                - 4-panel summary figure
└── EXECUTIVE_REPORT.md                - Comprehensive markdown report

Usage:
    python -m chameleon.analysis.analysis --project MyProject
    
    # Or via CLI:
    python cli.py analyze --project MyProject
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional sklearn for clustering
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Optional statsmodels for McNemar
try:
    from statsmodels.stats.contingency_tables import mcnemar
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# =============================================================================
# HELPERS
# =============================================================================

def ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def coerce_bool(series: pd.Series) -> pd.Series:
    """Coerce mixed bool/string to proper boolean."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False})
    )


def setup_plot_style():
    """Configure consistent plot styling."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'text.color': '#333333',
        'font.family': 'sans-serif',
        'font.size': 10,
    })


def load_results(path: Path) -> pd.DataFrame:
    """Load and clean results (supports CSV or JSONL)."""
    if path.suffix == '.jsonl':
        df = pd.read_json(path, orient='records', lines=True)
    else:
        df = pd.read_csv(path)
    
    # Ensure numeric miu
    df["miu"] = pd.to_numeric(df["miu"], errors="coerce").astype(float)
    
    # Ensure boolean is_correct
    if "is_correct" in df.columns:
        df["is_correct"] = coerce_bool(df["is_correct"]).fillna(False).astype(bool)
    else:
        df["is_correct"] = False
    
    # Filter invalid model names
    if "target_model_name" in df.columns:
        df["target_model_name"] = df["target_model_name"].astype(str)
        df = df[~df["target_model_name"].isin(["nan", "None", "", "NaN", "null"])].copy()
    else:
        df["target_model_name"] = "unknown"
    
    # Ensure subject
    if "subject" not in df.columns:
        df["subject"] = "General"
    
    # Drop invalid rows
    df = df.dropna(subset=["miu"])
    
    return df


# =============================================================================
# CORE METRICS
# =============================================================================

def compute_accuracy_by_miu(df: pd.DataFrame) -> pd.DataFrame:
    """01: Accuracy by model and miu."""
    return (
        df.groupby(["target_model_name", "miu"])["is_correct"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n"})
        .sort_values(["target_model_name", "miu"])
    )


def compute_accuracy_by_subject_miu(df: pd.DataFrame) -> pd.DataFrame:
    """02: Accuracy by subject and miu."""
    return (
        df.groupby(["target_model_name", "subject", "miu"])["is_correct"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n"})
        .sort_values(["target_model_name", "subject", "miu"])
    )


def compute_cri(df: pd.DataFrame, group_cols: List[str], alpha: float = 2.0) -> pd.DataFrame:
    """03: Chameleon Robustness Index."""
    mus = np.sort(df["miu"].unique())
    weights_raw = np.exp(alpha * mus)
    weights = weights_raw / weights_raw.sum()
    weight_map = dict(zip(mus, weights))
    
    rows = []
    for keys, group in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        
        acc_by_miu = group.groupby("miu")["is_correct"].mean()
        
        cri = sum(acc * weight_map.get(mu, 0) for mu, acc in acc_by_miu.items())
        total_weight = sum(weight_map.get(mu, 0) for mu in acc_by_miu.index)
        cri = cri / total_weight if total_weight > 0 else np.nan
        
        row = dict(zip(group_cols, keys))
        row["CRI"] = float(cri) if not np.isnan(cri) else np.nan
        row["n_samples"] = len(group)
        rows.append(row)
    
    return pd.DataFrame(rows).sort_values("CRI", ascending=False)


def compute_elasticity(acc_df: pd.DataFrame) -> pd.DataFrame:
    """04: Elasticity (degradation slope)."""
    rows = []
    for model, group in acc_df.groupby("target_model_name"):
        sub = group.sort_values("miu").copy()
        x = pd.to_numeric(sub["miu"], errors="coerce").astype(float).values
        y = pd.to_numeric(sub["accuracy"], errors="coerce").astype(float).values
        
        valid = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid], y[valid]
        
        if len(np.unique(x)) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
        else:
            slope, intercept = np.nan, np.nan
        
        rows.append({
            "target_model_name": model,
            "elasticity_slope": float(slope) if not np.isnan(slope) else np.nan,
            "intercept": float(intercept) if not np.isnan(intercept) else np.nan,
            "baseline_accuracy": float(intercept) if not np.isnan(intercept) else np.nan,
        })
    
    return pd.DataFrame(rows).sort_values("elasticity_slope", ascending=False)


def build_model_comparison(df: pd.DataFrame, acc_df: pd.DataFrame, 
                           cri_df: pd.DataFrame, elast_df: pd.DataFrame) -> pd.DataFrame:
    """05: Model comparison table."""
    mus = np.sort(acc_df["miu"].unique())
    base_mu, high_mu = mus[0], mus[-1]
    
    rows = []
    for model in df["target_model_name"].unique():
        sub_acc = acc_df[acc_df["target_model_name"] == model]
        sub_df = df[df["target_model_name"] == model]
        
        def get_acc(target_mu):
            if len(sub_acc) == 0:
                return np.nan
            closest = sub_acc.iloc[(sub_acc["miu"] - target_mu).abs().argsort().iloc[0]]
            return float(closest["accuracy"])
        
        cri_row = cri_df[cri_df["target_model_name"] == model]
        cri_val = float(cri_row["CRI"].iloc[0]) if len(cri_row) > 0 else np.nan
        
        elast_row = elast_df[elast_df["target_model_name"] == model]
        elast_val = float(elast_row["elasticity_slope"].iloc[0]) if len(elast_row) > 0 else np.nan
        
        base_acc, high_acc = get_acc(base_mu), get_acc(high_mu)
        degradation = (base_acc - high_acc) / base_acc * 100 if base_acc > 0 else np.nan
        
        rows.append({
            "target_model_name": model,
            "n_samples": len(sub_df),
            "overall_accuracy": float(sub_df["is_correct"].mean()),
            f"accuracy_mu_{base_mu:.1f}": base_acc,
            f"accuracy_mu_{high_mu:.1f}": high_acc,
            "degradation_pct": degradation,
            "CRI": cri_val,
            "elasticity_slope": elast_val,
        })
    
    return pd.DataFrame(rows).sort_values("CRI", ascending=False)


# =============================================================================
# ERROR ANALYSIS
# =============================================================================

def classify_error(row: pd.Series) -> str:
    """Classify error type."""
    if bool(row["is_correct"]):
        return "correct"
    
    ans = str(row.get("target_model_answer", "")).strip()
    if ans == "" or ans.lower() in {"none", "nan", "null", "n/a"}:
        return "blank_answer"
    
    if any(sep in ans for sep in [",", "/", " and ", " or ", ";"]):
        return "multiple_options"
    
    # try:
    #     options = json.loads(str(row.get("options_json", "{}")))
    #     if isinstance(options, dict) and len(ans) == 1 and ans.upper() in options:
    #         return "wrong_choice"
    # except:
    #     pass
    
    return "invalid_format"


def build_error_taxonomy(df: pd.DataFrame) -> Dict:
    """06: Error taxonomy."""
    df = df.copy()
    df["error_type"] = df.apply(classify_error, axis=1)
    
    total = len(df)
    global_counts = df["error_type"].value_counts().to_dict()
    
    # Per miu bucket
    incorrect = df[df["is_correct"] == False].copy()
    incorrect["miu_bucket"] = incorrect["miu"].round(1)
    per_miu = {}
    for (bucket, etype), sub in incorrect.groupby(["miu_bucket", "error_type"]):
        per_miu.setdefault(str(bucket), {})[etype] = len(sub)
    
    return {
        "total_samples": total,
        "global_counts": {k: int(v) for k, v in global_counts.items()},
        "global_frequencies": {k: v / total for k, v in global_counts.items()},
        "per_miu_bucket": per_miu,
    }


def build_confusion_clusters(df: pd.DataFrame, max_clusters: int = 8) -> Optional[Dict]:
    """07: TF-IDF confusion clusters."""
    if not SKLEARN_AVAILABLE:
        return {"error": "sklearn not installed"}
    
    incorrect = df[df["is_correct"] == False].dropna(subset=["distorted_question"])
    if len(incorrect) < 20:
        return {"error": f"Only {len(incorrect)} incorrect samples"}
    
    texts = incorrect["distorted_question"].astype(str).tolist()
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(texts)
    
    n_clusters = min(max_clusters, max(2, len(incorrect) // 50))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    
    incorrect = incorrect.reset_index(drop=True)
    terms = np.array(vectorizer.get_feature_names_out())
    
    clusters = []
    for cid in range(n_clusters):
        mask = labels == cid
        if mask.sum() == 0:
            continue
        
        center = km.cluster_centers_[cid]
        top_terms = terms[center.argsort()[-10:][::-1]].tolist()
        
        clusters.append({
            "cluster_id": cid,
            "size": int(mask.sum()),
            "top_terms": top_terms,
        })
    
    return {"n_clusters": len(clusters), "clusters": sorted(clusters, key=lambda x: -x["size"])}


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def bootstrap_ci(df: pd.DataFrame, group_cols: List[str], n_boot: int = 500) -> pd.DataFrame:
    """08: Bootstrap confidence intervals."""
    rng = np.random.default_rng(42)
    rows = []
    
    for keys, group in df.groupby(group_cols) if group_cols else [("global", df)]:
        arr = group["is_correct"].astype(int).values
        n = len(arr)
        if n == 0:
            continue
        
        means = [rng.choice(arr, size=n, replace=True).mean() for _ in range(n_boot)]
        
        if not isinstance(keys, tuple):
            keys = (keys,)
        
        row = dict(zip(group_cols if group_cols else ["level"], keys))
        row.update({
            "n": n,
            "accuracy": float(arr.mean()),
            "ci_lower": float(np.quantile(means, 0.025)),
            "ci_upper": float(np.quantile(means, 0.975)),
        })
        rows.append(row)
    
    return pd.DataFrame(rows)


def mcnemar_distortion(df: pd.DataFrame) -> pd.DataFrame:
    """11: McNemar test - baseline vs each miu level."""
    if not STATSMODELS_AVAILABLE:
        return pd.DataFrame()
    
    results = []
    mus = sorted([m for m in df["miu"].unique() if m > 0])
    
    for miu in mus:
        baseline = df[df["miu"] == 0.0].set_index("question_id")["is_correct"]
        comparison = df[df["miu"] == miu].groupby("question_id")["is_correct"].first()
        
        common = baseline.index.intersection(comparison.index)
        if len(common) < 10:
            continue
        
        b, c = baseline.loc[common].values, comparison.loc[common].values
        
        # Contingency table
        n00 = ((~b) & (~c)).sum()
        n01 = ((~b) & c).sum()
        n10 = (b & (~c)).sum()
        n11 = (b & c).sum()
        
        try:
            result = mcnemar([[n00, n01], [n10, n11]], exact=False)
            p_value = result.pvalue
        except:
            p_value = np.nan
        
        results.append({
            "miu_level": miu,
            "baseline_accuracy": float(b.mean()),
            "comparison_accuracy": float(c.mean()),
            "n_pairs": len(common),
            "p_value": p_value,
            "is_significant": p_value < 0.05 if not np.isnan(p_value) else False,
        })
    
    return pd.DataFrame(results)


def mcnemar_subject(df: pd.DataFrame) -> pd.DataFrame:
    """12: McNemar test per subject."""
    if not STATSMODELS_AVAILABLE:
        return pd.DataFrame()
    
    results = []
    high_miu = df["miu"].max()
    
    for subject in df["subject"].unique():
        sub = df[df["subject"] == subject]
        baseline = sub[sub["miu"] == 0.0].set_index("question_id")["is_correct"]
        comparison = sub[sub["miu"] == high_miu].groupby("question_id")["is_correct"].first()
        
        common = baseline.index.intersection(comparison.index)
        if len(common) < 5:
            continue
        
        b, c = baseline.loc[common].values, comparison.loc[common].values
        
        n00 = ((~b) & (~c)).sum()
        n01 = ((~b) & c).sum()
        n10 = (b & (~c)).sum()
        n11 = (b & c).sum()
        
        try:
            result = mcnemar([[n00, n01], [n10, n11]], exact=True)
            p_value = result.pvalue
        except:
            p_value = np.nan
        
        results.append({
            "subject": subject,
            "baseline_accuracy": float(b.mean()),
            "comparison_accuracy": float(c.mean()),
            "degradation_pct": (b.mean() - c.mean()) * 100,
            "n_pairs": len(common),
            "p_value": p_value,
            "is_significant": p_value < 0.05 if not np.isnan(p_value) else False,
        })
    
    return pd.DataFrame(results)


# =============================================================================
# ADVANCED ANALYSIS
# =============================================================================

def compute_delta_heatmap(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """09: Delta accuracy heatmap data."""
    mus = np.sort(df["miu"].unique())
    base_mu = mus[0]
    
    results = {}
    for model, g in df.groupby("target_model_name"):
        baseline = g[g["miu"] == base_mu].groupby("subject")["is_correct"].mean()
        pivot = g.groupby(["subject", "miu"])["is_correct"].mean().unstack(fill_value=np.nan)
        delta = pivot.subtract(baseline, axis=0)
        
        results[model] = {
            "delta_accuracy": delta.to_dict(),
            "baseline_by_subject": baseline.to_dict(),
        }
    
    first_model = list(results.keys())[0] if results else None
    delta_df = pd.DataFrame(results[first_model]["delta_accuracy"]) if first_model else pd.DataFrame()
    
    return delta_df, results


def compute_question_tiers(df: pd.DataFrame) -> Dict:
    """10: Question difficulty tiers."""
    mus = np.sort(df["miu"].unique())
    base_mu, high_mu = mus[0], mus[-1]
    
    tiers = {"easy": [], "medium": [], "hard": [], "chameleon_breakers": []}
    
    for qid, g in df.groupby("question_id"):
        base_acc = g[g["miu"] == base_mu]["is_correct"].mean() if len(g[g["miu"] == base_mu]) > 0 else np.nan
        high_acc = g[g["miu"] == high_mu]["is_correct"].mean() if len(g[g["miu"] == high_mu]) > 0 else np.nan
        
        if pd.isna(base_acc) or pd.isna(high_acc):
            continue
        
        if base_acc >= 0.8 and high_acc >= 0.7:
            tiers["easy"].append(qid)
        elif base_acc >= 0.7 and high_acc < 0.3:
            tiers["chameleon_breakers"].append(qid)
        elif base_acc < 0.5:
            tiers["hard"].append(qid)
        else:
            tiers["medium"].append(qid)
    
    total = sum(len(v) for v in tiers.values())
    return {
        "total_classified": total,
        "tier_counts": {k: len(v) for k, v in tiers.items()},
        "tier_percentages": {k: len(v) / total * 100 if total > 0 else 0 for k, v in tiers.items()},
        "chameleon_breaker_ids": tiers["chameleon_breakers"][:20],
    }


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_accuracy_by_miu(acc_df: pd.DataFrame, output_path: Path):
    """Plot accuracy curves."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set2.colors
    for i, model in enumerate(acc_df["target_model_name"].unique()):
        sub = acc_df[acc_df["target_model_name"] == model].sort_values("miu")
        x = sub["miu"].astype(float).values
        y = sub["accuracy"].astype(float).values
        ax.plot(x, y, marker="o", linewidth=2.5, markersize=8, label=model, color=colors[i % len(colors)])
        ax.fill_between(x, y - 0.02, y + 0.02, alpha=0.15, color=colors[i % len(colors)])
    
    ax.set_xlabel("Distortion Level (μ)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
    ax.set_title("Model Accuracy vs Semantic Distortion", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.05, 1.0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_accuracy_by_subject(acc_subj_df: pd.DataFrame, output_path: Path):
    """Bar chart: accuracy by subject."""
    setup_plot_style()
    
    # Get baseline (miu=0) accuracy per subject
    baseline = acc_subj_df[acc_subj_df["miu"] == 0.0].copy()
    if len(baseline) == 0:
        baseline = acc_subj_df.groupby("subject")["accuracy"].mean().reset_index()
    
    subjects = baseline["subject"].values
    accuracies = baseline["accuracy"].astype(float).values
    
    # Sort by accuracy
    sort_idx = np.argsort(accuracies)[::-1]
    subjects = subjects[sort_idx]
    accuracies = accuracies[sort_idx]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.RdYlGn(accuracies)
    bars = ax.barh(range(len(subjects)), accuracies, color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects)
    ax.set_xlabel("Accuracy (μ=0 Baseline)", fontsize=12, fontweight='bold')
    ax.set_title("Subject Performance Ranking", fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax.grid(True, axis='x', linestyle="--", alpha=0.3)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(acc + 0.02, i, f'{acc:.1%}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_subject_miu_heatmap(acc_subj_df: pd.DataFrame, output_path: Path):
    """Heatmap: subject × miu accuracy."""
    setup_plot_style()
    
    # Pivot to matrix
    pivot = acc_subj_df.pivot_table(
        index="subject", columns="miu", values="accuracy", aggfunc="mean"
    )
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(pivot.values.astype(float), cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{float(m):.1f}" for m in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Distortion Level (μ)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Subject", fontsize=12, fontweight='bold')
    ax.set_title("Accuracy Heatmap: Subject × Distortion Level", fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.0%}', ha='center', va='center', fontsize=8, color=color)
    
    plt.colorbar(im, ax=ax, shrink=0.8, label="Accuracy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_degradation_heatmap(acc_subj_df: pd.DataFrame, output_path: Path):
    """Heatmap: degradation from baseline by subject × miu."""
    setup_plot_style()
    
    # Calculate degradation from baseline
    baseline = acc_subj_df[acc_subj_df["miu"] == 0.0].set_index("subject")["accuracy"]
    pivot = acc_subj_df.pivot_table(index="subject", columns="miu", values="accuracy", aggfunc="mean")
    
    # Calculate degradation percentage
    degradation = (baseline.values.reshape(-1, 1) - pivot.values) * 100
    degradation_df = pd.DataFrame(degradation, index=pivot.index, columns=pivot.columns)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(degradation_df.values.astype(float), cmap="RdYlGn_r", aspect="auto", vmin=-5, vmax=30)
    
    ax.set_xticks(range(len(degradation_df.columns)))
    ax.set_xticklabels([f"{float(m):.1f}" for m in degradation_df.columns], rotation=45)
    ax.set_yticks(range(len(degradation_df.index)))
    ax.set_yticklabels(degradation_df.index)
    ax.set_xlabel("Distortion Level (μ)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Subject", fontsize=12, fontweight='bold')
    ax.set_title("Performance Degradation from Baseline (%)", fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(degradation_df.index)):
        for j in range(len(degradation_df.columns)):
            val = degradation_df.values[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 15 else 'black'
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize=8, color=color)
    
    plt.colorbar(im, ax=ax, shrink=0.8, label="Degradation (%)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_elasticity(acc_df: pd.DataFrame, output_path: Path):
    """Plot elasticity regression (scatter + line)."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set2.colors
    for i, model in enumerate(acc_df["target_model_name"].unique()):
        sub = acc_df[acc_df["target_model_name"] == model].sort_values("miu")
        x = sub["miu"].astype(float).values
        y = sub["accuracy"].astype(float).values
        
        ax.scatter(x, y, alpha=0.8, color=colors[i % len(colors)], s=100, edgecolors='white', linewidth=1)
        
        if len(np.unique(x)) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, intercept + slope * x_line, color=colors[i % len(colors)], 
                   linewidth=2.5, label=f"{model} (slope={slope:.4f})")
    
    ax.set_xlabel("Distortion Level (μ)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
    ax.set_title("Elasticity: Accuracy Degradation Rate", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_model_comparison(comp_df: pd.DataFrame, output_path: Path):
    """Scatter plot: CRI vs overall accuracy."""
    if comp_df.empty:
        return
    
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 7))
    
    x = comp_df["CRI"].astype(float).values
    y = comp_df["overall_accuracy"].astype(float).values
    sizes = (comp_df["n_samples"].astype(float) / comp_df["n_samples"].max() * 300 + 100).values
    
    scatter = ax.scatter(x, y, s=sizes, alpha=0.7, c=range(len(comp_df)), cmap="viridis", 
                        edgecolors='white', linewidth=2)
    
    for xi, yi, label in zip(x, y, comp_df["target_model_name"]):
        ax.annotate(label, (xi, yi), xytext=(8, 8), textcoords="offset points", 
                   fontsize=11, fontweight='bold')
    
    ax.set_xlabel("Chameleon Robustness Index (CRI)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Overall Accuracy", fontsize=12, fontweight='bold')
    ax.set_title("Model Comparison: Robustness vs Accuracy", fontsize=14, fontweight='bold')
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_mcnemar_significance(mcnemar_df: pd.DataFrame, output_path: Path):
    """Bar chart with significance markers for McNemar results."""
    if mcnemar_df.empty:
        return
    
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(mcnemar_df))
    baseline = mcnemar_df["baseline_accuracy"].astype(float).values
    comparison = mcnemar_df["comparison_accuracy"].astype(float).values
    
    width = 0.35
    bars1 = ax.bar([i - width/2 for i in x], baseline, width, label='Baseline (μ=0)', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], comparison, width, label='High μ', color='#e74c3c', alpha=0.8)
    
    # Add significance markers
    for i, (is_sig, p_val) in enumerate(zip(mcnemar_df["is_significant"], mcnemar_df["p_value"])):
        if is_sig:
            ax.annotate('*', (i, max(baseline[i], comparison[i]) + 0.03), ha='center', fontsize=16, fontweight='bold')
    
    if "miu_level" in mcnemar_df.columns:
        labels = [f"μ={m:.1f}" for m in mcnemar_df["miu_level"]]
    else:
        labels = mcnemar_df["subject"].values if "subject" in mcnemar_df.columns else [str(i) for i in x]
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
    ax.set_title("McNemar Test: Baseline vs Distorted (* = p<0.05)", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, axis='y', linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_key_insights(df: pd.DataFrame, tiers: Dict, output_path: Path):
    """4-panel key insights summary."""
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Overall accuracy by miu
    ax1 = axes[0, 0]
    acc_by_miu = df.groupby("miu")["is_correct"].mean()
    x = acc_by_miu.index.astype(float)
    y = acc_by_miu.values.astype(float)
    ax1.plot(x, y, marker='o', linewidth=2.5, markersize=8, color='#3498db')
    ax1.fill_between(x, y, alpha=0.3, color='#3498db')
    ax1.set_xlabel("μ")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Overall Accuracy by Distortion Level", fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # Panel 2: Subject performance bars
    ax2 = axes[0, 1]
    subj_acc = df[df["miu"] == 0.0].groupby("subject")["is_correct"].mean().sort_values(ascending=True)
    colors = plt.cm.RdYlGn(subj_acc.values)
    ax2.barh(range(len(subj_acc)), subj_acc.values, color=colors)
    ax2.set_yticks(range(len(subj_acc)))
    ax2.set_yticklabels(subj_acc.index)
    ax2.set_xlabel("Baseline Accuracy")
    ax2.set_title("Subject Performance (μ=0)", fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.grid(True, axis='x', linestyle="--", alpha=0.3)
    
    # Panel 3: Question tier distribution (pie)
    ax3 = axes[1, 0]
    tier_counts = tiers["tier_counts"]
    labels = ['Easy', 'Medium', 'Hard', 'Chameleon\nBreakers']
    sizes = [tier_counts.get(k, 0) for k in ['easy', 'medium', 'hard', 'chameleon_breakers']]
    colors_pie = ['#2ecc71', '#f1c40f', '#e74c3c', '#9b59b6']
    explode = (0, 0, 0, 0.1)  # Explode Chameleon Breakers
    
    if sum(sizes) > 0:
        ax3.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
               shadow=True, startangle=90)
    ax3.set_title("Question Difficulty Distribution", fontweight='bold')
    
    # Panel 4: Key stats text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    overall_acc = df["is_correct"].mean()
    baseline_acc = df[df["miu"] == 0.0]["is_correct"].mean()
    high_miu_acc = df[df["miu"] == df["miu"].max()]["is_correct"].mean()
    degradation = (baseline_acc - high_miu_acc) * 100
    
    stats_text = f"""
    KEY STATISTICS
    ══════════════════════════════════
    
    Overall Accuracy:     {overall_acc:.1%}
    Baseline (μ=0):       {baseline_acc:.1%}
    High μ (μ={df["miu"].max():.1f}):       {high_miu_acc:.1%}
    Total Degradation:    {degradation:.1f}%
    
    Questions:            {df["question_id"].nunique():,}
    Total Samples:        {len(df):,}
    Subjects:             {df["subject"].nunique()}
    
    Chameleon Breakers:   {tier_counts.get('chameleon_breakers', 0)}
    (High baseline, catastrophic at high μ)
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_delta_heatmap(df: pd.DataFrame, output_path: Path):
    """Plot delta accuracy heatmap (change from baseline)."""
    setup_plot_style()
    models = df["target_model_name"].unique()
    
    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 8), squeeze=False)
    
    for idx, model in enumerate(models):
        ax = axes[0, idx]
        g = df[df["target_model_name"] == model].copy()
        g["miu"] = pd.to_numeric(g["miu"], errors="coerce")
        
        mus = np.sort(g["miu"].dropna().unique())
        if len(mus) == 0:
            continue
        
        base_mu = mus[0]
        baseline = g[g["miu"] == base_mu].groupby("subject")["is_correct"].mean()
        pivot = g.groupby(["subject", "miu"])["is_correct"].mean().unstack(fill_value=np.nan)
        delta = pivot.subtract(baseline, axis=0)
        
        # Use dynamic scale based on actual data range
        data_min = np.nanmin(delta.values)
        data_max = np.nanmax(delta.values)
        # Ensure symmetric scale around 0, with reasonable bounds
        abs_max = max(abs(data_min), abs(data_max), 0.05)  # minimum 5% range
        im = ax.imshow(delta.values.astype(float), cmap="RdYlGn", aspect="auto", 
                      vmin=-abs_max * 1.2, vmax=abs_max * 0.3)  # Bias toward showing degradation (negative values)
        
        ax.set_xticks(range(len(delta.columns)))
        ax.set_xticklabels([f"{float(m):.1f}" for m in delta.columns], rotation=45)
        ax.set_yticks(range(len(delta.index)))
        ax.set_yticklabels(delta.index)
        ax.set_xlabel("μ (distortion level)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Subject", fontsize=11, fontweight='bold')
        ax.set_title(f"{model}\nΔ-Accuracy from Baseline", fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(len(delta.index)):
            for j in range(len(delta.columns)):
                val = delta.values[i, j]
                if not np.isnan(val):
                    color = 'white' if val < -0.2 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)
        
        plt.colorbar(im, ax=ax, shrink=0.6, label="Δ Accuracy")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


# =============================================================================
# EXECUTIVE REPORT
# =============================================================================

def generate_executive_report(df: pd.DataFrame, comp_df: pd.DataFrame, 
                              tiers: Dict, output_path: Path):
    """Generate comprehensive markdown executive report."""
    models = [str(m) for m in df["target_model_name"].unique()]
    n_questions = df["question_id"].nunique()
    n_subjects = df["subject"].nunique()
    subjects = sorted(df["subject"].unique())
    mus = sorted([float(m) for m in df["miu"].unique()])
    overall_acc = float(df["is_correct"].mean())
    baseline_acc = float(df[df["miu"] == 0.0]["is_correct"].mean())
    high_miu_acc = float(df[df["miu"] == max(mus)]["is_correct"].mean())
    
    top_model = str(comp_df.iloc[0]["target_model_name"]) if len(comp_df) > 0 else "N/A"
    top_cri = float(comp_df.iloc[0]["CRI"]) if len(comp_df) > 0 else 0
    
    # Per-subject stats
    subject_stats = df.groupby("subject").agg({
        "is_correct": ["mean", "count"],
        "question_id": "nunique"
    }).round(4)
    subject_stats.columns = ["accuracy", "samples", "questions"]
    subject_stats = subject_stats.sort_values("accuracy", ascending=False)
    
    # Baseline vs high-miu per subject
    baseline_by_subj = df[df["miu"] == 0.0].groupby("subject")["is_correct"].mean()
    high_miu_by_subj = df[df["miu"] == max(mus)].groupby("subject")["is_correct"].mean()
    degradation_by_subj = ((baseline_by_subj - high_miu_by_subj) / baseline_by_subj * 100).round(1)
    
    # CRI per subject
    cri_by_subject = compute_cri(df, ["subject"])
    
    # Accuracy by miu
    acc_by_miu = df.groupby("miu")["is_correct"].mean()
    
    md = f"""# 🦎 Chameleon Analysis Report

**Comprehensive LLM Robustness Evaluation**

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

---

## 📊 Executive Summary

This report analyzes the robustness of **{', '.join(models)}** under semantic distortions (paraphrasing) at 10 intensity levels (μ=0.0 to μ=0.9).

### Key Metrics at a Glance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Overall Accuracy** | {overall_acc:.1%} | Across all distortion levels |
| **Baseline Accuracy (μ=0)** | {baseline_acc:.1%} | Original questions |
| **High-μ Accuracy (μ=0.9)** | {high_miu_acc:.1%} | Fully paraphrased |
| **Total Degradation** | {(baseline_acc - high_miu_acc) * 100:.1f}% | Drop from baseline to high-μ |
| **Chameleon Robustness Index** | {top_cri:.3f} | Weighted accuracy (0-1, higher=better) |

---

## 📈 Dataset Overview

| Metric | Value |
|--------|-------|
| **Model(s) Evaluated** | {', '.join(models)} |
| **Unique Questions** | {n_questions:,} |
| **Subjects/Categories** | {n_subjects} ({', '.join(subjects)}) |
| **Distortion Levels** | {len(mus)} (μ = {', '.join([f'{m:.1f}' for m in mus])}) |
| **Total Samples** | {len(df):,} |
| **Distortions per Question** | {len(df) // n_questions // len(mus)} per μ level |

---

## 📉 Accuracy Degradation Analysis

### Accuracy by Distortion Level (μ)

| μ Level | Accuracy | Change from Baseline | Interpretation |
|---------|----------|---------------------|----------------|
"""
    
    for miu in mus:
        acc = acc_by_miu.get(miu, 0)
        change = (acc - baseline_acc) * 100
        change_str = f"{change:+.1f}%" if miu > 0 else "—"
        
        if miu == 0:
            interp = "Baseline (original questions)"
        elif abs(change) < 1:
            interp = "✅ Minimal impact"
        elif abs(change) < 3:
            interp = "🟡 Slight degradation"
        elif abs(change) < 5:
            interp = "🟠 Moderate degradation"
        else:
            interp = "🔴 Significant degradation"
        
        md += f"| {miu:.1f} | {acc:.1%} | {change_str} | {interp} |\n"
    
    md += f"""
### Key Observations

- **Degradation Pattern**: The model shows {'minimal' if abs(baseline_acc - high_miu_acc) < 0.03 else 'moderate' if abs(baseline_acc - high_miu_acc) < 0.06 else 'significant'} degradation under semantic distortion
- **Most Impactful μ Range**: Performance drops most noticeably at {'low μ (0.1-0.3)' if acc_by_miu.get(0.2, 1) < acc_by_miu.get(0.7, 0) else 'high μ (0.7-0.9)'}
- **Robustness Assessment**: {'Highly robust' if top_cri > 0.7 else 'Moderately robust' if top_cri > 0.5 else 'Fragile'} to paraphrasing attacks

---

## 📚 Subject-Level Analysis

### Performance by Subject (Overall)

| Subject | Accuracy | Samples | Questions | Rank |
|---------|----------|---------|-----------|------|
"""
    
    for rank, (subj, row) in enumerate(subject_stats.iterrows(), 1):
        md += f"| {subj} | {row['accuracy']:.1%} | {int(row['samples']):,} | {int(row['questions'])} | #{rank} |\n"
    
    md += f"""
### Subject Robustness (Baseline vs High-μ)

| Subject | Baseline (μ=0) | High-μ (μ=0.9) | Degradation | CRI | Vulnerability |
|---------|----------------|----------------|-------------|-----|---------------|
"""
    
    for subj in subjects:
        base = baseline_by_subj.get(subj, 0)
        high = high_miu_by_subj.get(subj, 0)
        deg = degradation_by_subj.get(subj, 0)
        cri_row = cri_by_subject[cri_by_subject["subject"] == subj]
        cri_val = float(cri_row["CRI"].iloc[0]) if len(cri_row) > 0 else 0
        
        if deg < 2:
            vuln = "🟢 Low"
        elif deg < 5:
            vuln = "🟡 Medium"
        elif deg < 10:
            vuln = "🟠 High"
        else:
            vuln = "🔴 Critical"
        
        md += f"| {subj} | {base:.1%} | {high:.1%} | {deg:.1f}% | {cri_val:.3f} | {vuln} |\n"
    
    # Find worst and best subjects
    worst_subj = degradation_by_subj.idxmax() if len(degradation_by_subj) > 0 else "N/A"
    best_subj = degradation_by_subj.idxmin() if len(degradation_by_subj) > 0 else "N/A"
    
    md += f"""
### Subject Insights

- **Most Vulnerable Subject**: **{worst_subj}** ({degradation_by_subj.get(worst_subj, 0):.1f}% degradation)
- **Most Robust Subject**: **{best_subj}** ({degradation_by_subj.get(best_subj, 0):.1f}% degradation)
- **Recommendation**: Focus quality improvements on {worst_subj} questions

---

## 🎯 Question Difficulty Tiers

Questions are classified based on how models perform at baseline vs high distortion:

| Tier | Definition | Count | Percentage | Interpretation |
|------|------------|-------|------------|----------------|
| 🟢 **Easy** | ≥80% at μ=0, ≥70% at μ=0.9 | {tiers['tier_counts'].get('easy', 0)} | {tiers['tier_percentages'].get('easy', 0):.1f}% | True understanding |
| 🟡 **Medium** | Good at low μ, some struggle at high | {tiers['tier_counts'].get('medium', 0)} | {tiers['tier_percentages'].get('medium', 0):.1f}% | Partial understanding |
| 🔴 **Hard** | <50% even at μ=0 | {tiers['tier_counts'].get('hard', 0)} | {tiers['tier_percentages'].get('hard', 0):.1f}% | Knowledge gap |
| 💀 **Chameleon Breakers** | ≥70% at μ=0, <30% at μ=0.9 | {tiers['tier_counts'].get('chameleon_breakers', 0)} | {tiers['tier_percentages'].get('chameleon_breakers', 0):.1f}% | **Surface pattern matching** |

### What are Chameleon Breakers?

**Chameleon Breakers** are the most critical finding. These are questions where:
- The model answers correctly when the question is in its "expected" form (μ=0)
- The model fails catastrophically when the same question is paraphrased (high μ)

This indicates the model is **matching lexical patterns** rather than truly understanding the question. These {tiers['tier_counts'].get('chameleon_breakers', 0)} questions reveal fundamental limitations in the model's reasoning.

---

## 📊 Statistical Significance

### McNemar's Test Results

McNemar's test compares paired correct/incorrect responses between baseline (μ=0) and distorted questions.

"""
    
    # Try to load McNemar results
    mcnemar_path = output_path.parent / "11_mcnemar_distortion.csv"
    if mcnemar_path.exists():
        mcnemar_df = pd.read_csv(mcnemar_path)
        md += "| μ Level | Baseline Acc | Distorted Acc | Change | p-value | Significant? |\n"
        md += "|---------|--------------|---------------|--------|---------|-------------|\n"
        
        for _, row in mcnemar_df.iterrows():
            sig = "✅ Yes (p<0.05)" if row.get("is_significant", False) else "❌ No"
            change = (row["comparison_accuracy"] - row["baseline_accuracy"]) * 100
            md += f"| μ={row['miu_level']:.1f} | {row['baseline_accuracy']:.1%} | {row['comparison_accuracy']:.1%} | {change:+.1f}% | {row['p_value']:.4f} | {sig} |\n"
        
        sig_count = mcnemar_df["is_significant"].sum()
        md += f"\n**Conclusion**: {sig_count} of {len(mcnemar_df)} distortion levels show statistically significant degradation.\n"
    else:
        md += "*McNemar test results not available.*\n"
    
    md += """
---

## 🔬 Error Analysis

### Error Taxonomy

"""
    
    # Try to load error taxonomy
    error_path = output_path.parent / "06_error_taxonomy.json"
    if error_path.exists():
        with open(error_path, "r") as f:
            taxonomy = json.load(f)
        
        md += "| Error Type | Count | Percentage | Description |\n"
        md += "|------------|-------|------------|-------------|\n"
        
        error_desc = {
            "correct": "Answered correctly",
            "wrong_choice": "Valid option, but incorrect",
            "blank_answer": "No answer provided",
            "multiple_options": "Multiple options guessed",
            "invalid_format": "Unrecognizable response format"
        }
        
        for etype, count in taxonomy.get("global_counts", {}).items():
            freq = taxonomy.get("global_frequencies", {}).get(etype, 0)
            desc = error_desc.get(etype, "Unknown")
            md += f"| {etype} | {count:,} | {freq:.1%} | {desc} |\n"
    else:
        md += "*Error taxonomy not available.*\n"
    
    md += """
---

## 🔑 Key Metrics Explained

### Chameleon Robustness Index (CRI)

The CRI is a **weighted accuracy metric** that emphasizes performance at high distortion levels:

```
CRI = Σ(accuracy(μ) × w(μ))
where w(μ) = exp(2.0 × μ) / Σ exp(2.0 × μ)
```

This means:
- Accuracy at μ=0.9 is weighted **~6x more** than accuracy at μ=0.0
- A model that maintains accuracy under paraphrasing scores higher
- CRI > 0.7 = Highly robust | CRI 0.5-0.7 = Moderate | CRI < 0.5 = Fragile

### Elasticity Slope

Linear regression of accuracy vs μ level:

```
accuracy = intercept + slope × μ
```

- **Slope ≈ 0**: Model is robust (stable across distortions)
- **Slope < -0.05**: Model is fragile (loses >5% accuracy per 0.1 μ increase)

### Bootstrap Confidence Intervals

95% confidence intervals computed via 500 bootstrap resamples. If confidence intervals don't overlap, differences are likely significant.

---

## 📋 Files Generated

| File | Description |
|------|-------------|
| `01_accuracy_by_miu.csv/png` | Accuracy curve across μ levels |
| `02_*.png` | Subject ranking, heatmaps, degradation |
| `03_chameleon_robustness_index.csv` | CRI scores (global + per-subject) |
| `04_elasticity.csv/png` | Degradation slope analysis |
| `05_model_comparison.csv/png` | Model comparison scatter |
| `06_error_taxonomy.json` | Error classification |
| `07_confusion_clusters.json` | TF-IDF clustering of failures |
| `08_bootstrap_intervals.csv` | 95% confidence intervals |
| `09_delta_accuracy_heatmap.csv/png` | Δ-accuracy from baseline |
| `10_question_difficulty_tiers.json` | Easy/Medium/Hard/Breakers |
| `11_mcnemar_distortion.csv/png` | McNemar test by μ level |
| `12_mcnemar_subject.csv/png` | McNemar test by subject |
| `13_key_insights.png` | 4-panel visual summary |

---

## 💡 Recommendations

"""
    
    # Generate recommendations based on findings
    recommendations = []
    
    if top_cri < 0.6:
        recommendations.append("⚠️ **Low robustness detected**: Consider fine-tuning with paraphrased examples")
    
    if tiers['tier_counts'].get('chameleon_breakers', 0) > 10:
        recommendations.append(f"⚠️ **{tiers['tier_counts'].get('chameleon_breakers', 0)} Chameleon Breakers found**: Review these questions for pattern-matching vulnerabilities")
    
    if degradation_by_subj.max() > 10:
        recommendations.append(f"⚠️ **{worst_subj} shows {degradation_by_subj.max():.1f}% degradation**: Focus training data augmentation on this subject")
    
    if baseline_acc < 0.7:
        recommendations.append("⚠️ **Baseline accuracy below 70%**: Model may have fundamental knowledge gaps")
    
    if len(recommendations) == 0:
        recommendations.append("✅ Model shows good robustness to semantic distortions")
    
    for rec in recommendations:
        md += f"- {rec}\n"
    
    md += f"""
---

## 📖 Methodology

1. **Distortion Generation**: Questions paraphrased using Mistral AI at 10 intensity levels (μ=0.0 to μ=0.9)
2. **Evaluation**: Target model ({', '.join(models)}) answers all distorted questions
3. **Analysis**: Statistical comparison of baseline vs distorted performance
4. **Metrics**: CRI, elasticity, McNemar's test, bootstrap CIs, error taxonomy

---

*Generated by Chameleon Analysis Engine*  
*For more details, see the accompanying CSV and JSON files.*
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_analysis(
    project_name: str,
    projects_dir: str = "Projects",
    n_bootstrap: int = 500
) -> Dict[str, Any]:
    """
    Run complete unified analysis pipeline.
    
    All outputs go to: Projects/{project_name}/results/analysis/
    """
    print("\n" + "=" * 60)
    print("🦎 CHAMELEON ANALYSIS")
    print("=" * 60)
    
    project_path = Path(projects_dir) / project_name
    results_dir = project_path / "results"
    analysis_dir = results_dir / "analysis"
    
    # Find results file
    csv_path = results_dir / "results.csv"
    if not csv_path.exists():
        csv_path = project_path / "distorted_data" / "distortions_complete.csv"
    
    if not csv_path.exists():
        print(f"❌ Results file not found")
        return {"status": "error", "message": "Results file not found"}
    
    print(f"📁 Project: {project_name}")
    print(f"📄 Input: {csv_path}")
    print(f"📂 Output: {analysis_dir}")
    
    # Clean output directory
    if analysis_dir.exists():
        import shutil
        shutil.rmtree(analysis_dir)
    ensure_dir(analysis_dir)
    
    # Load data
    print("\n[1/14] Loading data...")
    df = load_results(csv_path)
    print(f"   ✅ {len(df):,} rows, {df['target_model_name'].nunique()} models")
    
    files = []
    
    # Core metrics
    print("[2/14] Computing accuracy by μ...")
    acc_df = compute_accuracy_by_miu(df)
    acc_df.to_csv(analysis_dir / "01_accuracy_by_miu.csv", index=False)
    plot_accuracy_by_miu(acc_df, analysis_dir / "01_accuracy_by_miu.png")
    files.extend(["01_accuracy_by_miu.csv", "01_accuracy_by_miu.png"])
    
    print("[3/14] Computing accuracy by subject...")
    acc_subj = compute_accuracy_by_subject_miu(df)
    acc_subj.to_csv(analysis_dir / "02_accuracy_by_subject_miu.csv", index=False)
    plot_accuracy_by_subject(acc_subj, analysis_dir / "02_subject_ranking.png")
    plot_subject_miu_heatmap(acc_subj, analysis_dir / "02_subject_miu_heatmap.png")
    plot_degradation_heatmap(acc_subj, analysis_dir / "02_degradation_heatmap.png")
    files.extend(["02_accuracy_by_subject_miu.csv", "02_subject_ranking.png", 
                  "02_subject_miu_heatmap.png", "02_degradation_heatmap.png"])
    
    print("[4/14] Computing CRI...")
    cri_global = compute_cri(df, ["target_model_name"])
    cri_subject = compute_cri(df, ["target_model_name", "subject"])
    cri_df = pd.concat([cri_global.assign(scope="global"), cri_subject.assign(scope="per_subject")])
    cri_df.to_csv(analysis_dir / "03_chameleon_robustness_index.csv", index=False)
    files.append("03_chameleon_robustness_index.csv")
    
    print("[5/14] Computing elasticity...")
    elast_df = compute_elasticity(acc_df)
    elast_df.to_csv(analysis_dir / "04_elasticity.csv", index=False)
    plot_elasticity(acc_df, analysis_dir / "04_elasticity.png")
    files.extend(["04_elasticity.csv", "04_elasticity.png"])
    
    print("[6/14] Building model comparison...")
    comp_df = build_model_comparison(df, acc_df, cri_global, elast_df)
    comp_df.to_csv(analysis_dir / "05_model_comparison.csv", index=False)
    plot_model_comparison(comp_df, analysis_dir / "05_model_comparison.png")
    files.extend(["05_model_comparison.csv", "05_model_comparison.png"])
    
    # Error analysis
    print("[7/14] Building error taxonomy...")
    taxonomy = build_error_taxonomy(df)
    with open(analysis_dir / "06_error_taxonomy.json", "w") as f:
        json.dump(taxonomy, f, indent=2)
    files.append("06_error_taxonomy.json")
    
    print("[8/14] Building confusion clusters...")
    clusters = build_confusion_clusters(df)
    with open(analysis_dir / "07_confusion_clusters.json", "w") as f:
        json.dump(clusters, f, indent=2)
    files.append("07_confusion_clusters.json")
    
    # Statistical
    print(f"[9/14] Computing bootstrap CIs ({n_bootstrap} samples)...")
    ci_df = bootstrap_ci(df, ["target_model_name", "miu"], n_bootstrap)
    ci_df.to_csv(analysis_dir / "08_bootstrap_intervals.csv", index=False)
    files.append("08_bootstrap_intervals.csv")
    
    # Advanced
    print("[10/14] Computing delta heatmap...")
    delta_df, _ = compute_delta_heatmap(df)
    delta_df.to_csv(analysis_dir / "09_delta_accuracy_heatmap.csv")
    plot_delta_heatmap(df, analysis_dir / "09_delta_accuracy_heatmap.png")
    files.extend(["09_delta_accuracy_heatmap.csv", "09_delta_accuracy_heatmap.png"])
    
    print("[11/14] Computing question tiers...")
    tiers = compute_question_tiers(df)
    with open(analysis_dir / "10_question_difficulty_tiers.json", "w") as f:
        json.dump(tiers, f, indent=2)
    files.append("10_question_difficulty_tiers.json")
    
    # McNemar tests
    print("[12/14] Running McNemar tests...")
    mcnemar_dist = mcnemar_distortion(df)
    if len(mcnemar_dist) > 0:
        mcnemar_dist.to_csv(analysis_dir / "11_mcnemar_distortion.csv", index=False)
        plot_mcnemar_significance(mcnemar_dist, analysis_dir / "11_mcnemar_distortion.png")
        files.extend(["11_mcnemar_distortion.csv", "11_mcnemar_distortion.png"])
    
    mcnemar_subj = mcnemar_subject(df)
    if len(mcnemar_subj) > 0:
        mcnemar_subj.to_csv(analysis_dir / "12_mcnemar_subject.csv", index=False)
        plot_mcnemar_significance(mcnemar_subj, analysis_dir / "12_mcnemar_subject.png")
        files.extend(["12_mcnemar_subject.csv", "12_mcnemar_subject.png"])
    
    # Key insights summary
    print("[13/14] Creating key insights panel...")
    plot_key_insights(df, tiers, analysis_dir / "13_key_insights.png")
    files.append("13_key_insights.png")
    
    # Executive report
    print("[14/14] Generating executive report...")
    generate_executive_report(df, comp_df, tiers, analysis_dir / "EXECUTIVE_REPORT.md")
    files.append("EXECUTIVE_REPORT.md")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\n📁 Output: {analysis_dir}")
    print(f"📊 Files: {len(files)}")
    print(f"\n🚀 Start with: {analysis_dir / 'EXECUTIVE_REPORT.md'}")
    
    return {
        "status": "complete",
        "output_dir": str(analysis_dir),
        "files": files,
        "overall_accuracy": float(df["is_correct"].mean()),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chameleon Analysis")
    parser.add_argument("--project", "-p", required=True, help="Project name")
    parser.add_argument("--projects-dir", default="Projects", help="Projects directory")
    args = parser.parse_args()
    
    run_analysis(args.project, args.projects_dir)

