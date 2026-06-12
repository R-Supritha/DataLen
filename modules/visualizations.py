"""
visualizations.py
Generate and save all Matplotlib/Seaborn charts to static/charts/.
"""

from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

CHART_DIR = os.path.join(os.path.dirname(__file__), "..", "static", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

# Consistent palette
PALETTE = ["#6C63FF", "#FF6584", "#43CEA2", "#F7971E", "#2193b0", "#cc5333"]
sns.set_style("whitegrid")


def _save(fig, name: str) -> str:
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return name


# ─── 1. Missing Values Bar Chart ──────────────────────────────────────────────

def plot_missing_values(quality_report: dict) -> str:
    pcts = quality_report["missing_values"]["percentages"]
    df_miss = pd.DataFrame(
        {"column": list(pcts.keys()), "missing_pct": list(pcts.values())}
    )
    df_miss = df_miss[df_miss["missing_pct"] > 0].sort_values("missing_pct", ascending=False)

    if df_miss.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "✅ No missing values!", ha="center", va="center",
                fontsize=16, color="#2ecc71", fontweight="bold")
        ax.axis("off")
        return _save(fig, "missing_values.png")

    fig, ax = plt.subplots(figsize=(max(8, len(df_miss) * 0.7), 5))
    bars = ax.bar(df_miss["column"], df_miss["missing_pct"],
                  color=[PALETTE[i % len(PALETTE)] for i in range(len(df_miss))],
                  edgecolor="white", linewidth=0.8)
    ax.axhline(5, color="#e74c3c", linestyle="--", linewidth=1.2, label="5% threshold")
    ax.axhline(20, color="#c0392b", linestyle="--", linewidth=1.2, label="20% threshold")
    for bar, val in zip(bars, df_miss["missing_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Missing Values by Column", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Missing %", fontsize=11)
    ax.set_xlabel("Column", fontsize=11)
    plt.xticks(rotation=35, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(0, min(100, df_miss["missing_pct"].max() * 1.25))
    return _save(fig, "missing_values.png")


# ─── 2. Correlation Heatmap ───────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Not enough numeric columns for correlation.",
                ha="center", va="center", fontsize=13)
        ax.axis("off")
        return _save(fig, "correlation_heatmap.png")

    corr = num_df.corr()
    n = corr.shape[0]
    size = max(7, min(14, n))
    fig, ax = plt.subplots(figsize=(size, size * 0.8))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
        square=True, linewidths=0.5, annot=(n <= 12), fmt=".2f",
        annot_kws={"size": 8}, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    return _save(fig, "correlation_heatmap.png")


# ─── 3. Class Distribution Chart ─────────────────────────────────────────────

def plot_class_distribution(quality_report: dict) -> str:
    imbalance = quality_report.get("class_imbalance", {})
    if not imbalance:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No categorical target columns detected.",
                ha="center", va="center", fontsize=13)
        ax.axis("off")
        return _save(fig, "class_distribution.png")

    n_plots = len(imbalance)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5))
    if n_plots == 1:
        axes = [axes]

    for ax, (col, info) in zip(axes, imbalance.items()):
        dist = info["distribution"]
        labels = [str(k) for k in dist.keys()]
        values = list(dist.values())
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
        bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(f"Distribution: {col}\n(ratio {info['imbalance_ratio']}:1)",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Count")
        plt.sca(ax)
        plt.xticks(rotation=20, ha="right", fontsize=8)
        if info["is_imbalanced"]:
            ax.set_facecolor("#fff5f5")

    fig.suptitle("Class Distributions", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "class_distribution.png")


# ─── 4. Outlier Boxplots ──────────────────────────────────────────────────────

def plot_outlier_boxplots(df: pd.DataFrame, max_cols: int = 10) -> str:
    num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if num_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No numeric columns for boxplot.",
                ha="center", va="center", fontsize=13)
        ax.axis("off")
        return _save(fig, "outlier_boxplots.png")

    cols = num_df.columns.tolist()[:max_cols]
    n = len(cols)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(cols):
        data = num_df[col].dropna()
        bp = axes[i].boxplot(data, patch_artist=True, notch=False,
                             boxprops=dict(facecolor=PALETTE[i % len(PALETTE)], alpha=0.7),
                             medianprops=dict(color="white", linewidth=2),
                             flierprops=dict(marker="o", markersize=4,
                                             markerfacecolor="#e74c3c", alpha=0.6))
        axes[i].set_title(col, fontsize=10, fontweight="bold")
        axes[i].set_xticks([])

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Outlier Boxplots (Numeric Features)", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "outlier_boxplots.png")


# ─── 5. Health Score Summary Chart ───────────────────────────────────────────

def plot_health_summary(health_report: dict) -> str:
    breakdown = health_report.get("breakdown", {})
    if not breakdown:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Health score not computed.", ha="center", va="center")
        ax.axis("off")
        return _save(fig, "health_summary.png")

    dims = list(breakdown.keys())
    scores = [breakdown[d]["score"] for d in dims]
    maxes = [breakdown[d]["max"] for d in dims]
    labels = [d.replace("_", "\n").title() for d in dims]
    pcts = [s / m * 100 for s, m in zip(scores, maxes)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Radar / horizontal bar ──
    colors = ["#2ecc71" if p >= 80 else "#f39c12" if p >= 60 else "#e74c3c" for p in pcts]
    bars = ax1.barh(labels, scores, color=colors, edgecolor="white", height=0.55)
    for bar, s, m in zip(bars, scores, maxes):
        ax1.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                 f"{s:.1f}/{m}", va="center", fontsize=9, fontweight="bold")
    ax1.set_xlim(0, max(maxes) * 1.25)
    ax1.set_title("Score by Dimension", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Points Earned")

    # Legend patches
    g = mpatches.Patch(color="#2ecc71", label="≥ 80% (Good)")
    a = mpatches.Patch(color="#f39c12", label="60–79% (Fair)")
    r = mpatches.Patch(color="#e74c3c", label="< 60% (Poor)")
    ax1.legend(handles=[g, a, r], fontsize=8, loc="lower right")

    # ── Donut gauge ──
    total = health_report.get("score", 0)
    color = health_report.get("color", "#6C63FF")
    donut_vals = [total, 100 - total]
    donut_colors = [color, "#eaecef"]
    wedges, _ = ax2.pie(donut_vals, colors=donut_colors, startangle=90,
                        counterclock=False, wedgeprops=dict(width=0.4))
    ax2.text(0, 0, f"{total:.0f}", ha="center", va="center",
             fontsize=36, fontweight="bold", color=color)
    ax2.text(0, -0.25, f"Grade {health_report.get('grade', '?')}",
             ha="center", va="center", fontsize=14, color="#555")
    ax2.set_title("Overall Health Score", fontsize=13, fontweight="bold")

    fig.suptitle("Dataset Health Summary", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "health_summary.png")


# ─── 6. Autoencoder Anomaly Score Distribution ───────────────────────────────

def plot_autoencoder_scores(ae_result: dict) -> str:
    scores = ae_result.get("anomaly_scores", [])
    threshold_pct = ae_result.get("threshold_percentile", 95)

    fig, ax = plt.subplots(figsize=(9, 4))
    if not scores:
        ax.text(0.5, 0.5, "No autoencoder scores available.",
                ha="center", va="center", fontsize=13)
        ax.axis("off")
        return _save(fig, "autoencoder_scores.png")

    scores_arr = np.array(scores)
    threshold_val = np.percentile(scores_arr, threshold_pct)

    normal = scores_arr[scores_arr <= threshold_val]
    anomaly = scores_arr[scores_arr > threshold_val]

    ax.hist(normal, bins=40, color="#6C63FF", alpha=0.75, label="Normal")
    ax.hist(anomaly, bins=20, color="#FF6584", alpha=0.85, label="Anomaly")
    ax.axvline(threshold_val, color="#c0392b", linestyle="--", linewidth=1.5,
               label=f"Threshold (p{int(threshold_pct)})")
    ax.set_title("Autoencoder Anomaly Score Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Normalised Anomaly Score")
    ax.set_ylabel("Frequency")
    ax.legend()
    return _save(fig, "autoencoder_scores.png")


# ─── Master Generate Function ─────────────────────────────────────────────────

def generate_all_charts(
    df: pd.DataFrame,
    quality_report: dict,
    health_report: dict,
    ae_result: dict | None = None,
) -> dict[str, str]:
    """Generate all charts and return a mapping of {chart_key: filename}."""
    charts = {}
    charts["missing_values"] = plot_missing_values(quality_report)
    charts["correlation"] = plot_correlation_heatmap(df)
    charts["class_distribution"] = plot_class_distribution(quality_report)
    charts["boxplots"] = plot_outlier_boxplots(df)
    charts["health_summary"] = plot_health_summary(health_report)
    if ae_result:
        charts["autoencoder"] = plot_autoencoder_scores(ae_result)
    return charts