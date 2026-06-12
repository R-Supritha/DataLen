"""
health_score.py
Compute a 0–100 Dataset Health Score from all analysis results.
"""

from __future__ import annotations


def compute_health_score(
    quality_report: dict,
    outlier_report: dict,
    autoencoder_report: dict | None = None,
    semantic_report: dict | None = None,
) -> dict:
    """
    Aggregate penalties across quality dimensions into a 0–100 health score.
    Higher = healthier dataset.
    """
    breakdown = {}
    penalties = {}

    # ── 1. Missing Values (max penalty: 25 pts) ──
    mv = quality_report.get("missing_values", {})
    overall_missing_pct = mv.get("overall_pct", 0.0)
    missing_penalty = min(25.0, overall_missing_pct * 1.5)
    breakdown["missing_values"] = {
        "score": round(25 - missing_penalty, 2),
        "max": 25,
        "pct_missing": overall_missing_pct,
        "explanation": (
            f"{overall_missing_pct:.1f}% of all cells are missing. "
            + _missing_advice(overall_missing_pct)
        ),
    }
    penalties["missing_values"] = missing_penalty

    # ── 2. Duplicates (max penalty: 15 pts) ──
    dup = quality_report.get("duplicates", {})
    dup_pct = dup.get("percentage", 0.0)
    dup_penalty = min(15.0, dup_pct * 2.0)
    breakdown["duplicates"] = {
        "score": round(15 - dup_penalty, 2),
        "max": 15,
        "pct_duplicates": dup_pct,
        "explanation": (
            f"{dup_pct:.1f}% of rows are exact duplicates. "
            + _dup_advice(dup_pct)
        ),
    }
    penalties["duplicates"] = dup_penalty

    # ── 3. Outliers (max penalty: 20 pts) ──
    iqr_total = outlier_report.get("iqr", {}).get("total_outliers", 0)
    total_cells = quality_report["shape"]["rows"] * quality_report["shape"]["columns"]
    outlier_pct = (iqr_total / max(1, quality_report["shape"]["rows"])) * 100
    outlier_penalty = min(20.0, outlier_pct * 1.2)
    breakdown["outliers"] = {
        "score": round(20 - outlier_penalty, 2),
        "max": 20,
        "iqr_outlier_count": iqr_total,
        "outlier_pct": round(outlier_pct, 2),
        "explanation": (
            f"IQR method found {iqr_total} outlier values (~{outlier_pct:.1f}% of rows). "
            + _outlier_advice(outlier_pct)
        ),
    }
    penalties["outliers"] = outlier_penalty

    # ── 4. Class Imbalance (max penalty: 15 pts) ──
    imbalance = quality_report.get("class_imbalance", {})
    worst_ratio = max(
        (v.get("imbalance_ratio", 1) for v in imbalance.values()), default=1
    )
    imbalance_penalty = min(15.0, max(0.0, (worst_ratio - 1) * 1.0))
    breakdown["class_imbalance"] = {
        "score": round(15 - imbalance_penalty, 2),
        "max": 15,
        "worst_ratio": worst_ratio,
        "explanation": (
            f"Worst class imbalance ratio: {worst_ratio:.1f}:1. "
            + _imbalance_advice(worst_ratio)
        ),
    }
    penalties["class_imbalance"] = imbalance_penalty

    # ── 5. Label Noise / Semantic Issues (max penalty: 15 pts) ──
    noise_penalty = 0.0
    noise_explanation = "No semantic/label noise analysis performed."
    if semantic_report and "noisy_pairs" in semantic_report:
        noise_rate = semantic_report.get("noise_rate", 0.0)
        noise_penalty = min(15.0, noise_rate * 5)
        noise_explanation = (
            f"{semantic_report['total_noisy_pairs']} semantically similar pairs with different labels. "
            + _noise_advice(semantic_report["total_noisy_pairs"])
        )
    breakdown["label_noise"] = {
        "score": round(15 - noise_penalty, 2),
        "max": 15,
        "explanation": noise_explanation,
    }
    penalties["label_noise"] = noise_penalty

    # ── 6. Data Leakage (max penalty: 10 pts) ──
    leakage = quality_report.get("data_leakage", {})
    leakage_candidates = leakage.get("leakage_candidates", [])
    high_risk = sum(1 for c in leakage_candidates if c.get("risk") == "HIGH")
    med_risk = sum(1 for c in leakage_candidates if c.get("risk") == "MEDIUM")
    leakage_penalty = min(10.0, high_risk * 5 + med_risk * 2)
    breakdown["data_leakage"] = {
        "score": round(10 - leakage_penalty, 2),
        "max": 10,
        "high_risk": high_risk,
        "medium_risk": med_risk,
        "explanation": (
            f"{high_risk} high-risk, {med_risk} medium-risk leakage columns detected. "
            + _leakage_advice(high_risk, med_risk)
        ),
    }
    penalties["data_leakage"] = leakage_penalty

    # ── Final Score ──
    total_penalty = sum(penalties.values())
    total_score = max(0.0, min(100.0, 100 - total_penalty))

    return {
        "score": round(total_score, 1),
        "grade": _grade(total_score),
        "color": _color(total_score),
        "breakdown": breakdown,
        "recommendations": _build_recommendations(breakdown, leakage_candidates, quality_report),
    }


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 80: return "B"
    if score >= 70: return "C"
    if score >= 60: return "D"
    return "F"


def _color(score: float) -> str:
    if score >= 80: return "#2ecc71"   # green
    if score >= 60: return "#f39c12"   # amber
    return "#e74c3c"                   # red


def _missing_advice(pct: float) -> str:
    if pct < 1: return "Excellent – no significant missing data."
    if pct < 5: return "Consider imputation (median/mode) for affected columns."
    if pct < 20: return "Use KNN or iterative imputation; drop columns > 40% missing."
    return "High missingness – evaluate whether columns should be dropped entirely."


def _dup_advice(pct: float) -> str:
    if pct < 0.5: return "No action needed."
    if pct < 5: return "Remove duplicates with df.drop_duplicates()."
    return "Investigate source pipeline; data deduplication is critical."


def _outlier_advice(pct: float) -> str:
    if pct < 2: return "Minimal outliers; no immediate action required."
    if pct < 10: return "Review flagged rows; consider Winsorization or log transforms."
    return "High outlier rate – inspect for data entry errors or apply robust scaling."


def _imbalance_advice(ratio: float) -> str:
    if ratio < 2: return "Balanced classes – no action needed."
    if ratio < 5: return "Mild imbalance; consider stratified sampling."
    return "Severe imbalance – apply SMOTE, class weighting, or oversampling techniques."


def _noise_advice(count: int) -> str:
    if count == 0: return "Labels appear consistent."
    if count < 10: return "Review flagged pairs manually and correct labels."
    return "Significant label noise – audit labeling process and consider relabeling."


def _leakage_advice(high: int, med: int) -> str:
    if high + med == 0: return "No leakage risk detected."
    msgs = []
    if high: msgs.append(f"Drop {high} high-risk column(s) before training.")
    if med: msgs.append(f"Investigate {med} medium-risk column(s) for temporal or target-derived features.")
    return " ".join(msgs)


def _build_recommendations(breakdown: dict, leakage_candidates: list, quality_report: dict) -> list[str]:
    """Generate an ordered list of actionable recommendations."""
    recs = []

    mv_score = breakdown["missing_values"]["score"]
    if mv_score < 20:
        cols = [
            col
            for col, pct in quality_report.get("missing_values", {}).get("percentages", {}).items()
            if pct > 0
        ]
        recs.append(
            f"🔧 Impute or drop columns with missing values: {', '.join(cols[:5])}"
        )

    if breakdown["duplicates"]["score"] < 12:
        recs.append("🗑️ Remove duplicate rows using df.drop_duplicates(keep='first').")

    if breakdown["outliers"]["score"] < 15:
        recs.append(
            "📊 Handle outliers: apply IQR-based clipping or RobustScaler before model training."
        )

    if breakdown["class_imbalance"]["score"] < 10:
        recs.append(
            "⚖️ Address class imbalance with SMOTE (imbalanced-learn) or class_weight='balanced'."
        )

    if breakdown["label_noise"]["score"] < 12:
        recs.append(
            "🏷️ Review semantically similar samples with conflicting labels – possible annotation errors."
        )

    if leakage_candidates:
        names = [c["column"] for c in leakage_candidates[:3]]
        recs.append(f"🚨 Investigate potential data leakage in: {', '.join(names)}.")

    # Correlation-based advice
    high_corr = quality_report.get("correlations", {}).get("high_correlations", [])
    if high_corr:
        pair = high_corr[0]
        recs.append(
            f"📉 High correlation ({pair['correlation']}) between '{pair['col_a']}' and '{pair['col_b']}' – consider dropping one to reduce multicollinearity."
        )

    if not recs:
        recs.append("✅ Dataset looks healthy! Continue with standard preprocessing pipelines.")

    return recs