"""
quality_checker.py
Core dataset quality analysis: missing values, duplicates,
data types, class imbalance, correlation, and data leakage.
"""

import pandas as pd
import numpy as np
from scipy import stats


def analyze_missing_values(df: pd.DataFrame) -> dict:
    """Compute per-column missing value stats."""
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(2)
    return {
        "counts": missing.to_dict(),
        "percentages": pct.to_dict(),
        "total_missing": int(missing.sum()),
        "total_cells": int(total * len(df.columns)),
        "overall_pct": round(missing.sum() / (total * len(df.columns)) * 100, 2),
        "columns_with_missing": int((missing > 0).sum()),
    }


def detect_duplicates(df: pd.DataFrame) -> dict:
    """Detect exact duplicate rows."""
    n_dups = int(df.duplicated().sum())
    return {
        "count": n_dups,
        "percentage": round(n_dups / len(df) * 100, 2),
        "duplicate_indices": df[df.duplicated()].index.tolist()[:20],  # cap at 20
    }


def validate_data_types(df: pd.DataFrame) -> dict:
    """
    Infer column roles (numeric, categorical, datetime, text)
    and flag potential type mismatches.
    """
    report = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_unique = int(df[col].nunique())
        n_total = len(df)
        sample = df[col].dropna().head(3).tolist()

        issues = []
        # Numeric column stored as object?
        if df[col].dtype == object:
            try:
                pd.to_numeric(df[col].dropna())
                issues.append("Stored as object but looks numeric")
            except (ValueError, TypeError):
                pass
        # High-cardinality categorical
        if df[col].dtype == object and n_unique > 0.8 * n_total and n_total > 20:
            issues.append("High cardinality – may be a free-text or ID column")
        # Date-like strings
        if df[col].dtype == object and n_unique < n_total:
            sample_str = str(sample[0]) if sample else ""
            date_keywords = ["-", "/", "T", "Z"]
            if any(k in sample_str for k in date_keywords) and len(sample_str) >= 8:
                issues.append("May contain datetime values stored as strings")

        report[col] = {
            "dtype": dtype,
            "unique_values": n_unique,
            "sample": [str(s) for s in sample],
            "issues": issues,
        }
    return report


def detect_class_imbalance(df: pd.DataFrame, target_col: str | None = None) -> dict:
    """
    For each low-cardinality column (possible target), compute
    class distribution and imbalance ratio.
    """
    results = {}
    candidates = []
    if target_col and target_col in df.columns:
        candidates = [target_col]
    else:
        # Auto-detect: object or low-cardinality int columns
        for col in df.columns:
            if df[col].dtype == object or (
                df[col].nunique() <= 20 and df[col].dtype in ["int64", "int32"]
            ):
                candidates.append(col)

    for col in candidates[:5]:  # limit to 5
        vc = df[col].value_counts()
        if len(vc) < 2:
            continue
        imbalance_ratio = round(vc.iloc[0] / vc.iloc[-1], 2)
        is_imbalanced = imbalance_ratio > 5
        results[col] = {
            "distribution": vc.to_dict(),
            "imbalance_ratio": imbalance_ratio,
            "is_imbalanced": is_imbalanced,
            "majority_class": str(vc.index[0]),
            "minority_class": str(vc.index[-1]),
        }
    return results


def compute_correlations(df: pd.DataFrame) -> dict:
    """Pearson correlation matrix for numeric columns; flag high correlations."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return {"matrix": {}, "high_correlations": []}

    corr = num_df.corr()
    high = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if abs(val) >= 0.85:
                high.append(
                    {
                        "col_a": cols[i],
                        "col_b": cols[j],
                        "correlation": round(float(val), 3),
                    }
                )
    # Convert to JSON-safe dict
    corr_dict = {
        col: {c: round(float(v), 3) for c, v in row.items()}
        for col, row in corr.to_dict().items()
    }
    return {"matrix": corr_dict, "high_correlations": high}


def detect_data_leakage(df: pd.DataFrame, target_col: str) -> dict:
    """
    Flag columns that are suspiciously perfectly correlated
    with the target (potential leakage).
    """
    if target_col not in df.columns:
        return {"leakage_candidates": [], "message": "Target column not found"}

    leakage = []
    num_df = df.select_dtypes(include=[np.number]).copy()

    # Encode target if categorical
    target = df[target_col]
    if target.dtype == object:
        target = target.astype("category").cat.codes

    for col in num_df.columns:
        if col == target_col:
            continue
        try:
            r, _ = stats.pearsonr(num_df[col].fillna(0), target.fillna(0))
            if abs(r) > 0.95:
                leakage.append(
                    {
                        "column": col,
                        "correlation_with_target": round(float(r), 4),
                        "risk": "HIGH" if abs(r) > 0.99 else "MEDIUM",
                    }
                )
        except Exception:
            continue

    return {
        "leakage_candidates": leakage,
        "message": (
            f"Found {len(leakage)} potential leakage columns"
            if leakage
            else "No obvious data leakage detected"
        ),
    }


def run_full_quality_check(
    df: pd.DataFrame, target_col: str | None = None
) -> dict:
    """Orchestrate all quality checks and return a single report dict."""
    report = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "column_names": df.columns.tolist(),
        "missing_values": analyze_missing_values(df),
        "duplicates": detect_duplicates(df),
        "data_types": validate_data_types(df),
        "class_imbalance": detect_class_imbalance(df, target_col),
        "correlations": compute_correlations(df),
    }
    if target_col:
        report["data_leakage"] = detect_data_leakage(df, target_col)
    return report