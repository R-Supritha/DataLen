"""
outlier_detector.py
Outlier detection using Z-Score, IQR, and Isolation Forest.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_zscore_outliers(df: pd.DataFrame, threshold: float = 3.0) -> dict:
    """Flag values with |z-score| > threshold per numeric column."""
    num_df = df.select_dtypes(include=[np.number])
    results = {}
    for col in num_df.columns:
        series = num_df[col].dropna()
        if len(series) < 3:
            continue
        z = np.abs((series - series.mean()) / series.std(ddof=0))
        outlier_idx = series[z > threshold].index.tolist()
        results[col] = {
            "count": len(outlier_idx),
            "percentage": round(len(outlier_idx) / len(series) * 100, 2),
            "indices": outlier_idx[:10],
            "values": series.loc[outlier_idx[:10]].round(4).tolist(),
        }
    total = sum(v["count"] for v in results.values())
    return {"per_column": results, "total_outliers": total, "method": "Z-Score"}


def detect_iqr_outliers(df: pd.DataFrame, factor: float = 1.5) -> dict:
    """Flag values outside [Q1 - factor*IQR, Q3 + factor*IQR]."""
    num_df = df.select_dtypes(include=[np.number])
    results = {}
    for col in num_df.columns:
        series = num_df[col].dropna()
        if len(series) < 3:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        mask = (series < lower) | (series > upper)
        outlier_idx = series[mask].index.tolist()
        results[col] = {
            "count": len(outlier_idx),
            "percentage": round(len(outlier_idx) / len(series) * 100, 2),
            "lower_bound": round(float(lower), 4),
            "upper_bound": round(float(upper), 4),
            "indices": outlier_idx[:10],
            "values": series.loc[outlier_idx[:10]].round(4).tolist(),
        }
    total = sum(v["count"] for v in results.values())
    return {"per_column": results, "total_outliers": total, "method": "IQR"}


def detect_isolation_forest_outliers(
    df: pd.DataFrame, contamination: float = 0.05, random_state: int = 42
) -> dict:
    """
    Use Isolation Forest on all numeric columns combined.
    Returns per-row anomaly scores and a list of anomalous row indices.
    """
    num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if num_df.shape[1] == 0:
        return {
            "anomaly_indices": [],
            "anomaly_count": 0,
            "anomaly_percentage": 0.0,
            "method": "Isolation Forest",
            "scores": [],
        }

    # Fill remaining NaNs with column median
    X = num_df.fillna(num_df.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = IsolationForest(
        contamination=contamination, random_state=random_state, n_estimators=100
    )
    preds = clf.fit_predict(X_scaled)          # -1 = anomaly, 1 = normal
    scores = -clf.score_samples(X_scaled)      # higher = more anomalous

    anomaly_mask = preds == -1
    anomaly_indices = np.where(anomaly_mask)[0].tolist()

    return {
        "anomaly_indices": anomaly_indices[:50],
        "anomaly_count": int(anomaly_mask.sum()),
        "anomaly_percentage": round(float(anomaly_mask.mean()) * 100, 2),
        "scores": scores.round(4).tolist()[:200],   # cap for JSON size
        "method": "Isolation Forest",
    }


def run_all_outlier_detections(df: pd.DataFrame) -> dict:
    """Run all three methods and return combined results."""
    return {
        "zscore": detect_zscore_outliers(df),
        "iqr": detect_iqr_outliers(df),
        "isolation_forest": detect_isolation_forest_outliers(df),
    }