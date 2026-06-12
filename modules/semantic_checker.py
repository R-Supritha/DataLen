"""
semantic_checker.py
Hugging Face Sentence Transformers for:
  - Semantic duplicate detection across text columns
  - Text label-noise detection (inconsistent labels for similar text)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def _load_model():
    """Lazy-load the SentenceTransformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        raise RuntimeError(f"Failed to load SentenceTransformer: {e}")


def detect_semantic_duplicates(
    df: pd.DataFrame,
    text_col: str,
    threshold: float = 0.92,
    max_rows: int = 500,
) -> dict:
    """
    Embed text in `text_col` using MiniLM and find pairs with
    cosine similarity >= threshold that are not exact duplicates.
    """
    if text_col not in df.columns:
        return {"error": f"Column '{text_col}' not found", "pairs": []}

    series = df[text_col].dropna().astype(str)
    if len(series) > max_rows:
        series = series.sample(max_rows, random_state=42)

    texts = series.tolist()
    indices = series.index.tolist()

    model = _load_model()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
    sim_matrix = cosine_similarity(embeddings)

    pairs = []
    n = len(texts)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim >= threshold and texts[i] != texts[j]:
                pairs.append(
                    {
                        "idx_a": int(indices[i]),
                        "idx_b": int(indices[j]),
                        "text_a": texts[i][:120],
                        "text_b": texts[j][:120],
                        "similarity": round(sim, 4),
                    }
                )

    return {
        "column": text_col,
        "threshold": threshold,
        "semantic_duplicate_pairs": sorted(
            pairs, key=lambda x: x["similarity"], reverse=True
        )[:20],
        "total_pairs_found": len(pairs),
        "rows_analyzed": len(texts),
    }


def detect_label_noise(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    threshold: float = 0.90,
    max_rows: int = 500,
) -> dict:
    """
    Find text pairs that are semantically very similar but have different labels –
    potential label noise or inconsistency.
    """
    if text_col not in df.columns or label_col not in df.columns:
        return {
            "error": "One or more specified columns not found",
            "noisy_pairs": [],
        }

    sub = df[[text_col, label_col]].dropna().astype(str)
    if len(sub) > max_rows:
        sub = sub.sample(max_rows, random_state=42)

    texts = sub[text_col].tolist()
    labels = sub[label_col].tolist()
    idx = sub.index.tolist()

    model = _load_model()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
    sim_matrix = cosine_similarity(embeddings)

    noisy = []
    n = len(texts)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim >= threshold and labels[i] != labels[j]:
                noisy.append(
                    {
                        "idx_a": int(idx[i]),
                        "idx_b": int(idx[j]),
                        "text_a": texts[i][:120],
                        "text_b": texts[j][:120],
                        "label_a": labels[i],
                        "label_b": labels[j],
                        "similarity": round(sim, 4),
                    }
                )

    return {
        "text_column": text_col,
        "label_column": label_col,
        "threshold": threshold,
        "noisy_pairs": sorted(noisy, key=lambda x: x["similarity"], reverse=True)[
            :20
        ],
        "total_noisy_pairs": len(noisy),
        "rows_analyzed": len(texts),
        "noise_rate": round(len(noisy) / max(1, n * (n - 1) / 2) * 100, 4),
    }