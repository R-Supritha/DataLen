"""
ai_assistant.py
Build a context-rich system prompt from analysis results
for the AI chat assistant.
"""

from __future__ import annotations
import json


def build_system_prompt(analysis_context: dict) -> str:
    """
    Convert the full analysis result dict into a concise but
    information-rich system prompt for the chat assistant.
    """
    health = analysis_context.get("health", {})
    quality = analysis_context.get("quality", {})
    outliers = analysis_context.get("outliers", {})

    shape = quality.get("shape", {})
    mv = quality.get("missing_values", {})
    dups = quality.get("duplicates", {})
    leakage = quality.get("data_leakage", {})
    imbalance = quality.get("class_imbalance", {})
    corr = quality.get("correlations", {})
    breakdown = health.get("breakdown", {})
    recs = health.get("recommendations", [])

    prompt = f"""You are DataLen AI, an expert data quality assistant embedded in the DataLen Dataset Quality Inspector app.

You have already analysed the user's dataset. Here is the full analysis summary:

## Dataset Overview
- Shape: {shape.get('rows', '?')} rows × {shape.get('columns', '?')} columns
- Overall health score: {health.get('score', '?')}/100 (Grade: {health.get('grade', '?')})
- Columns: {', '.join(quality.get('column_names', [])[:20])}

## Missing Values
- Overall missing rate: {mv.get('overall_pct', 0):.2f}%
- Columns with missing data: {mv.get('columns_with_missing', 0)}
- Worst columns: {json.dumps({k: v for k, v in list(mv.get('percentages', {}).items())[:5]})}

## Duplicates
- Exact duplicate rows: {dups.get('count', 0)} ({dups.get('percentage', 0):.2f}%)

## Outliers
- Z-Score outliers total: {outliers.get('zscore', {}).get('total_outliers', 0)}
- IQR outliers total: {outliers.get('iqr', {}).get('total_outliers', 0)}
- Isolation Forest anomalies: {outliers.get('isolation_forest', {}).get('anomaly_count', 0)} ({outliers.get('isolation_forest', {}).get('anomaly_percentage', 0):.1f}%)

## Class Imbalance
{_fmt_imbalance(imbalance)}

## Data Leakage
{leakage.get('message', 'Not analysed (no target column selected)')}
High-risk columns: {[c['column'] for c in leakage.get('leakage_candidates', []) if c.get('risk') == 'HIGH']}

## High Correlations
{json.dumps(corr.get('high_correlations', [])[:5], indent=2)}

## Health Score Breakdown
{_fmt_breakdown(breakdown)}

## Top Recommendations
{chr(10).join(f'- {r}' for r in recs)}

---
Your role:
- Answer questions about the dataset's quality issues
- Explain each metric in simple, actionable terms
- Recommend specific preprocessing steps (with code snippets when helpful)
- Be concise, friendly, and technically precise
- Always relate answers back to this specific dataset's analysis results
- When asked for code, use pandas/scikit-learn/imbalanced-learn idioms
- Do NOT make up data not in the analysis above

Respond naturally in Markdown.
"""
    return prompt


def _fmt_imbalance(imbalance: dict) -> str:
    if not imbalance:
        return "No low-cardinality columns detected."
    lines = []
    for col, info in list(imbalance.items())[:3]:
        lines.append(
            f"- {col}: ratio {info.get('imbalance_ratio', '?')}:1 "
            f"({'imbalanced' if info.get('is_imbalanced') else 'balanced'})"
        )
    return "\n".join(lines) if lines else "No class imbalance detected."


def _fmt_breakdown(breakdown: dict) -> str:
    lines = []
    for dim, info in breakdown.items():
        lines.append(
            f"- {dim.replace('_', ' ').title()}: {info.get('score', '?')}/{info.get('max', '?')} pts"
        )
    return "\n".join(lines)