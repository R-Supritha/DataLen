"""
app.py – DataLen: Dataset Quality Inspector
Flask application entry point.
"""

import os
import json
import traceback
import uuid
from pathlib import Path

import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    jsonify,
    flash,
)
from werkzeug.utils import secure_filename

# ── Local modules ──────────────────────────────────────────────────────────────
from modules.quality_checker import run_full_quality_check
from modules.outlier_detector import run_all_outlier_detections
from modules.health_score import compute_health_score
from modules.visualizations import generate_all_charts
from modules.ai_assistant import build_system_prompt

# ── App Setup ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

ANALYSIS_STORE = BASE_DIR / "analysis_store"
ANALYSIS_STORE.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}


def _load_env_file(filepath: Path) -> None:
    """Load simple KEY=VALUE pairs from a local env file into os.environ."""
    if not filepath.exists():
        return

    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_env_file(BASE_DIR / "c.env")

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB


# ── Helpers ────────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_dataframe(filepath: str) -> pd.DataFrame:
    ext = filepath.rsplit(".", 1)[1].lower()
    if ext == "csv":
        return pd.read_csv(filepath)
    return pd.read_excel(filepath)


def _make_json_safe(obj):
    """Recursively convert non-JSON-safe values into built-in Python types."""
    import numpy as np

    if isinstance(obj, dict):
        return {key: _make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(value) for value in obj]
    if isinstance(obj, tuple):
        return [_make_json_safe(value) for value in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _save_analysis_blob(analysis: dict) -> str:
    """Save analysis results to a server-side JSON blob and return its key."""
    key = uuid.uuid4().hex
    path = ANALYSIS_STORE / f"{key}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(_make_json_safe(analysis), f)
    return key


def _load_analysis_blob(key: str) -> dict | None:
    """Load a previously saved analysis blob by key."""
    if not key:
        return None
    path = ANALYSIS_STORE / f"{key}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Home page – dataset upload form."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Handle file upload, run all analyses, save results to session."""
    if "file" not in request.files:
        flash("No file part in the request.", "danger")
        return redirect(url_for("index"))

    file = request.files["file"]
    target_col = request.form.get("target_column", "").strip() or None
    run_semantic = request.form.get("run_semantic") == "on"
    text_col = request.form.get("text_column", "").strip() or None

    if file.filename == "":
        flash("Please select a file before uploading.", "warning")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload a CSV or Excel file.", "danger")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        df = load_dataframe(filepath)

        # ── Run analyses ──
        try:
            quality_report = run_full_quality_check(df, target_col)
        except Exception:
            quality_report = {"error": traceback.format_exc()}

        try:
            outlier_report = run_all_outlier_detections(df)
        except Exception:
            outlier_report = {"error": traceback.format_exc()}

        # PyTorch Autoencoder
        ae_result = None
        try:
            from modules.autoencoder import train_autoencoder
            ae_result = train_autoencoder(df, epochs=40)
        except Exception:
            ae_result = {"error": traceback.format_exc()}

        # Health score
        try:
            health_report = compute_health_score(quality_report, outlier_report, ae_result)
        except Exception:
            health_report = {"score": 0, "error": traceback.format_exc()}

        # Semantic analysis (optional – slow due to model download)
        semantic_report = None
        if run_semantic and text_col:
            try:
                from modules.semantic_checker import detect_semantic_duplicates, detect_label_noise
                semantic_report = detect_semantic_duplicates(df, text_col)
                if target_col:
                    label_noise = detect_label_noise(df, text_col, target_col)
                    semantic_report["label_noise"] = label_noise
            except Exception:
                semantic_report = {"error": traceback.format_exc()}

        # Generate charts
        try:
            charts = generate_all_charts(df, quality_report, health_report, ae_result)
        except Exception:
            charts = {}

        # Dataset preview (first 10 rows, JSON-safe)
        preview_html = df.head(10).to_html(
            classes="table table-sm table-hover table-bordered preview-table",
            index=False,
            border=0,
            max_cols=20,
        )

        # Build analysis context for assistant
        analysis_context = {
            "quality": quality_report,
            "outliers": outlier_report,
            "health": health_report,
            "autoencoder": ae_result,
            "semantic": semantic_report,
        }

        # Store analysis server-side and keep only a lightweight session key
        session["analysis_key"] = _save_analysis_blob(analysis_context)
        session["filename"] = filename
        session["target_col"] = target_col
        session["charts"] = charts
        session["columns"] = df.columns.tolist()

        return redirect(url_for("report"))
    except Exception as e:
        app.logger.exception("Upload processing failed")
        flash(
            "An error occurred while analysing your dataset. Please try again or upload a different file.",
            "danger",
        )
        return redirect(url_for("index"))


@app.route("/report")
def report():
    """Render the full quality report page."""
    analysis = _load_analysis_blob(session.get("analysis_key", ""))
    if not analysis:
        flash("No analysis found. Please upload a dataset first.", "info")
        return redirect(url_for("index"))

    charts = session.get("charts", {})
    filename = session.get("filename", "Unknown")
    target_col = session.get("target_col")

    preview_html = ""
    if filename != "Unknown":
        try:
            df = load_dataframe(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            preview_html = df.head(10).to_html(
                classes="table table-sm table-hover table-bordered preview-table",
                index=False,
                border=0,
                max_cols=20,
            )
        except Exception:
            preview_html = ""

    return render_template(
        "report.html",
        analysis=analysis,
        charts=charts,
        preview_html=preview_html,
        filename=filename,
        target_col=target_col,
    )


@app.route("/assistant")
def assistant():
    """Render the AI assistant chat page."""
    analysis = _load_analysis_blob(session.get("analysis_key", ""))
    if not analysis:
        flash("Please upload a dataset first.", "info")
        return redirect(url_for("index"))
    return render_template(
        "assistant.html",
        filename=session.get("filename", "dataset"),
        score=analysis.get("health", {}).get("score", "?"),
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Proxy chat endpoint: receives user message, builds context-rich
    prompt, calls Gemini Flash 2.5 API, returns assistant reply.
    """
    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()
    history = data.get("history", [])   # list of {role, content}

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    analysis = _load_analysis_blob(session.get("analysis_key", "")) or {}
    system_prompt = build_system_prompt(analysis)

    # Build a single prompt string for Gemini Flash 2.5
    messages = history + [{"role": "user", "content": user_message}]
    prompt_parts = [f"SYSTEM: {system_prompt}"]
    for item in messages:
        role = item.get("role", "user").upper()
        content = item.get("content", "")
        prompt_parts.append(f"{role}: {content}")
    prompt_text = "\n\n".join(prompt_parts)

    try:
        import google.generativeai as genai
        from google.generativeai import GenerativeModel

        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)

        model = GenerativeModel(model_name="gemini-2.5-flash")
        response = model.generate_content(prompt_text)

        reply = getattr(response, "text", None)
        if not reply:
            reply = "No response text was returned from Gemini."

        return jsonify({"reply": reply})
    except ImportError:
        # Fallback: rule-based response when Gemini SDK is not installed
        reply = _rule_based_reply(user_message, analysis)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e), "reply": f"⚠️ API error: {e}"}), 500


def _rule_based_reply(message: str, analysis: dict) -> str:
    """Simple fallback replies when Gemini is unavailable."""
    msg = message.lower()
    health = analysis.get("health", {})
    score = health.get("score", "N/A")
    recs = health.get("recommendations", [])

    if any(k in msg for k in ["score", "health"]):
        return (
            f"Your dataset health score is **{score}/100** (Grade {health.get('grade', '?')}).\n\n"
            + "\n".join(f"- {r}" for r in recs)
        )
    if any(k in msg for k in ["missing", "null", "nan"]):
        mv = analysis.get("quality", {}).get("missing_values", {})
        return (
            f"**Missing values summary:**\n"
            f"- Overall: {mv.get('overall_pct', 0):.2f}% of cells missing\n"
            f"- Columns affected: {mv.get('columns_with_missing', 0)}\n\n"
            f"*Tip: Use `df.fillna(df.median())` for numeric columns or `SimpleImputer` from sklearn.*"
        )
    if any(k in msg for k in ["duplicate", "dup"]):
        dup = analysis.get("quality", {}).get("duplicates", {})
        return (
            f"**Duplicate rows:** {dup.get('count', 0)} ({dup.get('percentage', 0):.2f}%)\n\n"
            f"*Fix: `df.drop_duplicates(keep='first', inplace=True)`*"
        )
    if any(k in msg for k in ["outlier", "anomaly"]):
        iso = analysis.get("outliers", {}).get("isolation_forest", {})
        return (
            f"**Isolation Forest anomalies:** {iso.get('anomaly_count', 0)} rows "
            f"({iso.get('anomaly_percentage', 0):.1f}%)\n\n"
            f"*Consider using `RobustScaler` or removing extreme outliers before training.*"
        )
    if any(k in msg for k in ["recommend", "fix", "improve"]):
        return "**Top recommendations:**\n" + "\n".join(f"- {r}" for r in recs)

    return (
        "I'm the DataLen AI assistant. Ask me about:\n"
        "- Your dataset's health score\n"
        "- Missing values, duplicates, outliers\n"
        "- Class imbalance or data leakage\n"
        "- Recommended preprocessing steps\n"
    )


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)