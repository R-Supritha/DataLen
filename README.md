# 🔬 DataLen – Dataset Quality Inspector

A portfolio-quality Flask web application that automatically analyses CSV/Excel datasets and generates a comprehensive quality report, health score, visualisations, anomaly detection results, and AI-powered improvement recommendations.

---

## ✨ Features

| Feature | Technology |
|---|---|
| Missing value analysis | Pandas / NumPy |
| Duplicate detection | Pandas |
| Data type validation | Pandas |
| Class imbalance detection | Scikit-learn |
| Correlation analysis | Pandas / Seaborn |
| Outlier detection (Z-Score, IQR) | SciPy / NumPy |
| Anomaly detection (Isolation Forest) | Scikit-learn |
| Deep anomaly detection | **PyTorch Autoencoder** |
| Semantic duplicate detection | **HuggingFace MiniLM** |
| Label noise detection | **HuggingFace MiniLM** |
| Data leakage detection | SciPy |
| Dataset health score (0-100) | Custom scoring engine |
| AI chat assistant | Google Gemini 2.5 Flash |
| Visualisations | Matplotlib / Seaborn |
| Web UI | Flask / Bootstrap 5 |

---

## 🚀 Quick Start

### 1. Clone / Extract the project

```bash
cd DataLen
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch and sentence-transformers are large packages. First install may take a few minutes.

### 4. Set your Google API key (for AI chat assistant)

```bash
export GOOGLE_API_KEY="AIza..."
```

Get a free key at https://aistudio.google.com
The app runs without the key – the AI chat falls back to a rule-based assistant.

### 5. Run the app

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 📁 Project Structure

```
DataLen/
├── app.py                    # Flask routes & application entry point
├── requirements.txt
├── README.md
│
├── modules/
│   ├── quality_checker.py    # Missing values, duplicates, types, correlations, leakage
│   ├── outlier_detector.py   # Z-Score, IQR, Isolation Forest
│   ├── semantic_checker.py   # HuggingFace MiniLM semantic duplicates & label noise
│   ├── autoencoder.py        # PyTorch Autoencoder anomaly detection
│   ├── health_score.py       # 0-100 health score computation
│   ├── visualizations.py     # Matplotlib/Seaborn chart generation
│   └── ai_assistant.py       # System prompt builder for the chat assistant
│
├── templates/
│   ├── index.html            # Upload / home page
│   ├── report.html           # Full quality report
│   └── assistant.html        # AI chat interface
│
├── static/
│   ├── css/style.css
│   └── charts/               # Auto-generated chart PNGs
│
└── uploads/                  # Temporary uploaded files
```

---

## 📊 Health Score Breakdown

| Dimension | Max Points | Penalty Basis |
|---|---|---|
| Missing Values | 25 | Overall missing % |
| Duplicates | 15 | Duplicate row % |
| Outliers | 20 | IQR outlier rate |
| Class Imbalance | 15 | Worst imbalance ratio |
| Label Noise | 15 | Semantic noise rate |
| Data Leakage | 10 | High/medium risk columns |
| **Total** | **100** | |

---

## 🤖 PyTorch Autoencoder Architecture

```
Input (N features)
    → Linear(N → 4N) → ReLU → Dropout(0.1)
    → Linear(4N → 2N) → ReLU
    → Linear(2N → N) → ReLU         [Bottleneck]
    → Linear(N → 2N) → ReLU
    → Linear(2N → 4N) → ReLU → Dropout(0.1)
    → Linear(4N → N)
Output (N features)
```

Samples with reconstruction error > 95th percentile are flagged as anomalies.

---

## 🧠 HuggingFace Integration

Model: `sentence-transformers/all-MiniLM-L6-v2`

- **Semantic Duplicate Detection**: Embeds all text values in a column, computes pairwise cosine similarity, flags pairs above threshold (default 0.92) that are not exact duplicates.
- **Label Noise Detection**: Finds semantically similar text pairs that have different labels – a common source of training noise.

> The model (~90 MB) is downloaded automatically on first use.

---

## 📸 Pages

### Home – Upload
Drag-and-drop CSV/XLSX upload with advanced options (target column, semantic analysis toggle).

### Report
- Dataset overview stats
- Health score gauge with per-dimension breakdown
- AI recommendations
- Tabbed sections: Quality Checks, Outliers, Autoencoder, Visualisations, Semantic

### AI Assistant
Context-aware chat powered by the full analysis results. Includes suggested questions and Markdown rendering.

---

## 🛠 Supported File Formats

- `.csv` – any delimiter (auto-detected by pandas)
- `.xlsx` / `.xls` – Excel workbooks

Maximum file size: **50 MB**

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | env var | Required for Claude-powered chat |
| `MAX_CONTENT_LENGTH` | 50 MB | Max upload size |
| Autoencoder epochs | 40 | Adjustable in `app.py` |
| Semantic threshold | 0.92 | Adjustable in route call |
| Outlier contamination | 5% | Isolation Forest parameter |

---

## 📦 Dependencies

- Flask 3.0
- PyTorch 2.3
- sentence-transformers 3.0 (HuggingFace)
- scikit-learn 1.5
- pandas 2.2
- numpy 1.26
- matplotlib 3.9
- seaborn 0.13
- scipy 1.13
- google-generativeai (for Gemini 2.5 Flash chat)

---

## 🎓 Portfolio Notes

This project demonstrates:
- **Full-stack ML application** development with Flask
- **Deep learning** (PyTorch autoencoder) for unsupervised anomaly detection
- **NLP** (HuggingFace sentence transformers) for semantic similarity
- **Classical ML** (Isolation Forest, statistical outlier detection)
- **Data engineering** best practices (quality scoring, leakage detection)
- **Production-ready** patterns (modular architecture, error handling, session management)
