# PathSnap: Distributed Map-Matching & Route Reconstruction

End-to-end distributed machine learning system that reconstructs realistic vehicle trajectories by matching noisy 1 Hz GPS observations (`.track`) to directed road-network segments (`.arcs` over `.nodes`) using an **Optuna-Tuned XGBoost** candidate classifier paired with an **HMM + Viterbi** road-network decoder, deployed across a two-laptop **FastAPI + Interactive Leaflet UI** architecture.

## Project Structure

```text
pathsnap/
├── api/
│   └── server.py              # Laptop 1: FastAPI inference & server-side evaluation backend
├── ui/
│   ├── index.html             # Laptop 2: Evaluator-facing interactive Leaflet UI
│   └── serve_ui.py            # Zero-dependency static HTTP server for Laptop 2
├── artifacts/
│   ├── tuned_xgboost.joblib   # Self-contained CPU-portable Optuna-tuned XGBoost model (816 KB)
│   └── pipeline_config.json   # Feature schema, candidate-gen params & tuned HMM config
├── src/
│   ├── data_loader.py         # Raw trip loader & WGS84 local tangent-plane metric projection
│   ├── eda.py                 # Dataset inventory, quality checks, graph connectivity & route audit
│   ├── eda_plots.py           # Matplotlib/Seaborn EDA charts & interactive OpenStreetMap Folium maps
│   ├── features.py            # Exact Shapely STRtree candidate gen, monotonic DP alignment & 33 features
│   ├── prepare_features.py    # Extract & cache candidate feature datasets (train & student_test)
│   ├── metrics.py             # Candidate, Top-1, Route/Switch Continuity & Sequence LCS F1 metrics
│   ├── train_baseline.py      # Standalone Logistic Regression baseline script
│   ├── models/
│   │   ├── mlp.py             # GPU-accelerated PyTorch MLP with real epoch-wise train/val loss history
│   │   └── registry.py        # Central registry for all 5 models across 5 algorithm families
│   ├── experiment_runner.py   # Unified 5-Fold GroupKFold CV + student_test runner for all 5 models
│   ├── compare_models.py      # Consolidated comparison tables, compulsory plots & Folium maps
│   ├── tune_xgboost.py        # 30-trial Optuna TPE Bayesian optimization on XGBoost
│   ├── hmm_viterbi.py         # Log-space HMM/Viterbi decoder + OOF tuning + Clf-vs-HMM evaluation
│   ├── export_artifacts.py    # Export production inference artifacts to artifacts/
│   └── test_api_e2e.py        # End-to-end FastAPI integration & ground-truth isolation test
├── outputs/
│   ├── eda/                   # EDA summaries, plots & Folium maps
│   ├── features/              # Candidate generation summary metadata
│   ├── baseline/              # Baseline model results, plots & Folium maps
│   ├── experiments/           # 5-model CV/test logs, comparison tables, 7 plots & Folium maps
│   ├── tuning/                # Optuna trial logs, tuning results, tuned model & 3 diagnostic plots
│   └── hmm/                   # HMM tuning grid, Clf-vs-HMM results, 2 plots & 4 Folium comparison maps
├── requirements.txt
└── .gitignore
```

## Setup

Place the dataset folders under `dataset/` (`dataset/train/`, `dataset/student_test/`, and on evaluation day `dataset/evaluator_test/`), then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Full Offline Training & Evaluation Pipeline

```bash
# 1. Dataset exploration, plots & Folium maps
python src/eda.py
python src/eda_plots.py

# 2. Candidate generation (STRtree, R=60m, K=16) & 33 engineered features
python src/prepare_features.py

# 3. Train & compare all 5 models across 5 algorithm families (5-Fold GroupKFold CV + student_test)
python src/experiment_runner.py --all
python src/compare_models.py

# 4. Bayesian hyperparameter optimization (Optuna TPE, 30 trials on 5-Fold GroupKFold OOF Log Loss)
python src/tune_xgboost.py --n-trials 30

# 5. HMM + Viterbi sequential route reconstruction & decoder tuning (strictly on train OOF)
python src/hmm_viterbi.py

# 6. Export self-contained inference bundle & run end-to-end API integration test
python src/export_artifacts.py
python src/test_api_e2e.py
```

## Two-Laptop Deployment Guide

### Laptop 1 — Model & FastAPI Inference Backend
1. Place the 10 reserved evaluation trips in `dataset/evaluator_test/` (or point `PATHSNAP_EVAL_DIR` to their folder).
2. Start the FastAPI server bound to all network interfaces so Laptop 2 can reach it over LAN/hotspot:

```bash
# Uses dataset/evaluator_test automatically if present, else falls back to dataset/student_test
uvicorn api.server:app --host 0.0.0.0 --port 8000

# Or explicitly specify the evaluation trips folder:
PATHSNAP_EVAL_DIR=/path/to/evaluator_test uvicorn api.server:app --host 0.0.0.0 --port 8000
```

3. Keep this terminal visible during evaluation to show live incoming request & inference logs (`POST /predict`, stage latencies, and server-side evaluation metrics). Ground-truth `.route` files remain strictly server-side and are never exposed to the model or UI.

### Laptop 2 — Evaluator-Facing UI
On Laptop 2 (zero extra Python dependencies required — only standard library Python 3):

```bash
python ui/serve_ui.py --port 8501 --api-url http://<LAPTOP1_IP>:8000
```

Open the printed URL in a browser (`http://localhost:8501/?api=http://<LAPTOP1_IP>:8000`). You can also change the Laptop 1 API URL directly inside the UI header bar and click **Connect**.

---

## Experimental Results Summary

### 1. 5-Model Comparison (Classifier-Only, Pre-HMM, 5-Fold `GroupKFold(trip_id)`)

| Model | Algorithm Family | CV Macro F1 (OOF / Mean±SD) | Student-Test Top-1 Acc | Route Continuity (Step / Switch) | Student-Test Macro F1 | Log Loss (CV / Test) | Train Time | Test Inference |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear & Probabilistic | 0.9207 *(0.9150±0.0229)* | 87.85% | 93.60% / 76.63% | 0.9255 | 0.0853 / 0.0726 | 7.05s | 22.5 ms *(2.25 ms/trip)* |
| **Decision Tree** | Tree-based | 0.9084 *(0.9000±0.0502)* | 88.02% | 93.11% / 75.12% | 0.9305 | 0.1067 / 0.0781 | 57.39s | 10.0 ms *(1.00 ms/trip)* |
| **Random Forest** | Ensemble (Bagging) | 0.9300 *(0.9253±0.0201)* | 88.29% | 93.66% / 76.80% | 0.9348 | 0.0768 / 0.0585 | 119.45s | 220.8 ms *(22.08 ms/trip)* |
| **XGBoost (GPU)** | Boosting | **0.9306** *(0.9255±0.0214)* | **88.90%** | 94.33% / 78.63% | **0.9364** | **0.0715** / **0.0570** | 5.56s | 64.3 ms *(6.43 ms/trip)* |
| **PyTorch MLP (GPU)** | Neural Networks | 0.9295 *(0.9246±0.0202)* | 88.55% | **94.35%** / **78.75%** | 0.9347 | 0.0723 / 0.0587 | **4.90s** | 24.8 ms *(2.48 ms/trip)* |

### 2. Progression: Untuned XGBoost → Optuna-Tuned XGBoost → Tuned XGBoost + HMM/Viterbi (`student_test`, 10 trips)

| Pipeline Stage | Top-1 Acc (All) | Top-1 Acc (Covered) | Candidate Miss Rate | Step Continuity | Arc-Switch Continuity | Seq Arc F1 | Seq LCS F1 | Live API Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Untuned XGBoost (Clf-Only) | 88.90% | 89.04% | 0.16% | 94.33% | 78.63% | 0.8904 | 0.7684 | — |
| Optuna-Tuned XGBoost (Clf-Only) | 88.96% | 89.10% | 0.16% | 94.39% | 78.96% | 0.8915 | 0.7696 | ~120 ms/trip |
| **Optuna-Tuned XGBoost + HMM/Viterbi** | **94.96%** | **95.11%** | **0.16%** | **98.28%** | **90.75%** | **0.9396** | **0.9367** | **513.8 ms/trip** *(end-to-end)* |
