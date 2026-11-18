# PathSnap: Distributed Map-Matching & Route Reconstruction

End-to-end machine learning pipeline for matching noisy GPS observations (`.track`) to directed road-network segments (`.arcs` over `.nodes`), followed by HMM/Viterbi route reconstruction and distributed FastAPI + UI deployment.

## Project Structure

```text
pathsnap/
├── src/
│   ├── data_loader.py       # Trip file loader & WGS84 local tangent-plane metric projection
│   ├── eda.py               # Dataset inventory, quality checks, graph connectivity & route audit
│   ├── eda_plots.py         # Matplotlib/Seaborn EDA charts & interactive OpenStreetMap Folium maps
│   ├── features.py          # Exact Shapely STRtree candidate generation, monotonic DP alignment & 33 features
│   ├── prepare_features.py  # Extract & cache candidate feature datasets (train & student_test)
│   ├── metrics.py           # Candidate, observation-level Top-1, continuity & sequence-level LCS F1 metrics
│   └── train_baseline.py    # Baseline Logistic Regression (5-Fold GroupKFold CV + student_test evaluation)
├── outputs/
│   ├── eda/                 # EDA CSV/JSON summaries, diagnostic plots, interactive Folium maps
│   ├── features/            # Candidate generation summary metadata
│   └── baseline/            # Baseline model weights, CV & student_test metrics, plots & Folium maps
├── requirements.txt
└── .gitignore
```

## Setup

Place the raw dataset in `dataset/` (`dataset/train/` with 80 trips and `dataset/student_test/` with 10 trips), then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Pipeline

```bash
# 1. Run dataset exploration & generate plots + Folium maps
python src/eda.py
python src/eda_plots.py

# 2. Generate & cache candidate road segments and 33 engineered features
python src/prepare_features.py

# 3. Train & evaluate the Logistic Regression baseline (5-Fold GroupKFold CV + student_test)
python src/train_baseline.py
```

## Current Baseline Summary (Classifier-Only, Pre-HMM)

| Model | Algorithm Family | 5-Fold CV Macro F1 | Student-Test Macro F1 | Student-Test Top-1 Acc | Candidate Miss Rate | Log Loss | Route Continuity (Step / Switch) | Seq LCS F1 | Inference Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression (Baseline) | Linear & Probabilistic | 0.9207 *(0.9150 ± 0.0229)* | 0.9255 | 87.85% *(87.99% cov)* | 0.16% | 0.0726 | 93.60% / 76.63% | 0.7359 | 20.90 ms (10 trips) |
