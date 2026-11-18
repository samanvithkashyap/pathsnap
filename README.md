# PathSnap: Distributed Map-Matching & Route Reconstruction

End-to-end machine learning pipeline for matching noisy GPS observations (`.track`) to directed road-network segments (`.arcs` over `.nodes`), followed by Bayesian hyperparameter optimization, HMM/Viterbi route reconstruction, and distributed FastAPI + UI deployment.

## Project Structure

```text
pathsnap/
├── src/
│   ├── data_loader.py         # Trip file loader & WGS84 local tangent-plane metric projection
│   ├── eda.py                 # Dataset inventory, quality checks, graph connectivity & route audit
│   ├── eda_plots.py           # Matplotlib/Seaborn EDA charts & interactive OpenStreetMap Folium maps
│   ├── features.py            # Exact Shapely STRtree candidate generation, monotonic DP alignment & 33 features
│   ├── prepare_features.py    # Extract & cache candidate feature datasets (train & student_test)
│   ├── metrics.py             # Candidate, observation-level Top-1, continuity & sequence-level LCS F1 metrics
│   ├── train_baseline.py      # Standalone Logistic Regression baseline script
│   ├── models/
│   │   ├── mlp.py             # GPU-accelerated PyTorch MLP with real epoch-wise train/val loss history
│   │   └── registry.py        # Central registry for all 5 models across 5 algorithm families
│   ├── experiment_runner.py   # Unified 5-Fold GroupKFold CV + student_test runner for all models
│   └── compare_models.py      # Consolidated comparison tables, compulsory plots & Folium maps
├── outputs/
│   ├── eda/                   # EDA CSV/JSON summaries, diagnostic plots, interactive Folium maps
│   ├── features/              # Candidate generation summary metadata
│   ├── baseline/              # Baseline model results, plots & Folium maps
│   └── experiments/           # 5-model CV/test logs, comparison tables, plots & Folium maps
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

# 3. Train & evaluate all 5 models across 5 algorithm families (5-Fold GroupKFold CV + student_test)
python src/experiment_runner.py --all

# 4. Generate consolidated comparison tables, plots, confusion matrices & Folium maps
python src/compare_models.py
```

## 5-Model Comparison Summary (Classifier-Only, Pre-HMM)

*All models use the exact same 33 engineered features, 5-Fold `GroupKFold(trip_id)` CV on the 80 training trips, and `0.16%` Candidate Miss Rate on `student_test`.*

| Model | Algorithm Family | Key Configuration | CV Macro F1 (OOF / Mean±SD) | Student-Test Top-1 Acc | Route Continuity (Step / Switch) | Student-Test Macro F1 | Log Loss (CV / Test) | Train Time | Test Inference (10 trips) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear & Probabilistic | StandardScaler, C=1.0, L2, lbfgs | 0.9207 *(0.9150±0.0229)* | 87.85% | 93.60% / 76.63% | 0.9255 | 0.0853 / 0.0726 | 7.05s | 22.5 ms *(2.25 ms/trip)* |
| **Decision Tree** | Tree-based | max_depth=14, min_samples_leaf=50, gini | 0.9084 *(0.9000±0.0502)* | 88.02% | 93.11% / 75.12% | 0.9305 | 0.1067 / 0.0781 | 57.39s | 10.0 ms *(1.00 ms/trip)* |
| **Random Forest** | Ensemble (Bagging) | n_estimators=150, max_depth=18, min_leaf=20 | **0.9300** *(0.9253±0.0201)* | 88.29% | 93.66% / 76.80% | 0.9348 | 0.0768 / 0.0585 | 119.45s | 220.8 ms *(22.08 ms/trip)* |
| **LightGBM** | Boosting | n_est=400, lr=0.05, leaves=63, subsample=0.8 | 0.9276 *(0.9215±0.0259)* | **89.16%** | 94.16% / 78.40% | **0.9373** | 0.0737 / **0.0573** | 33.93s | 378.3 ms *(37.83 ms/trip)* |
| **PyTorch MLP (GPU)** | Neural Networks | Hidden (128,64,32), BN, Drop=0.15, AdamW | 0.9295 *(0.9246±0.0202)* | 88.55% | **94.35%** / **78.75%** | 0.9347 | **0.0723** / 0.0587 | **4.90s** | 24.8 ms *(2.48 ms/trip)* |
