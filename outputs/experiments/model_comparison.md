# PathSnap — Consolidated 5-Model Comparison (Classifier-Only, Pre-HMM)

| Model | Algorithm Family | Key Configuration | CV Macro F1 (OOF / Fold Mean±SD) | Student-Test Top-1 Acc | Route Continuity (Step / Switch) | Student-Test Macro F1 | Log Loss (CV / Test) | Train Time | Test Inference (10 trips) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear and probabilistic | StandardScaler, C=1.0, L2, lbfgs | 0.9207 (0.9150±0.0229) | 87.85% | 93.60% / 76.63% | 0.9255 | 0.0853 / 0.0726 | 7.05s | 22.5 ms (2.25 ms/trip) |
| **Decision Tree** | Tree-based | max_depth=14, min_samples_leaf=50, gini | 0.9084 (0.9000±0.0502) | 88.02% | 93.11% / 75.12% | 0.9305 | 0.1067 / 0.0781 | 57.38s | 10.0 ms (1.00 ms/trip) |
| **Random Forest** | Ensemble (Bagging) | n_estimators=150, max_depth=18, min_leaf=20 | 0.9300 (0.9253±0.0201) | 88.29% | 93.66% / 76.80% | 0.9348 | 0.0768 / 0.0585 | 119.45s | 220.8 ms (22.08 ms/trip) |
| **LightGBM** | Boosting | n_est=400, lr=0.05, leaves=63, subsample=0.8 | 0.9276 (0.9215±0.0259) | 89.16% | 94.16% / 78.40% | 0.9373 | 0.0737 / 0.0573 | 33.93s | 378.3 ms (37.83 ms/trip) |
| **PyTorch MLP** | Neural networks | Hidden (128,64,32), BN, Drop=0.15, AdamW | 0.9295 (0.9246±0.0202) | 88.55% | 94.35% / 78.75% | 0.9347 | 0.0723 / 0.0587 | 4.90s | 24.8 ms (2.48 ms/trip) |

*Candidate Miss Rate is identical across all models (`0.16%` on `student_test`, `2.48%` trip-mean on `train`) because all models share the exact same candidate generation (`STRtree`, `radius=60m`, `max_k=16`).*
