"""
Centralized 5-Model Registry spanning 5 Distinct Algorithm Families for PathSnap.

1. logreg        : Logistic Regression     (Linear and probabilistic)
2. decision_tree : Decision Tree           (Tree-based)
3. random_forest : Random Forest           (Ensemble / Bagging)
4. lightgbm      : LightGBM                (Boosting)
5. mlp           : PyTorch MLP (GPU)       (Neural networks)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .mlp import TorchMLPCandidateClassifier

RANDOM_SEED = 42


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    algorithm_family: str
    key_config_summary: str
    loss_objective: str
    hyperparams: Dict[str, Any]
    builder: Callable[[], Any]
    supports_val_history: bool = False


def get_model_registry(random_seed: int = RANDOM_SEED) -> Dict[str, ModelSpec]:
    """Return ordered dictionary of the 5 distinct PathSnap candidate classifiers."""
    specs: List[ModelSpec] = [
        ModelSpec(
            key="logreg",
            display_name="Logistic Regression",
            algorithm_family="Linear and probabilistic",
            key_config_summary="StandardScaler, C=1.0, L2, lbfgs",
            loss_objective="L2-regularized Binary Cross-Entropy (Log Loss)",
            hyperparams={
                "scaler": "StandardScaler",
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": random_seed,
            },
            builder=lambda: Pipeline([
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=random_seed,
                    ),
                ),
            ]),
        ),
        ModelSpec(
            key="decision_tree",
            display_name="Decision Tree",
            algorithm_family="Tree-based",
            key_config_summary="max_depth=14, min_samples_leaf=50, gini",
            loss_objective="Gini Impurity Minimization",
            hyperparams={
                "criterion": "gini",
                "max_depth": 14,
                "min_samples_leaf": 50,
                "random_state": random_seed,
            },
            builder=lambda: DecisionTreeClassifier(
                criterion="gini",
                max_depth=14,
                min_samples_leaf=50,
                random_state=random_seed,
            ),
        ),
        ModelSpec(
            key="random_forest",
            display_name="Random Forest",
            algorithm_family="Ensemble (Bagging)",
            key_config_summary="n_estimators=150, max_depth=18, min_leaf=20",
            loss_objective="Bagged Gini Impurity Minimization",
            hyperparams={
                "n_estimators": 150,
                "max_depth": 18,
                "min_samples_leaf": 20,
                "max_features": "sqrt",
                "n_jobs": -1,
                "random_state": random_seed,
            },
            builder=lambda: RandomForestClassifier(
                n_estimators=150,
                max_depth=18,
                min_samples_leaf=20,
                max_features="sqrt",
                n_jobs=-1,
                random_state=random_seed,
            ),
        ),
        ModelSpec(
            key="lightgbm",
            display_name="LightGBM",
            algorithm_family="Boosting",
            key_config_summary="n_est=400, lr=0.05, leaves=63, subsample=0.8",
            loss_objective="Binary Log Loss (Leaf-wise Gradient Boosting)",
            hyperparams={
                "n_estimators": 400,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "min_child_samples": 50,
                "subsample": 0.8,
                "subsample_freq": 1,
                "colsample_bytree": 0.8,
                "objective": "binary",
                "n_jobs": -1,
                "random_state": random_seed,
            },
            builder=lambda: LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=63,
                min_child_samples=50,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                objective="binary",
                n_jobs=-1,
                random_state=random_seed,
                verbose=-1,
            ),
        ),
        ModelSpec(
            key="mlp",
            display_name="PyTorch MLP",
            algorithm_family="Neural networks",
            key_config_summary="Hidden (128,64,32), BN, Drop=0.15, AdamW",
            loss_objective="Binary Cross-Entropy with Logits (AdamW + Cosine LR)",
            hyperparams={
                "scaler": "StandardScaler",
                "hidden_dims": [128, 64, 32],
                "dropout": 0.15,
                "lr": 2e-3,
                "weight_decay": 1e-4,
                "batch_size": 4096,
                "max_epochs": 25,
                "patience": 5,
                "random_state": random_seed,
            },
            builder=lambda: TorchMLPCandidateClassifier(
                hidden_dims=(128, 64, 32),
                dropout=0.15,
                lr=2e-3,
                weight_decay=1e-4,
                batch_size=4096,
                max_epochs=25,
                patience=5,
                random_state=random_seed,
            ),
            supports_val_history=True,
        ),
    ]
    return {s.key: s for s in specs}


MODEL_KEYS: List[str] = list(get_model_registry().keys())
