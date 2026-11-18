"""
Bayesian Hyperparameter Optimization (Optuna TPE) for XGBoost on PathSnap.

Strategy:
- Sampler: Optuna TPESampler (Tree-structured Parzen Estimator, seed=42, n_startup_trials=8)
- Objective: Minimize 5-Fold GroupKFold(trip_id) OOF Binary Log Loss on the 80 training trips
  (strictly zero leakage from student_test or evaluator_test).
- Trial 0 is enqueued with the untuned XGBoost configuration as a direct benchmark anchor.
- Uses GPU-accelerated xgboost.train (tree_method="hist", device="cuda" if available) with
  pre-built fold DMatrices and early_stopping_rounds=40 for fast, reproducible tuning.
- Refits the final tuned XGBClassifier on all 80 training trips (covered observations only)
  using median(best_iteration) from CV, then compares Untuned vs Tuned on CV and student_test.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
import xgboost as xgb
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment_runner import build_covered_row_mask, evaluate_predictions
from features import FEATURE_COLUMNS

PROJECT_ROOT = SRC_DIR.parent
FEATURES_DIR = PROJECT_ROOT / "outputs" / "features"
EXPERIMENTS_DIR = PROJECT_ROOT / "outputs" / "experiments"
OUT_DIR = PROJECT_ROOT / "outputs" / "tuning"
PLOTS_DIR = OUT_DIR / "plots"

RANDOM_SEED = 42
N_FOLDS = 5
MAX_BOOST_ROUNDS = 1200
EARLY_STOPPING_ROUNDS = 40

SEARCH_SPACE_SPEC: Dict[str, Any] = {
    "max_depth": {"type": "int", "low": 4, "high": 10},
    "learning_rate": {"type": "float", "low": 0.015, "high": 0.15, "log": True},
    "min_child_weight": {"type": "float", "low": 5.0, "high": 200.0, "log": True},
    "subsample": {"type": "float", "low": 0.60, "high": 1.00},
    "colsample_bytree": {"type": "float", "low": 0.50, "high": 1.00},
    "gamma": {"type": "float", "low": 1e-4, "high": 5.0, "log": True},
    "reg_alpha": {"type": "float", "low": 1e-4, "high": 20.0, "log": True},
    "reg_lambda": {"type": "float", "low": 1e-2, "high": 50.0, "log": True},
    "max_bin": {"type": "categorical", "choices": [256, 512]},
    "n_estimators_strategy": f"Early stopping (patience={EARLY_STOPPING_ROUNDS}, max_rounds={MAX_BOOST_ROUNDS}); full refit uses median CV best_iteration",
}


def plot_tuning_results(
    trials_df: pd.DataFrame,
    study: optuna.Study,
    untuned_res: Dict[str, Any],
    tuned_res: Dict[str, Any],
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150})

    # 1. Optimization history
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    xs = trials_df["trial"].to_numpy()
    ys = trials_df["cv_oof_log_loss"].to_numpy()
    best_so_far = np.minimum.accumulate(ys)
    untuned_ll = untuned_res["cv_5fold_oof"]["candidate_metrics"]["log_loss"]

    ax.scatter(xs, ys, color="#64748b", alpha=0.75, s=45, label="Trial 5-Fold CV OOF Log Loss")
    ax.plot(xs, best_so_far, color="#2563eb", linewidth=2.4, label="Best-So-Far CV Log Loss")
    ax.axhline(untuned_ll, color="#dc2626", linestyle="--", linewidth=1.6, label=f"Untuned XGBoost ({untuned_ll:.4f})")

    best_idx = int(np.argmin(ys))
    ax.scatter([xs[best_idx]], [ys[best_idx]], color="#16a34a", s=110, zorder=5, edgecolor="black", label=f"Best Trial #{xs[best_idx]} ({ys[best_idx]:.4f})")

    ax.set_xlabel("Optuna Trial Number (TPE Bayesian Search)")
    ax.set_ylabel("5-Fold GroupKFold OOF Binary Log Loss (Lower is Better)")
    ax.set_title("Bayesian Hyperparameter Optimization History — XGBoost", fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_optuna_optimization_history.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Hyperparameter importance (Optuna fANOVA) + top-2 parameter trends
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception:
        importances = {}

    if importances:
        imp_items = list(importances.items())[::-1]
        axes[0].barh([k for k, _ in imp_items], [v for _, v in imp_items], color="#2563eb", alpha=0.88)
        axes[0].set_xlabel("Relative Importance (fANOVA)")
        axes[0].set_title("Hyperparameter Importance")
        top2 = list(importances.keys())[:2]
    else:
        top2 = ["max_depth", "learning_rate"]
        axes[0].text(0.5, 0.5, "Importance unavailable", ha="center", va="center")

    for ax, param in zip(axes[1:], top2):
        vals = trials_df[f"param_{param}"]
        sc = ax.scatter(vals, ys, c=xs, cmap="viridis", s=55, alpha=0.85, edgecolor="white")
        ax.set_xlabel(param)
        ax.set_ylabel("5-Fold CV OOF Log Loss")
        ax.set_title(f"Objective vs {param}")
        if param in ("learning_rate", "min_child_weight", "gamma", "reg_alpha", "reg_lambda"):
            ax.set_xscale("log")
        fig.colorbar(sc, ax=ax, label="Trial #")

    fig.suptitle("Optuna Bayesian Search Diagnostics — Parameter Sensitivity", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_hyperparameter_importance.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Untuned vs Tuned side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    labels_acc = ["CV Macro F1", "Test Macro F1", "Test Top-1 Acc", "Step Cont.", "Switch Cont.", "Seq LCS F1"]
    untuned_vals = [
        untuned_res["cv_5fold_oof"]["candidate_metrics"]["macro_f1"] * 100,
        untuned_res["student_test"]["candidate_metrics"]["macro_f1"] * 100,
        untuned_res["student_test"]["route_metrics"]["top1_accuracy"] * 100,
        untuned_res["student_test"]["route_metrics"]["route_continuity"] * 100,
        untuned_res["student_test"]["route_metrics"]["transition_continuity_trip_mean"] * 100,
        untuned_res["student_test"]["route_metrics"]["seq_lcs_f1_mean"] * 100,
    ]
    tuned_vals = [
        tuned_res["cv_5fold_oof"]["candidate_metrics"]["macro_f1"] * 100,
        tuned_res["student_test"]["candidate_metrics"]["macro_f1"] * 100,
        tuned_res["student_test"]["route_metrics"]["top1_accuracy"] * 100,
        tuned_res["student_test"]["route_metrics"]["route_continuity"] * 100,
        tuned_res["student_test"]["route_metrics"]["transition_continuity_trip_mean"] * 100,
        tuned_res["student_test"]["route_metrics"]["seq_lcs_f1_mean"] * 100,
    ]

    x = np.arange(len(labels_acc))
    w = 0.36
    b1 = axes[0].bar(x - w / 2, untuned_vals, width=w, label="Untuned XGBoost", color="#94a3b8")
    b2 = axes[0].bar(x + w / 2, tuned_vals, width=w, label="Tuned XGBoost (Optuna)", color="#2563eb")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            axes[0].text(b.get_x() + b.get_width() / 2, h + 0.35, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)
    axes[0].set_ylim(72, 100)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels_acc, rotation=18, ha="right")
    axes[0].set_ylabel("Score (%)")
    axes[0].set_title("Accuracy, F1 & Continuity: Untuned vs Tuned")
    axes[0].legend(fontsize=8.5, loc="lower right")

    # Right: Log Loss (lower is better)
    ll_labels = ["5-Fold CV OOF Log Loss", "Student-Test Log Loss"]
    ll_u = [
        untuned_res["cv_5fold_oof"]["candidate_metrics"]["log_loss"],
        untuned_res["student_test"]["candidate_metrics"]["log_loss"],
    ]
    ll_t = [
        tuned_res["cv_5fold_oof"]["candidate_metrics"]["log_loss"],
        tuned_res["student_test"]["candidate_metrics"]["log_loss"],
    ]
    x2 = np.arange(len(ll_labels))
    l1 = axes[1].bar(x2 - w / 2, ll_u, width=w, label="Untuned XGBoost", color="#94a3b8")
    l2 = axes[1].bar(x2 + w / 2, ll_t, width=w, label="Tuned XGBoost (Optuna)", color="#16a34a")
    for bars in (l1, l2):
        for b in bars:
            h = b.get_height()
            axes[1].text(b.get_x() + b.get_width() / 2, h + 0.001, f"{h:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1].set_ylim(0, max(ll_u) * 1.25)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(ll_labels)
    axes[1].set_ylabel("Binary Log Loss (Lower is Better)")
    axes[1].set_title("Probability Calibration: Untuned vs Tuned")
    axes[1].legend(fontsize=8.5)

    fig.suptitle("XGBoost Performance Before vs After Bayesian Hyperparameter Optimization", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_untuned_vs_tuned_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian Hyperparameter Optimization for XGBoost")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna TPE trials")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading cached candidate datasets... (device={device}, n_trials={args.n_trials})")
    train_pack = joblib.load(FEATURES_DIR / "train_candidates.joblib")
    test_pack = joblib.load(FEATURES_DIR / "student_test_candidates.joblib")

    train_df: pd.DataFrame = train_pack["df"]
    train_meta: Dict[str, Any] = train_pack["trip_meta"]
    test_df: pd.DataFrame = test_pack["df"]
    test_meta: Dict[str, Any] = test_pack["trip_meta"]

    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=np.int8)
    groups_train = train_df["trip_id"].to_numpy()
    covered_train_rows = build_covered_row_mask(train_df, train_meta)
    X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    # Pre-build GroupKFold splits and DMatrices once (eliminates CPU->DMatrix overhead across 30 trials)
    print("Pre-building 5-Fold GroupKFold DMatrices...")
    gkf = GroupKFold(n_splits=N_FOLDS)
    folds: List[Tuple[np.ndarray, np.ndarray, xgb.DMatrix, xgb.DMatrix]] = []
    for tr_idx, va_idx in gkf.split(X_train, y_train, groups_train):
        tr_fit_idx = tr_idx[covered_train_rows[tr_idx]]
        dtr = xgb.DMatrix(X_train[tr_fit_idx], label=y_train[tr_fit_idx])
        dva = xgb.DMatrix(X_train[va_idx], label=y_train[va_idx])
        folds.append((tr_fit_idx, va_idx, dtr, dva))

    trial_records: List[Dict[str, Any]] = []
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "device": device,
            "seed": RANDOM_SEED,
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.15, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 200.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.60, 1.00),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 1.00),
            "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
            "max_bin": trial.suggest_categorical("max_bin", [256, 512]),
        }

        oof_probs = np.zeros(len(train_df), dtype=np.float64)
        best_iters: List[int] = []
        fold_lls: List[float] = []

        for fold_i, (_, va_idx, dtr, dva) in enumerate(folds):
            bst = xgb.train(
                params=params,
                dtrain=dtr,
                num_boost_round=MAX_BOOST_ROUNDS,
                evals=[(dva, "val")],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose_eval=False,
            )
            best_it = int(bst.best_iteration) + 1
            best_iters.append(best_it)
            va_probs = bst.predict(dva, iteration_range=(0, best_it))
            oof_probs[va_idx] = va_probs
            f_ll = float(log_loss(y_train[va_idx], np.clip(va_probs, 1e-7, 1.0 - 1e-7), labels=[0, 1]))
            fold_lls.append(f_ll)

            # Report intermediate running mean Log Loss for Optuna MedianPruner (after fold 2+)
            trial.report(float(np.mean(fold_lls)), step=fold_i)
            if fold_i >= 1 and trial.should_prune():
                raise optuna.TrialPruned()

        oof_ll = float(log_loss(y_train, np.clip(oof_probs, 1e-7, 1.0 - 1e-7), labels=[0, 1]))
        oof_f1 = float(f1_score(y_train, (oof_probs >= 0.5).astype(np.int8), average="macro"))
        med_iter = int(round(float(np.median(best_iters))))
        elapsed = time.time() - t0

        trial.set_user_attr("cv_oof_macro_f1", oof_f1)
        trial.set_user_attr("median_best_iteration", med_iter)
        trial.set_user_attr("best_iterations", best_iters)
        trial.set_user_attr("elapsed_s", round(elapsed, 2))

        rec = {
            "trial": trial.number,
            "cv_oof_log_loss": oof_ll,
            "cv_fold_log_loss_mean": float(np.mean(fold_lls)),
            "cv_fold_log_loss_std": float(np.std(fold_lls)),
            "cv_oof_macro_f1": oof_f1,
            "median_best_iteration": med_iter,
            "elapsed_s": round(elapsed, 2),
        }
        for k in ["max_depth", "learning_rate", "min_child_weight", "subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda", "max_bin"]:
            rec[f"param_{k}"] = params[k]
        trial_records.append(rec)

        print(
            f"  Trial {trial.number:2d}/{args.n_trials - 1}: "
            f"CV LogLoss={oof_ll:.5f} | Macro-F1={oof_f1:.4f} | "
            f"depth={params['max_depth']}, lr={params['learning_rate']:.4f}, "
            f"mcw={params['min_child_weight']:.1f}, iters={med_iter} ({elapsed:.1f}s)"
        )
        return oof_ll

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED, n_startup_trials=8)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner, study_name="pathsnap_xgboost_tpe")

    # Seed Trial 0 with our untuned XGBoost baseline hyperparameters
    study.enqueue_trial({
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 50.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 1e-4,
        "reg_alpha": 1e-4,
        "reg_lambda": 1.0,
        "max_bin": 256,
    })

    print(f"\nStarting Optuna Bayesian Optimization ({args.n_trials} trials, 5-Fold GroupKFold)...")
    t_search0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)
    search_time_s = time.time() - t_search0

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_n_estimators = int(best_trial.user_attrs["median_best_iteration"])

    print(f"\nOptuna search completed in {search_time_s:.1f}s!")
    print(f"Best Trial #{best_trial.number}: CV OOF Log Loss = {best_trial.value:.5f} | n_estimators = {best_n_estimators}")
    print("Best Hyperparameters:", json.dumps(best_params, indent=2))

    # --- Full 5-Fold GroupKFold OOF evaluation with the best configuration ---
    print("\nRunning full 5-Fold GroupKFold evaluation for Best Tuned XGBoost...")
    oof_probs_best = np.zeros(len(train_df), dtype=np.float64)
    fold_summaries: List[Dict[str, Any]] = []

    def make_tuned_clf(n_est: int) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            n_estimators=n_est,
            learning_rate=best_params["learning_rate"],
            max_depth=best_params["max_depth"],
            min_child_weight=best_params["min_child_weight"],
            subsample=best_params["subsample"],
            colsample_bytree=best_params["colsample_bytree"],
            gamma=best_params["gamma"],
            reg_alpha=best_params["reg_alpha"],
            reg_lambda=best_params["reg_lambda"],
            max_bin=best_params["max_bin"],
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device=device,
            random_state=RANDOM_SEED,
        )

    for fold_i, (tr_fit_idx, va_idx, _, _) in enumerate(folds, 1):
        t_f0 = time.time()
        clf_f = make_tuned_clf(best_n_estimators)
        clf_f.fit(X_train[tr_fit_idx], y_train[tr_fit_idx])
        fit_s = time.time() - t_f0
        va_probs = clf_f.predict_proba(X_train[va_idx])[:, 1]
        oof_probs_best[va_idx] = va_probs

        va_df = train_df.iloc[va_idx].reset_index(drop=True)
        va_trips = set(va_df["trip_id"].unique())
        va_meta = {k: train_meta[k] for k in va_trips}
        f_eval = evaluate_predictions(va_df, va_probs, va_meta)
        fold_summaries.append({
            "fold": fold_i,
            "n_val_trips": len(va_trips),
            "macro_f1": f_eval["candidate_metrics"]["macro_f1"],
            "log_loss": f_eval["candidate_metrics"]["log_loss"],
            "top1_accuracy": f_eval["route_metrics"]["top1_accuracy"],
            "top1_accuracy_covered": f_eval["route_metrics"]["top1_accuracy_covered"],
            "route_continuity": f_eval["route_metrics"]["route_continuity"],
            "transition_continuity": f_eval["route_metrics"]["transition_continuity_trip_mean"],
            "seq_lcs_f1": f_eval["route_metrics"]["seq_lcs_f1_mean"],
            "fit_time_s": round(fit_s, 3),
        })
        print(
            f"  Fold {fold_i}: Macro-F1={fold_summaries[-1]['macro_f1']:.4f} | "
            f"LogLoss={fold_summaries[-1]['log_loss']:.4f} | "
            f"Top1-Cov={fold_summaries[-1]['top1_accuracy_covered']*100:.2f}% | "
            f"LCS-F1={fold_summaries[-1]['seq_lcs_f1']:.4f} ({fit_s:.2f}s)"
        )

    cv_oof_eval = evaluate_predictions(train_df, oof_probs_best, train_meta)

    # --- Refit final Tuned XGBoost on all 80 training trips (covered obs only) ---
    print(f"\nRefitting final Tuned XGBoost (n_estimators={best_n_estimators}) on all 80 training trips...")
    t_tr0 = time.time()
    final_clf = make_tuned_clf(best_n_estimators)
    final_clf.fit(X_train[covered_train_rows], y_train[covered_train_rows])
    full_train_s = time.time() - t_tr0

    # Switch booster device to cpu for portable FastAPI inference timing & serialization
    final_clf.set_params(device="cpu")
    final_clf.get_booster().set_param({"device": "cpu"})

    tr_probs = final_clf.predict_proba(X_train)[:, 1]
    train_in_eval = evaluate_predictions(train_df, tr_probs, train_meta)

    t_te0 = time.time()
    test_probs = final_clf.predict_proba(X_test)[:, 1]
    inf_time_s = time.time() - t_te0
    test_eval = evaluate_predictions(test_df, test_probs, test_meta)

    n_test_trips = test_eval["route_metrics"]["n_trips"]
    n_test_obs = test_eval["route_metrics"]["total_obs"]

    # Load untuned XGBoost results for direct comparison
    with open(EXPERIMENTS_DIR / "xgboost" / "results.json") as f:
        untuned_res = json.load(f)

    tuned_res: Dict[str, Any] = {
        "model_key": "xgboost_tuned",
        "model_name": "XGBoost (Optuna Tuned)",
        "algorithm_family": "Boosting",
        "search_config": {
            "strategy": "Optuna TPESampler (Bayesian Optimization)",
            "pruner": "MedianPruner(n_startup_trials=10, n_warmup_steps=2)",
            "n_trials_requested": args.n_trials,
            "n_trials_completed": len(trial_records),
            "n_trials_pruned": int(len(study.trials) - len(trial_records)),
            "objective": "5-Fold GroupKFold(trip_id) OOF Binary Log Loss",
            "search_space": SEARCH_SPACE_SPEC,
            "random_seed": RANDOM_SEED,
        },
        "best_trial": {
            "trial_number": int(best_trial.number),
            "cv_oof_log_loss_early_stop": float(best_trial.value),
            "n_estimators": best_n_estimators,
            "params": best_params,
        },
        "timings": {
            "total_search_time_s": round(search_time_s, 2),
            "full_train_s": round(full_train_s, 3),
            "student_test_inference_cpu_s": round(inf_time_s, 5),
            "ms_per_trip_cpu": round(inf_time_s * 1000.0 / n_test_trips, 3),
            "us_per_obs_cpu": round(inf_time_s * 1e6 / n_test_obs, 3),
        },
        "train_insample": {
            "candidate_metrics": train_in_eval["candidate_metrics"],
            "route_metrics": train_in_eval["route_metrics"],
        },
        "cv_5fold_oof": {
            "candidate_metrics": cv_oof_eval["candidate_metrics"],
            "route_metrics": cv_oof_eval["route_metrics"],
            "fold_summaries": fold_summaries,
            "fold_macro_f1_mean": float(np.mean([f["macro_f1"] for f in fold_summaries])),
            "fold_macro_f1_std": float(np.std([f["macro_f1"] for f in fold_summaries])),
        },
        "student_test": {
            "candidate_metrics": test_eval["candidate_metrics"],
            "route_metrics": test_eval["route_metrics"],
        },
    }

    # Save artifacts
    trials_df = pd.DataFrame(trial_records)
    trials_df.to_csv(OUT_DIR / "optuna_trials.csv", index=False)
    pd.DataFrame(test_eval["per_trip"]).to_csv(OUT_DIR / "student_test_per_trip.csv", index=False)
    np.save(OUT_DIR / "oof_probs.npy", oof_probs_best.astype(np.float32))
    np.save(OUT_DIR / "test_probs.npy", test_probs.astype(np.float32))
    joblib.dump(
        {
            "model": final_clf,
            "model_key": "xgboost_tuned",
            "display_name": "XGBoost (Optuna Tuned)",
            "features": FEATURE_COLUMNS,
            "best_params": {**best_params, "n_estimators": best_n_estimators},
            "random_seed": RANDOM_SEED,
        },
        OUT_DIR / "tuned_xgboost.joblib",
        compress=3,
    )

    # Save side-by-side comparison JSON & Markdown
    comparison_summary = {
        "untuned_xgboost": {
            "cv_macro_f1_oof": untuned_res["cv_5fold_oof"]["candidate_metrics"]["macro_f1"],
            "cv_macro_f1_fold_mean": untuned_res["cv_5fold_oof"]["fold_macro_f1_mean"],
            "cv_macro_f1_fold_std": untuned_res["cv_5fold_oof"]["fold_macro_f1_std"],
            "cv_log_loss": untuned_res["cv_5fold_oof"]["candidate_metrics"]["log_loss"],
            "cv_top1_cov": untuned_res["cv_5fold_oof"]["route_metrics"]["top1_accuracy_covered"],
            "test_macro_f1": untuned_res["student_test"]["candidate_metrics"]["macro_f1"],
            "test_top1_acc": untuned_res["student_test"]["route_metrics"]["top1_accuracy"],
            "test_top1_cov": untuned_res["student_test"]["route_metrics"]["top1_accuracy_covered"],
            "test_log_loss": untuned_res["student_test"]["candidate_metrics"]["log_loss"],
            "test_route_continuity": untuned_res["student_test"]["route_metrics"]["route_continuity"],
            "test_switch_continuity": untuned_res["student_test"]["route_metrics"]["transition_continuity_trip_mean"],
            "test_seq_lcs_f1": untuned_res["student_test"]["route_metrics"]["seq_lcs_f1_mean"],
            "ms_per_trip": untuned_res["timings"]["ms_per_trip"],
        },
        "tuned_xgboost": {
            "cv_macro_f1_oof": tuned_res["cv_5fold_oof"]["candidate_metrics"]["macro_f1"],
            "cv_macro_f1_fold_mean": tuned_res["cv_5fold_oof"]["fold_macro_f1_mean"],
            "cv_macro_f1_fold_std": tuned_res["cv_5fold_oof"]["fold_macro_f1_std"],
            "cv_log_loss": tuned_res["cv_5fold_oof"]["candidate_metrics"]["log_loss"],
            "cv_top1_cov": tuned_res["cv_5fold_oof"]["route_metrics"]["top1_accuracy_covered"],
            "test_macro_f1": tuned_res["student_test"]["candidate_metrics"]["macro_f1"],
            "test_top1_acc": tuned_res["student_test"]["route_metrics"]["top1_accuracy"],
            "test_top1_cov": tuned_res["student_test"]["route_metrics"]["top1_accuracy_covered"],
            "test_log_loss": tuned_res["student_test"]["candidate_metrics"]["log_loss"],
            "test_route_continuity": tuned_res["student_test"]["route_metrics"]["route_continuity"],
            "test_switch_continuity": tuned_res["student_test"]["route_metrics"]["transition_continuity_trip_mean"],
            "test_seq_lcs_f1": tuned_res["student_test"]["route_metrics"]["seq_lcs_f1_mean"],
            "ms_per_trip_cpu": tuned_res["timings"]["ms_per_trip_cpu"],
        },
    }
    tuned_res["untuned_vs_tuned"] = comparison_summary
    with open(OUT_DIR / "tuning_results.json", "w") as f:
        json.dump(tuned_res, f, indent=2)

    plot_tuning_results(trials_df, study, untuned_res, tuned_res)

    u = comparison_summary["untuned_xgboost"]
    t = comparison_summary["tuned_xgboost"]
    print("\n================ UNTUNED vs OPTUNA-TUNED XGBOOST ================")
    print(f"CV Macro F1 (OOF)  : {u['cv_macro_f1_oof']:.4f} -> {t['cv_macro_f1_oof']:.4f} ({t['cv_macro_f1_oof']-u['cv_macro_f1_oof']:+.4f})")
    print(f"CV OOF Log Loss    : {u['cv_log_loss']:.4f} -> {t['cv_log_loss']:.4f} ({t['cv_log_loss']-u['cv_log_loss']:+.4f})")
    print(f"CV Covered Top-1   : {u['cv_top1_cov']*100:.2f}% -> {t['cv_top1_cov']*100:.2f}% ({(t['cv_top1_cov']-u['cv_top1_cov'])*100:+.2f}%)")
    print(f"Test Macro F1      : {u['test_macro_f1']:.4f} -> {t['test_macro_f1']:.4f} ({t['test_macro_f1']-u['test_macro_f1']:+.4f})")
    print(f"Test Top-1 Acc     : {u['test_top1_acc']*100:.2f}% -> {t['test_top1_acc']*100:.2f}% ({(t['test_top1_acc']-u['test_top1_acc'])*100:+.2f}%)")
    print(f"Test Log Loss      : {u['test_log_loss']:.4f} -> {t['test_log_loss']:.4f} ({t['test_log_loss']-u['test_log_loss']:+.4f})")
    print(f"Test Step / Switch : {u['test_route_continuity']*100:.2f}% / {u['test_switch_continuity']*100:.2f}% -> {t['test_route_continuity']*100:.2f}% / {t['test_switch_continuity']*100:.2f}%")
    print(f"Test Seq LCS F1    : {u['test_seq_lcs_f1']:.4f} -> {t['test_seq_lcs_f1']:.4f} ({t['test_seq_lcs_f1']-u['test_seq_lcs_f1']:+.4f})")
    print(f"Inference (CPU)    : {t['ms_per_trip_cpu']:.2f} ms/trip | Total Search Time: {search_time_s:.1f}s")
    print(f"Saved all tuning outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
