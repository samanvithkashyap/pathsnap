"""
Modular Multi-Model Experiment Runner for PathSnap.

Runs any or all models registered in src/models/registry.py through the exact same:
- Feature matrix & 33 engineered features (loaded from outputs/features/)
- 5-Fold GroupKFold CV grouped strictly by trip_id across the 80 training trips
- Covered-observation training filter (applied to training folds ONLY; validation & test always 100%)
- Refit on all 80 training trips + evaluation on the 10 student_test trips
- Standardized metric computation, timing logs, saved models, and probability predictions

Usage:
    python src/experiment_runner.py --all
    python src/experiment_runner.py --models lightgbm mlp
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# Ensure src/ is importable regardless of working directory
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features import FEATURE_COLUMNS
from metrics import (
    aggregate_trip_metrics,
    evaluate_candidate_predictions,
    evaluate_trip_predictions,
    select_top1_arcs,
)
from models import MODEL_KEYS, ModelSpec, get_model_registry

PROJECT_ROOT = SRC_DIR.parent
FEATURES_DIR = PROJECT_ROOT / "outputs" / "features"
EXPERIMENTS_DIR = PROJECT_ROOT / "outputs" / "experiments"

N_FOLDS = 5
RANDOM_SEED = 42


def build_covered_row_mask(df: pd.DataFrame, trip_meta: Dict[str, Any]) -> np.ndarray:
    """Boolean mask for candidate rows belonging to covered GPS observations."""
    mask = np.zeros(len(df), dtype=bool)
    for tid, grp in df.groupby("trip_id", sort=False):
        cov = trip_meta[tid]["covered_mask"]
        obs_idx = grp["obs_idx"].to_numpy(dtype=np.int64)
        mask[grp.index.to_numpy()] = cov[obs_idx]
    return mask


def evaluate_predictions(
    df: pd.DataFrame,
    probs: np.ndarray,
    trip_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute candidate-level and per-trip/aggregated route metrics."""
    y_true = df["label"].to_numpy(dtype=np.int8)
    trip_codes = pd.Categorical(df["trip_id"]).codes
    obs_idx = df["obs_idx"].to_numpy(dtype=np.int64)

    cand_metrics = evaluate_candidate_predictions(
        y_true=y_true,
        y_prob=probs,
        threshold=0.5,
        obs_groups=(trip_codes, obs_idx),
    )

    per_trip: List[Dict[str, Any]] = []
    for tid, grp in df.groupby("trip_id", sort=True):
        meta = trip_meta[tid]
        trip_probs = probs[grp.index.to_numpy()]
        pred_arcs = select_top1_arcs(grp, trip_probs, meta["n_obs"])
        tm = evaluate_trip_predictions(
            pred_arcs=pred_arcs,
            gt_arc_ids=meta["gt_arc_ids"],
            ref_route=meta["ref_route"],
            arcs=meta["arcs"],
            covered_mask=meta["covered_mask"],
        )
        tm["trip_id"] = tid
        per_trip.append(tm)

    return {
        "candidate_metrics": cand_metrics,
        "route_metrics": aggregate_trip_metrics(per_trip),
        "per_trip": per_trip,
    }


def run_single_model(
    spec: ModelSpec,
    train_df: pd.DataFrame,
    train_meta: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    covered_train_rows: np.ndarray,
    test_df: pd.DataFrame,
    test_meta: Dict[str, Any],
    X_test: np.ndarray,
) -> Dict[str, Any]:
    """Execute 5-fold GroupKFold CV + full-train refit + student_test evaluation for one model."""
    out_dir = EXPERIMENTS_DIR / spec.key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 78}")
    print(f"MODEL: {spec.display_name} [{spec.key}] | Family: {spec.algorithm_family}")
    print(f"Config: {spec.key_config_summary}")
    print(f"{'=' * 78}")

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_probs = np.zeros(len(train_df), dtype=np.float64)
    fold_summaries: List[Dict[str, Any]] = []
    fold_histories: List[Dict[str, Any]] = []
    best_epochs: List[int] = []

    cv_t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups_train), 1):
        tr_fit_idx = tr_idx[covered_train_rows[tr_idx]]
        model = spec.builder()

        t_fit0 = time.time()
        if spec.supports_val_history:
            model.fit(
                X_train[tr_fit_idx],
                y_train[tr_fit_idx],
                X_val=X_train[va_idx],
                y_val=y_train[va_idx],
            )
            fold_histories.append({
                "fold": fold,
                "best_epoch": int(model.best_epoch_),
                "history": model.history_,
            })
            best_epochs.append(int(model.best_epoch_))
        else:
            model.fit(X_train[tr_fit_idx], y_train[tr_fit_idx])
        fit_s = time.time() - t_fit0

        t_inf0 = time.time()
        va_probs = model.predict_proba(X_train[va_idx])[:, 1]
        inf_s = time.time() - t_inf0
        oof_probs[va_idx] = va_probs

        va_df = train_df.iloc[va_idx].reset_index(drop=True)
        va_trips = set(va_df["trip_id"].unique())
        va_meta = {k: train_meta[k] for k in va_trips}
        f_eval = evaluate_predictions(va_df, va_probs, va_meta)

        fold_rec = {
            "fold": fold,
            "n_val_trips": len(va_trips),
            "n_val_obs": f_eval["route_metrics"]["total_obs"],
            "macro_f1": f_eval["candidate_metrics"]["macro_f1"],
            "top1_candidate_macro_f1": f_eval["candidate_metrics"]["top1_candidate_macro_f1"],
            "log_loss": f_eval["candidate_metrics"]["log_loss"],
            "top1_accuracy": f_eval["route_metrics"]["top1_accuracy"],
            "top1_accuracy_covered": f_eval["route_metrics"]["top1_accuracy_covered"],
            "candidate_miss_rate": f_eval["route_metrics"]["candidate_miss_rate"],
            "route_continuity": f_eval["route_metrics"]["route_continuity"],
            "transition_continuity": f_eval["route_metrics"]["transition_continuity_trip_mean"],
            "seq_lcs_f1": f_eval["route_metrics"]["seq_lcs_f1_mean"],
            "fit_time_s": round(fit_s, 3),
            "inf_time_s": round(inf_s, 4),
        }
        if spec.supports_val_history:
            fold_rec["best_epoch"] = int(model.best_epoch_)
        fold_summaries.append(fold_rec)

        ep_str = f" [best_ep={model.best_epoch_}]" if spec.supports_val_history else ""
        print(
            f"  Fold {fold} ({len(va_trips):2d} trips): "
            f"Macro-F1={fold_rec['macro_f1']:.4f} | LogLoss={fold_rec['log_loss']:.4f} | "
            f"Top1-All={fold_rec['top1_accuracy']*100:.2f}% (Cov {fold_rec['top1_accuracy_covered']*100:.2f}%) | "
            f"Cont={fold_rec['route_continuity']*100:.2f}% | LCS-F1={fold_rec['seq_lcs_f1']:.4f} "
            f"({fit_s:.2f}s){ep_str}"
        )

    cv_oof_eval = evaluate_predictions(train_df, oof_probs, train_meta)
    cv_total_s = time.time() - cv_t0

    # --- Refit on all 80 training trips (covered obs only) ---
    print(f"  Refitting {spec.display_name} on all 80 training trips...")
    final_model = spec.builder()
    t_tr0 = time.time()
    if spec.supports_val_history:
        refit_epochs = max(1, int(round(float(np.median(best_epochs)))))
        final_model.fit(X_train[covered_train_rows], y_train[covered_train_rows], fixed_epochs=refit_epochs)
    else:
        refit_epochs = None
        final_model.fit(X_train[covered_train_rows], y_train[covered_train_rows])
    full_train_s = time.time() - t_tr0

    # Training in-sample evaluation (for overfitting / underfitting diagnostics)
    tr_probs = final_model.predict_proba(X_train)[:, 1]
    train_in_eval = evaluate_predictions(train_df, tr_probs, train_meta)

    # Student-test evaluation & inference timing
    t_te0 = time.time()
    test_probs = final_model.predict_proba(X_test)[:, 1]
    inf_time_s = time.time() - t_te0
    test_eval = evaluate_predictions(test_df, test_probs, test_meta)

    n_test_trips = test_eval["route_metrics"]["n_trips"]
    n_test_obs = test_eval["route_metrics"]["total_obs"]

    print(
        f"  --> Train In-Sample | Macro-F1={train_in_eval['candidate_metrics']['macro_f1']:.4f} | "
        f"Top1-Cov={train_in_eval['route_metrics']['top1_accuracy_covered']*100:.2f}% | "
        f"LogLoss={train_in_eval['candidate_metrics']['log_loss']:.4f}"
    )
    print(
        f"  --> 5-Fold CV (OOF) | Macro-F1={cv_oof_eval['candidate_metrics']['macro_f1']:.4f} "
        f"(fold {np.mean([f['macro_f1'] for f in fold_summaries]):.4f} ± {np.std([f['macro_f1'] for f in fold_summaries]):.4f}) | "
        f"Top1-Cov={cv_oof_eval['route_metrics']['top1_accuracy_covered']*100:.2f}% | "
        f"LogLoss={cv_oof_eval['candidate_metrics']['log_loss']:.4f}"
    )
    print(
        f"  --> Student-Test    | Macro-F1={test_eval['candidate_metrics']['macro_f1']:.4f} | "
        f"Top1-All={test_eval['route_metrics']['top1_accuracy']*100:.2f}% "
        f"(Cov {test_eval['route_metrics']['top1_accuracy_covered']*100:.2f}%) | "
        f"LogLoss={test_eval['candidate_metrics']['log_loss']:.4f} | "
        f"Cont={test_eval['route_metrics']['route_continuity']*100:.2f}% "
        f"(Switch {test_eval['route_metrics']['transition_continuity_trip_mean']*100:.2f}%) | "
        f"LCS-F1={test_eval['route_metrics']['seq_lcs_f1_mean']:.4f}"
    )
    print(
        f"  --> Timing          | Train={full_train_s:.2f}s | Test Inf={inf_time_s*1000:.2f}ms "
        f"({inf_time_s*1000/n_test_trips:.2f} ms/trip)"
    )

    # Save artifacts
    pd.DataFrame(test_eval["per_trip"]).to_csv(out_dir / "student_test_per_trip.csv", index=False)
    np.save(out_dir / "oof_probs.npy", oof_probs.astype(np.float32))
    np.save(out_dir / "test_probs.npy", test_probs.astype(np.float32))
    joblib.dump(
        {
            "model": final_model,
            "model_key": spec.key,
            "display_name": spec.display_name,
            "features": FEATURE_COLUMNS,
            "random_seed": RANDOM_SEED,
        },
        out_dir / "model.joblib",
        compress=3,
    )

    if spec.supports_val_history:
        with open(out_dir / "training_history.json", "w") as f:
            json.dump(
                {
                    "fold_histories": fold_histories,
                    "median_best_epoch": refit_epochs,
                    "full_refit_train_loss": final_model.history_["train_loss"],
                },
                f,
                indent=2,
            )

    results_payload: Dict[str, Any] = {
        "model_key": spec.key,
        "model_name": spec.display_name,
        "algorithm_family": spec.algorithm_family,
        "key_config_summary": spec.key_config_summary,
        "loss_objective": spec.loss_objective,
        "hyperparams": spec.hyperparams,
        "cv_config": {
            "n_folds": N_FOLDS,
            "cv_strategy": "GroupKFold(trip_id)",
            "training_filter": "covered_observations_only",
            "n_features": len(FEATURE_COLUMNS),
            "random_seed": RANDOM_SEED,
            "refit_epochs": refit_epochs,
        },
        "timings": {
            "cv_total_s": round(cv_total_s, 3),
            "full_train_s": round(full_train_s, 3),
            "student_test_inference_s": round(inf_time_s, 5),
            "ms_per_trip": round(inf_time_s * 1000.0 / n_test_trips, 3),
            "us_per_obs": round(inf_time_s * 1e6 / n_test_obs, 3),
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
            "fold_log_loss_mean": float(np.mean([f["log_loss"] for f in fold_summaries])),
            "fold_log_loss_std": float(np.std([f["log_loss"] for f in fold_summaries])),
            "fold_top1_cov_mean": float(np.mean([f["top1_accuracy_covered"] for f in fold_summaries])),
            "fold_top1_cov_std": float(np.std([f["top1_accuracy_covered"] for f in fold_summaries])),
        },
        "student_test": {
            "candidate_metrics": test_eval["candidate_metrics"],
            "route_metrics": test_eval["route_metrics"],
        },
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results_payload, f, indent=2)

    return results_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="PathSnap Modular Multi-Model Experiment Runner")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_KEYS,
        default=None,
        help="Specific model keys to run",
    )
    parser.add_argument("--all", action="store_true", help="Run all registered models")
    args = parser.parse_args()

    selected = MODEL_KEYS if (args.all or not args.models) else args.models
    registry = get_model_registry(RANDOM_SEED)

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading cached candidate datasets from {FEATURES_DIR}...")
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

    print(
        f"Loaded train ({len(train_df):,} rows, {covered_train_rows.sum():,} covered) & "
        f"student_test ({len(test_df):,} rows) | Running models: {selected}"
    )

    t_all = time.time()
    for key in selected:
        run_single_model(
            spec=registry[key],
            train_df=train_df,
            train_meta=train_meta,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            covered_train_rows=covered_train_rows,
            test_df=test_df,
            test_meta=test_meta,
            X_test=X_test,
        )

    print(f"\nCompleted {len(selected)} model experiment(s) in {time.time() - t_all:.1f}s.")


if __name__ == "__main__":
    main()
