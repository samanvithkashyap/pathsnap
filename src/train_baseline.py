"""
Baseline Model Training & Evaluation for PathSnap.

Model #1 (Baseline): Logistic Regression (Linear & Probabilistic family)
- Preprocessing: StandardScaler fit strictly inside each training fold / training set.
- Validation: 5-Fold GroupKFold grouped by trip_id across the 80 training trips (zero trip leakage).
- Training-only noise handling: Observations with zero positive candidates (off-road drift such as
  stationary parking periods identified during EDA) are excluded from model fitting ONLY; validation
  and student_test evaluation ALWAYS include 100% of GPS observations.
- Final evaluation: Refit on all 80 training trips and evaluate on the 10 student_test trips.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import folium
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import load_trip
from features import FEATURE_COLUMNS
from metrics import (
    aggregate_trip_metrics,
    evaluate_candidate_predictions,
    evaluate_trip_predictions,
    select_top1_arcs,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
FEATURES_DIR = PROJECT_ROOT / "outputs" / "features"
OUT_DIR = PROJECT_ROOT / "outputs" / "baseline"
PLOTS_DIR = OUT_DIR / "plots"
MAPS_DIR = OUT_DIR / "maps"

RANDOM_SEED = 42
N_FOLDS = 5


def build_covered_row_mask(df: pd.DataFrame, trip_meta: Dict[str, Any]) -> np.ndarray:
    """Boolean mask selecting candidate rows belonging to observations that have a positive candidate."""
    mask = np.zeros(len(df), dtype=bool)
    for tid, grp in df.groupby("trip_id", sort=False):
        cov = trip_meta[tid]["covered_mask"]
        obs_idx = grp["obs_idx"].to_numpy(dtype=np.int64)
        mask[grp.index.to_numpy()] = cov[obs_idx]
    return mask


def evaluate_partition(
    df: pd.DataFrame,
    probs: np.ndarray,
    trip_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute both candidate-level and per-trip / aggregated route metrics."""
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
    pred_arcs_by_trip: Dict[str, np.ndarray] = {}

    # Evaluate each trip independently
    for tid, grp in df.groupby("trip_id", sort=True):
        meta = trip_meta[tid]
        trip_probs = probs[grp.index.to_numpy()]
        pred_arcs = select_top1_arcs(grp, trip_probs, meta["n_obs"])
        pred_arcs_by_trip[tid] = pred_arcs
        tm = evaluate_trip_predictions(
            pred_arcs=pred_arcs,
            gt_arc_ids=meta["gt_arc_ids"],
            ref_route=meta["ref_route"],
            arcs=meta["arcs"],
            covered_mask=meta["covered_mask"],
        )
        tm["trip_id"] = tid
        per_trip.append(tm)

    agg = aggregate_trip_metrics(per_trip)
    return {
        "candidate_metrics": cand_metrics,
        "route_metrics": agg,
        "per_trip": per_trip,
        "pred_arcs_by_trip": pred_arcs_by_trip,
    }


def make_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                random_state=RANDOM_SEED,
            ),
        ),
    ])


def plot_diagnostics(
    test_eval: Dict[str, Any],
    model: Pipeline,
    out_path: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # 1. Confusion matrix on student_test
    cm = np.array(test_eval["candidate_metrics"]["confusion_matrix"])
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0 (Wrong)", "Pred 1 (Match)"],
        yticklabels=["True 0 (Wrong)", "True 1 (Match)"],
        ax=axes[0],
    )
    axes[0].set_title(
        f"Student-Test Confusion Matrix\nMacro F1 = {test_eval['candidate_metrics']['macro_f1']:.4f} | "
        f"Log Loss = {test_eval['candidate_metrics']['log_loss']:.4f}"
    )

    # 2. Top 15 standardized Logistic Regression coefficients
    coefs = model.named_steps["clf"].coef_[0]
    coef_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "coef": coefs})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    top_c = coef_df.sort_values("abs_coef", ascending=True).tail(15)
    colors = ["#dc2626" if c < 0 else "#2563eb" for c in top_c["coef"]]
    axes[1].barh(top_c["feature"], top_c["coef"], color=colors)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Top 15 Standardized Feature Weights\n(Blue = Increases Match Prob, Red = Decreases)")
    axes[1].set_xlabel("Standardized Log-Odds Coefficient")

    # 3. Per-trip Top-1 Accuracy & Transition Continuity on student_test
    pt_df = pd.DataFrame(test_eval["per_trip"]).sort_values("trip_id")
    x = np.arange(len(pt_df))
    w = 0.28
    axes[2].bar(x - w, pt_df["top1_accuracy"] * 100, width=w, label="Top-1 Accuracy (%)", color="#2563eb")
    axes[2].bar(x, pt_df["route_continuity"] * 100, width=w, label="Step Continuity (%)", color="#16a34a")
    axes[2].bar(x + w, pt_df["transition_continuity"] * 100, width=w, label="Switch Continuity (%)", color="#ea580c")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([t[-4:] for t in pt_df["trip_id"]], rotation=45)
    axes[2].set_ylim(0, 108)
    axes[2].set_ylabel("Percentage (%)")
    axes[2].set_xlabel("Student-Test Trip ID (last 4 digits)")
    axes[2].set_title("Per-Trip Baseline Performance (Classifier-Only)")
    axes[2].legend(fontsize=8, loc="lower left")

    fig.suptitle("Baseline Model (Logistic Regression) — Diagnostic Summary", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def create_prediction_folium_map(
    partition: str,
    trip_id: str,
    pred_arcs: np.ndarray,
    metrics_row: Dict[str, Any],
    out_path: Path,
) -> None:
    trip = load_trip(DATASET_ROOT / partition / trip_id, partition, load_route=True)
    lons = trip.track["lon"].to_numpy()
    lats = trip.track["lat"].to_numpy()
    u, v = trip.arcs[:, 0], trip.arcs[:, 1]
    nlon, nlat = trip.nodes[:, 0], trip.nodes[:, 1]

    m = folium.Map(location=[float(np.median(lats)), float(np.median(lons))], zoom_start=14, tiles="OpenStreetMap")

    # Layer 1: Road network subset
    road_layer = folium.FeatureGroup(name="Nearby Road Network", show=True)
    lat_min, lat_max = lats.min() - 0.003, lats.max() + 0.003
    lon_min, lon_max = lons.min() - 0.004, lons.max() + 0.004
    in_box = (nlon[u] >= lon_min) & (nlon[u] <= lon_max) & (nlat[u] >= lat_min) & (nlat[u] <= lat_max)
    arc_ids = np.where(in_box)[0]
    if len(arc_ids) > 2500:
        arc_ids = arc_ids[:: max(1, len(arc_ids) // 2500)]
    folium.PolyLine(
        [[(float(nlat[u[i]]), float(nlon[u[i]])), (float(nlat[v[i]]), float(nlon[v[i]]))] for i in arc_ids],
        color="#94a3b8",
        weight=1.3,
        opacity=0.5,
    ).add_to(road_layer)
    road_layer.add_to(m)

    # Layer 2: Reference route (green)
    ref_layer = folium.FeatureGroup(name="Reference Route (.route)", show=True)
    folium.PolyLine(
        [[(float(nlat[u[i]]), float(nlon[u[i]])), (float(nlat[v[i]]), float(nlon[v[i]]))] for i in trip.route],
        color="#16a34a",
        weight=6,
        opacity=0.65,
        tooltip="Reference Route",
    ).add_to(ref_layer)
    ref_layer.add_to(m)

    # Layer 3: Predicted route (red dashed/solid)
    pred_layer = folium.FeatureGroup(name="Baseline Predicted Route (LogReg)", show=True)
    unique_pred = np.unique(pred_arcs[pred_arcs >= 0])
    folium.PolyLine(
        [[(float(nlat[u[i]]), float(nlon[u[i]])), (float(nlat[v[i]]), float(nlon[v[i]]))] for i in unique_pred],
        color="#dc2626",
        weight=3,
        opacity=0.9,
        tooltip="Baseline Logistic Regression Prediction",
    ).add_to(pred_layer)
    pred_layer.add_to(m)

    # Layer 4: Raw GPS points
    gps_layer = folium.FeatureGroup(name="Raw GPS Observations", show=True)
    step = max(1, len(lats) // 500)
    for idx in range(0, len(lats), step):
        folium.CircleMarker(
            [float(lats[idx]), float(lons[idx])],
            radius=2.2,
            color="#2563eb",
            fill=True,
            fill_opacity=0.65,
        ).add_to(gps_layer)
    gps_layer.add_to(m)

    m.fit_bounds([[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]])
    folium.LayerControl(collapsed=False).add_to(m)

    legend_html = f"""
    <div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999; background: white;
                padding: 10px 14px; border: 2px solid #cbd5e1; border-radius: 8px; font-size: 12px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
      <b>Baseline (LogReg) — {partition}/{trip_id}</b><br>
      Top-1 Acc: <b>{metrics_row['top1_accuracy']*100:.2f}%</b> |
      Continuity: <b>{metrics_row['route_continuity']*100:.2f}%</b> |
      Switch Cont.: <b>{metrics_row['transition_continuity']*100:.2f}%</b><br>
      <span style="color:#2563eb;">●</span> Raw GPS Points &nbsp;
      <span style="color:#16a34a;">━━</span> Reference Route &nbsp;
      <span style="color:#dc2626;">━━</span> Baseline Predicted Arcs
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(str(out_path))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading cached feature datasets...")
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
        f"Train rows: {len(train_df):,} ({covered_train_rows.sum():,} in covered obs), "
        f"Student-test rows: {len(test_df):,}, Features: {len(FEATURE_COLUMNS)}"
    )

    # --- 5-Fold GroupKFold Cross-Validation on 80 training trips ---
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_probs = np.zeros(len(train_df), dtype=np.float64)
    fold_summaries: List[Dict[str, Any]] = []

    print(f"\nRunning {N_FOLDS}-Fold GroupKFold CV on 80 training trips...")
    cv_t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups_train), 1):
        # Filter training fold ONLY to covered observations (excludes off-road stationary drift)
        tr_fit_idx = tr_idx[covered_train_rows[tr_idx]]

        t_fit0 = time.time()
        pipe = make_model()
        pipe.fit(X_train[tr_fit_idx], y_train[tr_fit_idx])
        fit_s = time.time() - t_fit0

        t_inf0 = time.time()
        va_probs = pipe.predict_proba(X_train[va_idx])[:, 1]
        inf_s = time.time() - t_inf0
        oof_probs[va_idx] = va_probs

        # Evaluate this fold on 100% of its validation observations
        va_df = train_df.iloc[va_idx].copy()
        va_df.reset_index(drop=True, inplace=True)
        va_trips = set(va_df["trip_id"].unique())
        va_meta = {k: train_meta[k] for k in va_trips}
        f_eval = evaluate_partition(va_df, va_probs, va_meta)

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
            "inf_time_s": round(inf_s, 3),
        }
        fold_summaries.append(fold_rec)
        print(
            f"  Fold {fold} ({len(va_trips)} trips): "
            f"Macro-F1={fold_rec['macro_f1']:.4f}, LogLoss={fold_rec['log_loss']:.4f}, "
            f"Top1-All={fold_rec['top1_accuracy']*100:.2f}%, Top1-Cov={fold_rec['top1_accuracy_covered']*100:.2f}%, "
            f"Cont={fold_rec['route_continuity']*100:.2f}%, LCS-F1={fold_rec['seq_lcs_f1']:.4f} "
            f"({fit_s:.2f}s)"
        )

    cv_oof_eval = evaluate_partition(train_df, oof_probs, train_meta)
    cv_total_s = time.time() - cv_t0

    # --- Refit on all 80 training trips (covered obs only) and evaluate on student_test ---
    print("\nRefitting baseline LogisticRegression on all 80 training trips...")
    t_tr0 = time.time()
    final_model = make_model()
    final_model.fit(X_train[covered_train_rows], y_train[covered_train_rows])
    train_time_s = time.time() - t_tr0

    # Training-set performance (for overfitting/underfitting check)
    tr_probs = final_model.predict_proba(X_train)[:, 1]
    train_in_eval = evaluate_partition(train_df, tr_probs, train_meta)

    # Student-test inference & evaluation
    t_te0 = time.time()
    test_probs = final_model.predict_proba(X_test)[:, 1]
    inf_time_s = time.time() - t_te0
    test_eval = evaluate_partition(test_df, test_probs, test_meta)

    n_test_obs = test_eval["route_metrics"]["total_obs"]
    n_test_trips = test_eval["route_metrics"]["n_trips"]

    # Print comprehensive summary
    print("\n================ BASELINE MODEL SUMMARY (Logistic Regression) ================")
    print(f"Train (in-sample)  | Macro-F1: {train_in_eval['candidate_metrics']['macro_f1']:.4f} | "
          f"Top-1 Acc: {train_in_eval['route_metrics']['top1_accuracy']*100:.2f}% (Cov: {train_in_eval['route_metrics']['top1_accuracy_covered']*100:.2f}%) | "
          f"LogLoss: {train_in_eval['candidate_metrics']['log_loss']:.4f}")
    print(f"5-Fold CV (OOF)    | Macro-F1: {cv_oof_eval['candidate_metrics']['macro_f1']:.4f} "
          f"(fold mean: {np.mean([f['macro_f1'] for f in fold_summaries]):.4f} ± {np.std([f['macro_f1'] for f in fold_summaries]):.4f}) | "
          f"Top-1 Acc: {cv_oof_eval['route_metrics']['top1_accuracy']*100:.2f}% (Cov: {cv_oof_eval['route_metrics']['top1_accuracy_covered']*100:.2f}%) | "
          f"LogLoss: {cv_oof_eval['candidate_metrics']['log_loss']:.4f} | "
          f"Route Cont: {cv_oof_eval['route_metrics']['route_continuity']*100:.2f}% | "
          f"LCS-F1: {cv_oof_eval['route_metrics']['seq_lcs_f1_mean']:.4f}")
    print(f"Student-Test (10)  | Macro-F1: {test_eval['candidate_metrics']['macro_f1']:.4f} | "
          f"Prec: {test_eval['candidate_metrics']['precision']:.4f} | Rec: {test_eval['candidate_metrics']['recall']:.4f} | "
          f"LogLoss: {test_eval['candidate_metrics']['log_loss']:.4f}")
    print(f"                   | Top-1 Acc: {test_eval['route_metrics']['top1_accuracy']*100:.2f}% "
          f"(Cov: {test_eval['route_metrics']['top1_accuracy_covered']*100:.2f}%, Miss Rate: {test_eval['route_metrics']['candidate_miss_rate']*100:.2f}%)")
    print(f"                   | Route Cont: {test_eval['route_metrics']['route_continuity']*100:.2f}% "
          f"(Switch Cont: {test_eval['route_metrics']['transition_continuity_trip_mean']*100:.2f}%) | "
          f"Seq Arc F1: {test_eval['route_metrics']['seq_arc_f1_mean']:.4f} | "
          f"Seq LCS F1: {test_eval['route_metrics']['seq_lcs_f1_mean']:.4f}")
    print(f"Timing             | Train time: {train_time_s:.3f}s | Student-test inference: {inf_time_s*1000:.2f}ms "
          f"({inf_time_s*1000/n_test_trips:.2f} ms/trip, {inf_time_s*1e6/n_test_obs:.2f} us/obs)")

    print("\n=== Student-Test Per-Trip Breakdown ===")
    pt_df = pd.DataFrame(test_eval["per_trip"])
    disp_cols = [
        "trip_id", "n_obs", "candidate_miss_rate", "top1_accuracy",
        "top1_accuracy_covered", "route_continuity", "transition_continuity",
        "seq_arc_f1", "seq_lcs_f1",
    ]
    print(pt_df[disp_cols].to_string(index=False))

    # Save artifacts
    pt_df.to_csv(OUT_DIR / "student_test_per_trip.csv", index=False)
    joblib.dump(
        {"model": final_model, "features": FEATURE_COLUMNS, "random_seed": RANDOM_SEED},
        OUT_DIR / "baseline_logreg.joblib",
    )

    coefs = dict(zip(FEATURE_COLUMNS, [round(float(c), 4) for c in final_model.named_steps["clf"].coef_[0]]))
    results_payload = {
        "model_name": "Logistic Regression (Baseline)",
        "algorithm_family": "Linear and probabilistic",
        "config": {
            "scaler": "StandardScaler",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "class_weight": None,
            "random_state": RANDOM_SEED,
            "n_folds": N_FOLDS,
            "cv_strategy": "GroupKFold(trip_id)",
            "training_filter": "covered_observations_only",
            "n_features": len(FEATURE_COLUMNS),
        },
        "timings": {
            "cv_total_s": round(cv_total_s, 3),
            "full_train_s": round(train_time_s, 3),
            "student_test_inference_s": round(inf_time_s, 5),
            "ms_per_trip": round(inf_time_s * 1000.0 / n_test_trips, 3),
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
            "fold_top1_mean": float(np.mean([f["top1_accuracy"] for f in fold_summaries])),
            "fold_top1_std": float(np.std([f["top1_accuracy"] for f in fold_summaries])),
        },
        "student_test": {
            "candidate_metrics": test_eval["candidate_metrics"],
            "route_metrics": test_eval["route_metrics"],
        },
        "coefficients": coefs,
    }

    with open(OUT_DIR / "baseline_results.json", "w") as f:
        json.dump(results_payload, f, indent=2)

    # Generate diagnostic plots & Folium prediction maps
    plot_diagnostics(test_eval, final_model, PLOTS_DIR / "baseline_diagnostics.png")
    pt_lookup = {row["trip_id"]: row for row in test_eval["per_trip"]}
    for tid in ["00000000", "00000056"]:
        create_prediction_folium_map(
            "student_test",
            tid,
            test_eval["pred_arcs_by_trip"][tid],
            pt_lookup[tid],
            MAPS_DIR / f"baseline_pred_{tid}.html",
        )

    print(f"\nSaved all baseline results, model, plots, and Folium maps to {OUT_DIR}")


if __name__ == "__main__":
    main()
