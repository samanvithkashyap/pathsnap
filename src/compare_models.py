"""
Model Comparison Tables, Compulsory Visualizations, and Folium Maps for PathSnap.

Reads standardized outputs from outputs/experiments/<model_key>/ and produces:
- outputs/experiments/model_comparison.csv
- outputs/experiments/model_comparison.md
- outputs/experiments/plots/01_cv_macro_f1_comparison.png
- outputs/experiments/plots/02_student_test_comparison.png
- outputs/experiments/plots/03_log_loss_and_efficiency.png
- outputs/experiments/plots/04_confusion_matrices.png
- outputs/experiments/plots/05_mlp_training_curves.png  (real epoch history only!)
- outputs/experiments/plots/06_overfitting_check.png
- outputs/experiments/plots/07_per_trip_top1_heatmap.png
- outputs/experiments/maps/best_model_pred_<trip_id>.html
"""

from __future__ import annotations

import json
import sys
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

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import load_trip
from metrics import select_top1_arcs
from models import MODEL_KEYS

PROJECT_ROOT = SRC_DIR.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
FEATURES_DIR = PROJECT_ROOT / "outputs" / "features"
EXPERIMENTS_DIR = PROJECT_ROOT / "outputs" / "experiments"
PLOTS_DIR = EXPERIMENTS_DIR / "plots"
MAPS_DIR = EXPERIMENTS_DIR / "maps"

PALETTE = ["#64748b", "#d97706", "#059669", "#2563eb", "#7c3aed"]


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10,
    })


def load_all_results() -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for key in MODEL_KEYS:
        p = EXPERIMENTS_DIR / key / "results.json"
        if p.exists():
            with open(p) as f:
                results.append(json.load(f))
    if not results:
        raise FileNotFoundError(f"No model results found in {EXPERIMENTS_DIR}")
    return results


def build_comparison_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in results:
        tr_c = r["train_insample"]["candidate_metrics"]
        tr_r = r["train_insample"]["route_metrics"]
        cv_c = r["cv_5fold_oof"]["candidate_metrics"]
        cv_r = r["cv_5fold_oof"]["route_metrics"]
        te_c = r["student_test"]["candidate_metrics"]
        te_r = r["student_test"]["route_metrics"]
        tm = r["timings"]

        rows.append({
            "model_key": r["model_key"],
            "model_name": r["model_name"],
            "algorithm_family": r["algorithm_family"],
            "key_config": r["key_config_summary"],
            "train_macro_f1": tr_c["macro_f1"],
            "train_top1_cov": tr_r["top1_accuracy_covered"],
            "train_log_loss": tr_c["log_loss"],
            "cv_macro_f1_oof": cv_c["macro_f1"],
            "cv_macro_f1_fold_mean": r["cv_5fold_oof"]["fold_macro_f1_mean"],
            "cv_macro_f1_fold_std": r["cv_5fold_oof"]["fold_macro_f1_std"],
            "cv_top1_all": cv_r["top1_accuracy"],
            "cv_top1_cov": cv_r["top1_accuracy_covered"],
            "cv_log_loss": cv_c["log_loss"],
            "cv_route_continuity": cv_r["route_continuity"],
            "test_macro_f1": te_c["macro_f1"],
            "test_precision": te_c["precision"],
            "test_recall": te_c["recall"],
            "test_positive_f1": te_c["positive_f1"],
            "test_candidate_acc": te_c["candidate_accuracy"],
            "test_log_loss": te_c["log_loss"],
            "test_top1_acc": te_r["top1_accuracy"],
            "test_top1_cov": te_r["top1_accuracy_covered"],
            "test_miss_rate": te_r["candidate_miss_rate"],
            "test_route_continuity": te_r["route_continuity"],
            "test_switch_continuity": te_r["transition_continuity_trip_mean"],
            "test_seq_arc_f1": te_r["seq_arc_f1_mean"],
            "test_seq_lcs_f1": te_r["seq_lcs_f1_mean"],
            "train_time_s": tm["full_train_s"],
            "test_inf_ms": round(tm["student_test_inference_s"] * 1000.0, 2),
            "ms_per_trip": tm["ms_per_trip"],
        })
    return pd.DataFrame(rows)


def write_markdown_table(df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# PathSnap — Consolidated 5-Model Comparison (Classifier-Only, Pre-HMM)",
        "",
        "| Model | Algorithm Family | Key Configuration | CV Macro F1 (OOF / Fold Mean±SD) | Student-Test Top-1 Acc | Route Continuity (Step / Switch) | Student-Test Macro F1 | Log Loss (CV / Test) | Train Time | Test Inference (10 trips) |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| **{row['model_name']}** | {row['algorithm_family']} | {row['key_config']} "
            f"| {row['cv_macro_f1_oof']:.4f} ({row['cv_macro_f1_fold_mean']:.4f}±{row['cv_macro_f1_fold_std']:.4f}) "
            f"| {row['test_top1_acc']*100:.2f}% "
            f"| {row['test_route_continuity']*100:.2f}% / {row['test_switch_continuity']*100:.2f}% "
            f"| {row['test_macro_f1']:.4f} "
            f"| {row['cv_log_loss']:.4f} / {row['test_log_loss']:.4f} "
            f"| {row['train_time_s']:.2f}s "
            f"| {row['test_inf_ms']:.1f} ms ({row['ms_per_trip']:.2f} ms/trip) |"
        )
    lines.append("")
    lines.append(
        "*Candidate Miss Rate is identical across all models (`0.16%` on `student_test`, `2.48%` trip-mean on `train`) "
        "because all models share the exact same candidate generation (`STRtree`, `radius=60m`, `max_k=16`).*"
    )
    out_path.write_text("\n".join(lines) + "\n")


def plot_cv_macro_f1(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = np.arange(len(df))
    bars = ax.bar(
        x,
        df["cv_macro_f1_fold_mean"],
        yerr=df["cv_macro_f1_fold_std"],
        capsize=6,
        color=PALETTE[: len(df)],
        alpha=0.9,
        edgecolor="#1e293b",
        linewidth=0.8,
        error_kw={"elinewidth": 1.5, "ecolor": "#0f172a"},
    )
    # Overlay OOF pooled Macro F1 as diamond markers
    ax.scatter(x, df["cv_macro_f1_oof"], color="#0f172a", marker="D", s=50, zorder=5, label="Pooled OOF Macro F1")

    for i, (bar, mean_v, std_v, oof_v) in enumerate(
        zip(bars, df["cv_macro_f1_fold_mean"], df["cv_macro_f1_fold_std"], df["cv_macro_f1_oof"])
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean_v + std_v + 0.004,
            f"OOF: {oof_v:.4f}\n({mean_v:.4f}±{std_v:.4f})",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ymin = max(0.80, float(df["cv_macro_f1_fold_mean"].min() - 0.05))
    ax.set_ylim(ymin, min(1.01, float(df["cv_macro_f1_oof"].max() + 0.035)))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({f})" for n, f in zip(df["model_name"], df["algorithm_family"])], fontsize=9)
    ax.set_ylabel("Macro F1-score")
    ax.set_title("5-Fold GroupKFold Cross-Validation Macro F1-Score Across 5 Algorithm Families", fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_cv_macro_f1_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_student_test_comparison(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    x = np.arange(len(df))
    w = 0.26

    # Left: Top-1 Accuracy, Macro F1, Precision/Recall F1
    b1 = axes[0].bar(x - w, df["test_top1_acc"] * 100, width=w, label="Top-1 Accuracy (%)", color="#2563eb")
    b2 = axes[0].bar(x, df["test_macro_f1"] * 100, width=w, label="Macro F1 (×100)", color="#059669")
    b3 = axes[0].bar(x + w, df["test_positive_f1"] * 100, width=w, label="Positive-Class F1 (×100)", color="#7c3aed")
    for bars in (b1, b2, b3):
        for b in bars:
            h = b.get_height()
            axes[0].text(b.get_x() + b.get_width() / 2, h + 0.4, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)
    axes[0].set_ylim(80, 101)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["model_name"], rotation=15, ha="right")
    axes[0].set_ylabel("Score (%)")
    axes[0].set_title("Student-Test Classification & Observation-Level Metrics")
    axes[0].legend(fontsize=8.5, loc="lower right")

    # Right: Route Continuity (Step vs Switch) & Sequence LCS F1
    r1 = axes[1].bar(x - w, df["test_route_continuity"] * 100, width=w, label="Step Continuity (%)", color="#0284c7")
    r2 = axes[1].bar(x, df["test_switch_continuity"] * 100, width=w, label="Arc-Switch Continuity (%)", color="#ea580c")
    r3 = axes[1].bar(x + w, df["test_seq_lcs_f1"] * 100, width=w, label="Sequence LCS F1 (×100)", color="#16a34a")
    for bars in (r1, r2, r3):
        for b in bars:
            h = b.get_height()
            axes[1].text(b.get_x() + b.get_width() / 2, h + 0.6, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)
    axes[1].set_ylim(65, 103)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["model_name"], rotation=15, ha="right")
    axes[1].set_ylabel("Score (%)")
    axes[1].set_title("Student-Test Route Continuity & Sequence-Level Metrics")
    axes[1].legend(fontsize=8.5, loc="lower right")

    fig.suptitle("Student-Test (10 Held-Out Trips) Performance Comparison Across All 5 Models", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_student_test_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_log_loss_and_efficiency(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    x = np.arange(len(df))
    w = 0.35

    # Left: CV vs Student-Test Log Loss (lower is better!)
    b1 = axes[0].bar(x - w / 2, df["cv_log_loss"], width=w, label="5-Fold CV OOF Log Loss", color="#3b82f6")
    b2 = axes[0].bar(x + w / 2, df["test_log_loss"], width=w, label="Student-Test Log Loss", color="#f97316")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            axes[0].text(b.get_x() + b.get_width() / 2, h + 0.0015, f"{h:.4f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["model_name"], rotation=15, ha="right")
    axes[0].set_ylabel("Binary Log Loss (Lower is Better)")
    axes[0].set_title("Probability Calibration Quality (Log Loss)")
    axes[0].set_ylim(0, float(max(df["cv_log_loss"].max(), df["test_log_loss"].max()) * 1.2))
    axes[0].legend(fontsize=8.5)

    # Right: Inference Latency (ms/trip) vs Student-Test Top-1 Accuracy
    for i, row in df.iterrows():
        axes[1].scatter(
            row["ms_per_trip"],
            row["test_top1_acc"] * 100,
            s=140,
            color=PALETTE[i % len(PALETTE)],
            edgecolors="#0f172a",
            linewidth=1.0,
            zorder=4,
            label=f"{row['model_name']} ({row['train_time_s']:.1f}s train)",
        )
        axes[1].annotate(
            row["model_name"],
            (row["ms_per_trip"], row["test_top1_acc"] * 100),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
        )
    axes[1].set_xscale("log")
    axes[1].set_xlim(left=0.8, right=float(df["ms_per_trip"].max()) * 2.2)
    axes[1].set_xlabel("Student-Test Inference Latency (ms / trip, log scale)")
    axes[1].set_ylabel("Student-Test Top-1 Accuracy (%)")
    axes[1].set_title("Accuracy vs Inference Efficiency Trade-Off")
    axes[1].legend(fontsize=8, loc="lower right")

    fig.suptitle("Probability Quality (Log Loss) & Computational Efficiency", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_log_loss_and_efficiency.png", bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(results: List[Dict[str, Any]]) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.8))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        cm = np.array(r["student_test"]["candidate_metrics"]["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt=",d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
            ax=ax,
            annot_kws={"size": 9.5},
        )
        mf1 = r["student_test"]["candidate_metrics"]["macro_f1"]
        ll = r["student_test"]["candidate_metrics"]["log_loss"]
        ax.set_title(f"{r['model_name']}\nMacro F1={mf1:.4f} | LL={ll:.4f}", fontsize=10)

    fig.suptitle("Student-Test Confusion Matrices Across All 5 Models (191,384 Candidates)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)


def plot_mlp_training_curves() -> None:
    hist_path = EXPERIMENTS_DIR / "mlp" / "training_history.json"
    if not hist_path.exists():
        return
    with open(hist_path) as f:
        data = json.load(f)

    fold_hists = data["fold_histories"]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))

    for fh in fold_hists:
        ep = fh["history"]["epoch"]
        tr = fh["history"]["train_loss"]
        va = fh["history"]["val_loss"]
        fold = fh["fold"]
        best_ep = fh["best_epoch"]
        axes[0].plot(ep, tr, alpha=0.8, linewidth=1.6, label=f"Fold {fold} Train")
        axes[1].plot(ep, va, alpha=0.85, linewidth=1.8, label=f"Fold {fold} Val (best ep={best_ep})")
        axes[1].scatter([best_ep], [va[best_ep - 1]], s=45, zorder=5)

    axes[0].set_title("PyTorch MLP — Training Loss per Epoch (5-Fold GroupKFold)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].legend(fontsize=8.5)

    axes[1].set_title("PyTorch MLP — Validation Loss per Epoch (Early Stopping Check)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Binary Cross-Entropy Loss")
    axes[1].legend(fontsize=8.5)

    fig.suptitle(
        f"PyTorch MLP Real Epoch-Wise Training & Validation Loss Curves (Median Best Epoch = {data['median_best_epoch']})",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "05_mlp_training_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_overfitting_check(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0))
    x = np.arange(len(df))
    w = 0.26

    # Left: Macro F1 across Train / CV OOF / Student-Test
    axes[0].bar(x - w, df["train_macro_f1"], width=w, label="Train (In-Sample)", color="#94a3b8")
    axes[0].bar(x, df["cv_macro_f1_oof"], width=w, label="5-Fold CV (OOF)", color="#2563eb")
    axes[0].bar(x + w, df["test_macro_f1"], width=w, label="Student-Test (10 trips)", color="#16a34a")
    axes[0].set_ylim(0.88, 0.995)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["model_name"], rotation=15, ha="right")
    axes[0].set_ylabel("Macro F1-score")
    axes[0].set_title("Generalization & Overfitting Check — Macro F1")
    axes[0].legend(fontsize=8.5)

    # Right: Covered Top-1 Accuracy across Train / CV OOF / Student-Test
    axes[1].bar(x - w, df["train_top1_cov"] * 100, width=w, label="Train (In-Sample)", color="#94a3b8")
    axes[1].bar(x, df["cv_top1_cov"] * 100, width=w, label="5-Fold CV (OOF)", color="#2563eb")
    axes[1].bar(x + w, df["test_top1_cov"] * 100, width=w, label="Student-Test (10 trips)", color="#16a34a")
    axes[1].set_ylim(82, 100)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["model_name"], rotation=15, ha="right")
    axes[1].set_ylabel("Covered Observation Top-1 Accuracy (%)")
    axes[1].set_title("Generalization & Overfitting Check — Covered Top-1 Accuracy")
    axes[1].legend(fontsize=8.5)

    fig.suptitle("Train vs 5-Fold CV vs Student-Test Generalization Analysis", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "06_overfitting_check.png", bbox_inches="tight")
    plt.close(fig)


def plot_per_trip_heatmap(results: List[Dict[str, Any]]) -> None:
    mat: Dict[str, pd.Series] = {}
    for r in results:
        pt = pd.read_csv(EXPERIMENTS_DIR / r["model_key"] / "student_test_per_trip.csv")
        pt["trip_id"] = pt["trip_id"].astype(str).str.zfill(8)
        pt = pt.set_index("trip_id")
        mat[r["model_name"]] = pt["top1_accuracy"] * 100.0
    hm = pd.DataFrame(mat)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    sns.heatmap(hm, annot=True, fmt=".2f", cmap="YlGnBu", vmin=70, vmax=98, linewidths=0.5, ax=ax)
    ax.set_title("Student-Test Per-Trip Top-1 Accuracy (%) Across All 5 Models", fontweight="bold")
    ax.set_ylabel("Student-Test Trip ID")
    ax.set_xlabel("Model")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "07_per_trip_top1_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def generate_best_model_maps(best_key: str, best_name: str) -> None:
    test_pack = joblib.load(FEATURES_DIR / "student_test_candidates.joblib")
    test_df: pd.DataFrame = test_pack["df"]
    test_meta: Dict[str, Any] = test_pack["trip_meta"]
    probs = np.load(EXPERIMENTS_DIR / best_key / "test_probs.npy")
    pt = pd.read_csv(EXPERIMENTS_DIR / best_key / "student_test_per_trip.csv")
    pt["trip_id"] = pt["trip_id"].astype(str).str.zfill(8)
    pt_map = pt.set_index("trip_id").to_dict(orient="index")

    for tid in ["00000000", "00000056", "00000077"]:
        grp = test_df[test_df["trip_id"] == tid]
        trip_probs = probs[grp.index.to_numpy()]
        pred_arcs = select_top1_arcs(grp, trip_probs, test_meta[tid]["n_obs"])
        trip = load_trip(DATASET_ROOT / "student_test" / tid, "student_test", load_route=True)

        lons = trip.track["lon"].to_numpy()
        lats = trip.track["lat"].to_numpy()
        u, v = trip.arcs[:, 0], trip.arcs[:, 1]
        nlon, nlat = trip.nodes[:, 0], trip.nodes[:, 1]

        m = folium.Map(location=[float(np.median(lats)), float(np.median(lons))], zoom_start=14, tiles="OpenStreetMap")

        # Road network subset
        road_layer = folium.FeatureGroup(name="Nearby Road Network", show=True)
        in_box = (
            (nlon[u] >= lons.min() - 0.004) & (nlon[u] <= lons.max() + 0.004) &
            (nlat[u] >= lats.min() - 0.003) & (nlat[u] <= lats.max() + 0.003)
        )
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

        # Reference route
        ref_layer = folium.FeatureGroup(name="Reference Route (.route)", show=True)
        folium.PolyLine(
            [[(float(nlat[u[i]]), float(nlon[u[i]])), (float(nlat[v[i]]), float(nlon[v[i]]))] for i in trip.route],
            color="#16a34a",
            weight=6,
            opacity=0.65,
        ).add_to(ref_layer)
        ref_layer.add_to(m)

        # Best model classifier-only predictions
        pred_layer = folium.FeatureGroup(name=f"Predicted Route ({best_name}, Pre-HMM)", show=True)
        uniq_pred = np.unique(pred_arcs[pred_arcs >= 0])
        folium.PolyLine(
            [[(float(nlat[u[i]]), float(nlon[u[i]])), (float(nlat[v[i]]), float(nlon[v[i]]))] for i in uniq_pred],
            color="#dc2626",
            weight=3,
            opacity=0.9,
        ).add_to(pred_layer)
        pred_layer.add_to(m)

        # Raw GPS observations
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

        row = pt_map[tid]
        legend_html = f"""
        <div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999; background: white;
                    padding: 10px 14px; border: 2px solid #cbd5e1; border-radius: 8px; font-size: 12px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
          <b>{best_name} (Classifier-Only) — student_test/{tid}</b><br>
          Top-1 Acc: <b>{row['top1_accuracy']*100:.2f}%</b> |
          Step Cont.: <b>{row['route_continuity']*100:.2f}%</b> |
          Switch Cont.: <b>{row['transition_continuity']*100:.2f}%</b> |
          LCS F1: <b>{row['seq_lcs_f1']:.4f}</b><br>
          <span style="color:#2563eb;">●</span> Raw GPS &nbsp;
          <span style="color:#16a34a;">━━</span> Reference Route &nbsp;
          <span style="color:#dc2626;">━━</span> Predicted Arcs
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        m.save(str(MAPS_DIR / f"{best_key}_pred_{tid}.html"))


def main() -> None:
    setup_style()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    results = load_all_results()
    df = build_comparison_table(results)
    df.to_csv(EXPERIMENTS_DIR / "model_comparison.csv", index=False)
    write_markdown_table(df, EXPERIMENTS_DIR / "model_comparison.md")

    print("\n================ CONSOLIDATED 5-MODEL COMPARISON ================")
    print(
        df[
            [
                "model_name", "algorithm_family", "cv_macro_f1_oof", "cv_log_loss",
                "test_macro_f1", "test_top1_acc", "test_log_loss",
                "test_route_continuity", "test_switch_continuity", "test_seq_lcs_f1",
                "train_time_s", "ms_per_trip",
            ]
        ].to_string(index=False)
    )

    print("\nGenerating all comparison plots...")
    plot_cv_macro_f1(df)
    plot_student_test_comparison(df)
    plot_log_loss_and_efficiency(df)
    plot_confusion_matrices(results)
    plot_mlp_training_curves()
    plot_overfitting_check(df)
    plot_per_trip_heatmap(results)

    # Generate Folium prediction maps for the top two models (LightGBM & PyTorch MLP)
    for top_key in ["lightgbm", "mlp"]:
        sub = df[df["model_key"] == top_key]
        if len(sub) > 0:
            name = str(sub.iloc[0]["model_name"])
            print(f"Generating Folium prediction maps for {name}...")
            generate_best_model_maps(top_key, name)

    print(f"\nAll comparison tables, plots, and maps saved to {EXPERIMENTS_DIR}")


if __name__ == "__main__":
    main()
