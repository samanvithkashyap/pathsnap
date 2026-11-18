"""
HMM + Viterbi Sequential Route Reconstruction for PathSnap.

Combines calibrated candidate probabilities from the tuned classifier (emission model)
with road-network shortest-path & Newson-Krumm kinematic transitions (transition model)
using exact log-space Viterbi dynamic programming.

Key design features:
- Hidden states at time t: candidate directed arcs generated for observation t.
- Emission log-probability: emit_weight * log(clip(p_clf, eps, 1.0)).
- Transition log-probability:
    * Same arc (a == b): exact projection progress consistency + optional stay bonus.
    * Direct forward edge (to_node(a) == from_node(b)): exact road distance via endpoints,
      with explicit U-turn penalty if b is the exact reverse arc (v->u) of a (u->v).
    * Multi-hop directed shortest path (bounded Dijkstra with per-trip memoization):
      Newson-Krumm exponential penalty -|d_route - d_gps| / beta_eff.
    * Disconnected / unreachable transitions: finite fallback penalty (prevents Viterbi collapse).
    * Large GPS gaps (dt > 1s or dt >= gap_reset_s): adaptive beta scaling and soft reset across
      extreme dropouts so post-gap observations never corrupt pre-gap backtracking.
- Decoder hyperparameters are optimized strictly on the 80 training trips (using OOF
  probabilities) — never on student_test or evaluator_test.
"""

from __future__ import annotations

import heapq
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from data_loader import LocalMetricProjector, TripData, load_trip
from metrics import aggregate_trip_metrics, evaluate_trip_predictions, select_top1_arcs

PROJECT_ROOT = SRC_DIR.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
FEATURES_DIR = PROJECT_ROOT / "outputs" / "features"
TUNING_DIR = PROJECT_ROOT / "outputs" / "tuning"
OUT_DIR = PROJECT_ROOT / "outputs" / "hmm"
PLOTS_DIR = OUT_DIR / "plots"
MAPS_DIR = OUT_DIR / "maps"


@dataclass
class HMMConfig:
    emit_weight: float = 1.0          # Multiplier on log(p_classifier)
    beta: float = 10.0                # Exponential scale (m) for |d_route - d_gps|
    u_turn_penalty: float = 4.0       # Log-penalty for immediate reverse-arc flip (u->v to v->u)
    disconnect_penalty: float = 14.0  # Log-penalty for topologically unreachable transitions
    same_arc_bonus: float = 0.25      # Slight inertia bonus for staying on same road segment
    backward_proj_penalty_per_m: float = 0.08  # Penalty per meter of backward projection > 8m on same arc
    gap_reset_dt_s: float = 60.0      # Soft segment reset threshold for extreme GPS dropouts
    max_dijkstra_m: float = 2000.0    # Upper bound for bounded shortest-path search
    prob_eps: float = 1e-6            # Numerical floor for log probabilities


class RoadGraphIndex:
    """Lightweight directed adjacency list + memoized bounded Dijkstra for a single trip."""

    def __init__(self, trip: TripData) -> None:
        proj = LocalMetricProjector.from_trip(trip)
        nxy = proj.to_xy_array(trip.nodes)
        u = trip.arcs[:, 0]
        v = trip.arcs[:, 1]
        self.u = u
        self.v = v
        self.n_nodes = len(trip.nodes)
        self.arc_len = np.hypot(nxy[v, 0] - nxy[u, 0], nxy[v, 1] - nxy[u, 1]).astype(np.float64)

        # Directed adjacency list: adj[node] = list of (neighbor, edge_len_m)
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(self.n_nodes)]
        for a_i in range(len(u)):
            adj[int(u[a_i])].append((int(v[a_i]), float(self.arc_len[a_i])))
        self.adj = adj

        # Track metric coordinates & timestamps for GPS displacements
        tx, ty = proj.to_xy(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy())
        ts = trip.track["timestamp"].to_numpy(dtype=np.float64)
        self.tx = tx
        self.ty = ty
        self.ts = ts

        # Cache bounded Dijkstra results: source_node -> (computed_cutoff, dist_dict)
        self._sp_cache: Dict[int, Tuple[float, Dict[int, float]]] = {}

    def shortest_path_distance(self, src: int, dst: int, cutoff_m: float) -> Optional[float]:
        """Return shortest directed road distance from src to dst if <= cutoff_m, else None."""
        if src == dst:
            return 0.0
        cached = self._sp_cache.get(src)
        if cached is not None:
            cached_cutoff, dists = cached
            if dst in dists:
                d = dists[dst]
                return d if d <= cutoff_m else None
            if cached_cutoff >= cutoff_m:
                return None

        # Run bounded Dijkstra up to max(cutoff_m, 150.0) so subsequent queries reuse results
        run_cutoff = max(cutoff_m, 150.0)
        dists: Dict[int, float] = {src: 0.0}
        heap: List[Tuple[float, int]] = [(0.0, src)]
        adj = self.adj

        while heap:
            d, node = heapq.heappop(heap)
            if d > dists.get(node, float("inf")):
                continue
            if d > run_cutoff:
                break
            for nb, w in adj[node]:
                nd = d + w
                if nd <= run_cutoff and nd < dists.get(nb, float("inf")):
                    dists[nb] = nd
                    heapq.heappush(heap, (nd, nb))

        self._sp_cache[src] = (run_cutoff, dists)
        d_res = dists.get(dst)
        if d_res is not None and d_res <= cutoff_m:
            return d_res
        return None


def viterbi_decode_trip(
    trip_df: pd.DataFrame,
    cand_probs: np.ndarray,
    graph: RoadGraphIndex,
    n_obs: int,
    cfg: HMMConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run log-space Viterbi decoding for a single trip.

    Parameters
    ----------
    trip_df : pd.DataFrame
        Candidate rows for this trip (must contain obs_idx, arc_id, proj_frac).
    cand_probs : np.ndarray
        Classifier probabilities aligned row-for-row with trip_df.
    graph : RoadGraphIndex
        Precomputed road graph and metric GPS track for this trip.
    n_obs : int
        Number of GPS observations in the trip.
    cfg : HMMConfig
        Emission/transition hyperparameters.

    Returns
    -------
    pred_arcs : np.ndarray of shape (n_obs,), int64
        Reconstructed arc_id per GPS observation.
    diag : dict
        Decoder diagnostics (number of direct, multi-hop, same-arc, U-turn, disconnected steps).
    """
    obs_col = trip_df["obs_idx"].to_numpy(dtype=np.int64)
    arc_col = trip_df["arc_id"].to_numpy(dtype=np.int64)
    proj_col = trip_df["proj_frac"].to_numpy(dtype=np.float64)
    probs = np.clip(np.asarray(cand_probs, dtype=np.float64), cfg.prob_eps, 1.0)
    log_emit_all = cfg.emit_weight * np.log(probs)

    # Group candidate slices per obs_idx
    order = np.argsort(obs_col, kind="mergesort")
    obs_s = obs_col[order]
    arc_s = arc_col[order]
    proj_s = proj_col[order]
    emit_s = log_emit_all[order]

    counts = np.bincount(obs_s, minlength=n_obs)
    starts = np.zeros(n_obs + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])

    u = graph.u
    v = graph.v
    arc_len = graph.arc_len
    tx, ty, ts = graph.tx, graph.ty, graph.ts

    # Viterbi DP tables (variable number of candidates K_t per timestep)
    dp_tables: List[np.ndarray] = []
    backptrs: List[np.ndarray] = []
    cand_arcs_per_t: List[np.ndarray] = []

    prev_valid_t = -1
    for t in range(n_obs):
        s, e = int(starts[t]), int(starts[t + 1])
        cur_arcs = arc_s[s:e]
        cur_proj = proj_s[s:e]
        cur_emit = emit_s[s:e]
        cand_arcs_per_t.append(cur_arcs)

        if len(cur_arcs) == 0:
            # Missing candidate safeguard: carry empty placeholder
            dp_tables.append(np.empty(0, dtype=np.float64))
            backptrs.append(np.empty(0, dtype=np.int32))
            continue

        if prev_valid_t == -1:
            dp_tables.append(cur_emit.copy())
            backptrs.append(np.full(len(cur_arcs), -1, dtype=np.int32))
            prev_valid_t = t
            continue

        ps, pe = int(starts[prev_valid_t]), int(starts[prev_valid_t + 1])
        prev_arcs = arc_s[ps:pe]
        prev_proj = proj_s[ps:pe]
        prev_dp = dp_tables[prev_valid_t]

        dt = max(float(ts[t] - ts[prev_valid_t]), 1e-3)
        d_gps = float(np.hypot(tx[t] - tx[prev_valid_t], ty[t] - ty[prev_valid_t]))

        # Handle extreme GPS dropouts via clean soft-reset (avoid corrupting pre-gap path)
        if dt >= cfg.gap_reset_dt_s or d_gps >= cfg.max_dijkstra_m:
            best_prev_idx = int(np.argmax(prev_dp))
            best_prev_val = float(prev_dp[best_prev_idx])
            dp_cur = best_prev_val + cur_emit
            bp_cur = np.full(len(cur_arcs), best_prev_idx, dtype=np.int32)
            dp_tables.append(dp_cur)
            backptrs.append(bp_cur)
            prev_valid_t = t
            continue

        # Adaptive beta scales smoothly with multi-second GPS gaps
        beta_eff = cfg.beta * (1.0 + 0.15 * max(0.0, dt - 1.0))
        sp_cutoff = min(cfg.max_dijkstra_m, max(120.0, d_gps * 2.2 + 80.0))

        k_prev = len(prev_arcs)
        k_cur = len(cur_arcs)
        trans = np.full((k_prev, k_cur), -cfg.disconnect_penalty - d_gps / (beta_eff * 2.0), dtype=np.float64)

        for i in range(k_prev):
            a = int(prev_arcs[i])
            ua, va = int(u[a]), int(v[a])
            la = float(arc_len[a])
            rem_a = (1.0 - float(prev_proj[i])) * la

            for j in range(k_cur):
                b = int(cur_arcs[j])
                if a == b:
                    # Same road segment
                    delta_proj_m = (float(cur_proj[j]) - float(prev_proj[i])) * la
                    d_route = abs(delta_proj_m)
                    pen = abs(d_route - d_gps) / beta_eff
                    if delta_proj_m < -8.0:
                        pen += cfg.backward_proj_penalty_per_m * (-delta_proj_m - 8.0)
                    trans[i, j] = cfg.same_arc_bonus - pen
                    continue

                ub, vb = int(u[b]), int(v[b])
                lb = float(arc_len[b])
                ent_b = float(cur_proj[j]) * lb

                if va == ub:
                    # Direct 1-hop connected transition
                    d_route = rem_a + ent_b
                    score = -abs(d_route - d_gps) / beta_eff
                    if vb == ua:
                        # Exact U-turn reversal onto opposite direction of same physical street
                        score -= cfg.u_turn_penalty
                    trans[i, j] = score
                else:
                    # Multi-hop shortest path in directed graph
                    sp_d = graph.shortest_path_distance(va, ub, sp_cutoff)
                    if sp_d is not None:
                        d_route = rem_a + sp_d + ent_b
                        score = -abs(d_route - d_gps) / beta_eff
                        if vb == ua and sp_d < 15.0:
                            score -= cfg.u_turn_penalty
                        trans[i, j] = score

        # Viterbi recurrence: dp_cur[j] = emit[j] + max_i (prev_dp[i] + trans[i, j])
        total_scores = prev_dp[:, None] + trans
        bp_cur = np.argmax(total_scores, axis=0).astype(np.int32)
        dp_cur = cur_emit + total_scores[bp_cur, np.arange(k_cur)]

        # Normalize to prevent floating-point drift on very long trips (19k obs)
        dp_cur -= float(np.max(dp_cur))

        dp_tables.append(dp_cur)
        backptrs.append(bp_cur)
        prev_valid_t = t

    # Backtrack optimal state sequence
    pred_arcs = np.full(n_obs, -1, dtype=np.int64)
    if prev_valid_t == -1:
        return pred_arcs, {"n_same": 0, "n_direct": 0, "n_multihop": 0, "n_disconnected": 0}

    cur_idx = int(np.argmax(dp_tables[prev_valid_t]))
    pred_arcs[prev_valid_t] = int(cand_arcs_per_t[prev_valid_t][cur_idx])

    last_t = prev_valid_t
    for t in range(prev_valid_t - 1, -1, -1):
        if len(cand_arcs_per_t[t]) == 0:
            pred_arcs[t] = pred_arcs[last_t]
            continue
        cur_idx = int(backptrs[last_t][cur_idx])
        pred_arcs[t] = int(cand_arcs_per_t[t][cur_idx])
        last_t = t

    # Fill any trailing missing observations if any existed after prev_valid_t
    for t in range(prev_valid_t + 1, n_obs):
        pred_arcs[t] = pred_arcs[prev_valid_t]

    # Transition diagnostics
    n_same = n_direct = n_other = 0
    for t in range(1, n_obs):
        a, b = int(pred_arcs[t - 1]), int(pred_arcs[t])
        if a == b:
            n_same += 1
        elif v[a] == u[b]:
            n_direct += 1
        else:
            n_other += 1

    return pred_arcs, {
        "n_same_arc": n_same,
        "n_direct_1hop": n_direct,
        "n_multihop_or_gap": n_other,
    }


def evaluate_hmm_on_trips(
    partition: str,
    df: pd.DataFrame,
    probs: np.ndarray,
    trip_meta: Dict[str, Any],
    graphs: Dict[str, RoadGraphIndex],
    cfg: HMMConfig,
    trip_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run HMM/Viterbi across a set of trips and return per-trip & aggregated metrics."""
    per_trip: List[Dict[str, Any]] = []
    pred_arcs_by_trip: Dict[str, np.ndarray] = {}
    t0 = time.time()

    grouped = {tid: grp for tid, grp in df.groupby("trip_id", sort=True)}
    target_ids = sorted(trip_ids) if trip_ids is not None else sorted(grouped.keys())

    for tid in target_ids:
        grp = grouped[tid]
        meta = trip_meta[tid]
        trip_probs = probs[grp.index.to_numpy()]
        pred_arcs, diag = viterbi_decode_trip(grp, trip_probs, graphs[tid], meta["n_obs"], cfg)
        pred_arcs_by_trip[tid] = pred_arcs
        tm = evaluate_trip_predictions(
            pred_arcs=pred_arcs,
            gt_arc_ids=meta["gt_arc_ids"],
            ref_route=meta["ref_route"],
            arcs=meta["arcs"],
            covered_mask=meta["covered_mask"],
        )
        tm["trip_id"] = tid
        tm.update(diag)
        per_trip.append(tm)

    elapsed_s = time.time() - t0
    agg = aggregate_trip_metrics(per_trip)
    agg["decode_total_s"] = round(elapsed_s, 3)
    agg["decode_ms_per_trip"] = round(elapsed_s * 1000.0 / max(len(target_ids), 1), 2)
    return {
        "route_metrics": agg,
        "per_trip": per_trip,
        "pred_arcs_by_trip": pred_arcs_by_trip,
    }


def evaluate_clf_only_on_trips(
    df: pd.DataFrame,
    probs: np.ndarray,
    trip_meta: Dict[str, Any],
    trip_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate classifier-only Top-1 predictions on the same trips."""
    per_trip: List[Dict[str, Any]] = []
    pred_arcs_by_trip: Dict[str, np.ndarray] = {}
    grouped = {tid: grp for tid, grp in df.groupby("trip_id", sort=True)}
    target_ids = sorted(trip_ids) if trip_ids is not None else sorted(grouped.keys())

    for tid in target_ids:
        grp = grouped[tid]
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

    return {
        "route_metrics": aggregate_trip_metrics(per_trip),
        "per_trip": per_trip,
        "pred_arcs_by_trip": pred_arcs_by_trip,
    }


def plot_clf_vs_hmm(
    clf_test: Dict[str, Any],
    hmm_test: Dict[str, Any],
    clf_train: Dict[str, Any],
    hmm_train: Dict[str, Any],
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150})

    # 1. Overall Classifier-Only vs Classifier+HMM bar chart (Student-Test & Train OOF)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.0))
    metrics_keys = [
        ("top1_accuracy", "Top-1 Acc (All)"),
        ("top1_accuracy_covered", "Top-1 Acc (Cov)"),
        ("route_continuity", "Step Continuity"),
        ("transition_continuity_trip_mean", "Switch Continuity"),
        ("seq_arc_f1_mean", "Seq Arc F1"),
        ("seq_lcs_f1_mean", "Seq LCS F1"),
    ]

    for ax, title, c_res, h_res in [
        (axes[0], "Student-Test (10 Held-Out Trips)", clf_test["route_metrics"], hmm_test["route_metrics"]),
        (axes[1], "Train OOF (80 Training Trips)", clf_train["route_metrics"], hmm_train["route_metrics"]),
    ]:
        x = np.arange(len(metrics_keys))
        w = 0.36
        c_vals = [c_res[k] * 100 for k, _ in metrics_keys]
        h_vals = [h_res[k] * 100 for k, _ in metrics_keys]
        b1 = ax.bar(x - w / 2, c_vals, width=w, label="Tuned XGBoost (Classifier-Only)", color="#94a3b8")
        b2 = ax.bar(x + w / 2, h_vals, width=w, label="Tuned XGBoost + HMM/Viterbi", color="#16a34a")
        for bars in (b1, b2):
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.8)
        ax.set_ylim(60, 106)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in metrics_keys], rotation=18, ha="right")
        ax.set_ylabel("Score (%)")
        ax.set_title(title)
        ax.legend(fontsize=8.5, loc="lower right")

    fig.suptitle("Impact of HMM + Viterbi Decoding vs Classifier-Only Predictions", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_classifier_vs_hmm_overall.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Per-trip Student-Test improvement breakdown (Top-1 Acc, Switch Continuity, Seq LCS F1)
    c_df = pd.DataFrame(clf_test["per_trip"]).sort_values("trip_id").reset_index(drop=True)
    h_df = pd.DataFrame(hmm_test["per_trip"]).sort_values("trip_id").reset_index(drop=True)
    tids = [t[-4:] for t in c_df["trip_id"]]
    x = np.arange(len(tids))
    w = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    panels = [
        ("top1_accuracy", "Observation Top-1 Accuracy (%)"),
        ("transition_continuity", "Arc-Switch Continuity (%)"),
        ("seq_lcs_f1", "Sequence LCS F1 (×100)"),
    ]
    for ax, (col, title) in zip(axes, panels):
        cv = c_df[col].to_numpy() * 100
        hv = h_df[col].to_numpy() * 100
        ax.bar(x - w / 2, cv, width=w, label="Classifier-Only", color="#94a3b8")
        ax.bar(x + w / 2, hv, width=w, label="Classifier + HMM", color="#2563eb")
        ax.set_xticks(x)
        ax.set_xticklabels(tids, rotation=45)
        ax.set_ylim(min(30, float(min(cv.min(), hv.min()) - 8)), 104)
        ax.set_title(title)
        ax.set_xlabel("Student-Test Trip ID (last 4 digits)")
        ax.legend(fontsize=8, loc="lower left")

    fig.suptitle("Per-Trip Comparison on Student-Test: Classifier-Only vs Classifier + HMM/Viterbi", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_per_trip_hmm_gain.png", bbox_inches="tight")
    plt.close(fig)


def create_comparison_folium_map(
    partition: str,
    trip_id: str,
    clf_arcs: np.ndarray,
    hmm_arcs: np.ndarray,
    clf_row: Dict[str, Any],
    hmm_row: Dict[str, Any],
    out_path: Path,
) -> None:
    trip = load_trip(DATASET_ROOT / partition / trip_id, partition, load_route=True)
    lons = trip.track["lon"].to_numpy()
    lats = trip.track["lat"].to_numpy()
    u, v = trip.arcs[:, 0], trip.arcs[:, 1]
    nlon, nlat = trip.nodes[:, 0], trip.nodes[:, 1]

    m = folium.Map(location=[float(np.median(lats)), float(np.median(lons))], zoom_start=14, tiles="OpenStreetMap")

    # Layer 1: Road network
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
        opacity=0.45,
    ).add_to(road_layer)
    road_layer.add_to(m)

    # Layer 2: Reference route (green)
    ref_layer = folium.FeatureGroup(name="Reference Route (.route)", show=True)
    folium.PolyLine(
        [[(float(nlat[u[i]]), float(nlon[u[i]])), (float(nlat[v[i]]), float(nlon[v[i]]))] for i in trip.route],
        color="#16a34a",
        weight=7,
        opacity=0.55,
    ).add_to(ref_layer)
    ref_layer.add_to(m)

    # Layer 3: Classifier-only (orange, toggleable)
    clf_layer = folium.FeatureGroup(name="Classifier-Only (Tuned XGBoost)", show=False)
    uniq_clf = np.unique(clf_arcs[clf_arcs >= 0])
    folium.PolyLine(
        [[(float(nlat[u[i]]), float(nlon[u[i]])), (float(nlat[v[i]]), float(nlon[v[i]]))] for i in uniq_clf],
        color="#f97316",
        weight=3.5,
        opacity=0.85,
    ).add_to(clf_layer)
    clf_layer.add_to(m)

    # Layer 4: Classifier + HMM/Viterbi (red/crimson)
    hmm_layer = folium.FeatureGroup(name="Predicted Route (XGBoost + HMM/Viterbi)", show=True)
    uniq_hmm = np.unique(hmm_arcs[hmm_arcs >= 0])
    folium.PolyLine(
        [[(float(nlat[u[i]]), float(nlon[u[i]])), (float(nlat[v[i]]), float(nlon[v[i]]))] for i in uniq_hmm],
        color="#dc2626",
        weight=3.0,
        opacity=0.92,
    ).add_to(hmm_layer)
    hmm_layer.add_to(m)

    # Layer 5: Raw GPS observations
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
      <b>PathSnap Route Reconstruction — {partition}/{trip_id}</b><br>
      Clf-Only: Top-1 <b>{clf_row['top1_accuracy']*100:.2f}%</b> | Switch Cont <b>{clf_row['transition_continuity']*100:.2f}%</b> | LCS F1 <b>{clf_row['seq_lcs_f1']:.4f}</b><br>
      Clf+HMM: Top-1 <b>{hmm_row['top1_accuracy']*100:.2f}%</b> | Switch Cont <b>{hmm_row['transition_continuity']*100:.2f}%</b> | LCS F1 <b>{hmm_row['seq_lcs_f1']:.4f}</b><br>
      <span style="color:#2563eb;">●</span> Raw GPS &nbsp;
      <span style="color:#16a34a;">━━</span> Reference &nbsp;
      <span style="color:#dc2626;">━━</span> HMM Predicted &nbsp;
      <span style="color:#f97316;">━━</span> Clf-Only (toggle)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(str(out_path))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading cached feature datasets and Tuned XGBoost probabilities...")
    train_pack = joblib.load(FEATURES_DIR / "train_candidates.joblib")
    test_pack = joblib.load(FEATURES_DIR / "student_test_candidates.joblib")

    train_df: pd.DataFrame = train_pack["df"]
    train_meta: Dict[str, Any] = train_pack["trip_meta"]
    test_df: pd.DataFrame = test_pack["df"]
    test_meta: Dict[str, Any] = test_pack["trip_meta"]

    oof_probs = np.load(TUNING_DIR / "oof_probs.npy")
    test_probs = np.load(TUNING_DIR / "test_probs.npy")

    print("Building road-graph indices for 80 train trips and 10 student_test trips...")
    t_g0 = time.time()
    train_graphs: Dict[str, RoadGraphIndex] = {}
    for tid in sorted(train_meta.keys()):
        trip = load_trip(DATASET_ROOT / "train" / tid, "train", load_route=False)
        train_graphs[tid] = RoadGraphIndex(trip)

    test_graphs: Dict[str, RoadGraphIndex] = {}
    for tid in sorted(test_meta.keys()):
        trip = load_trip(DATASET_ROOT / "student_test" / tid, "student_test", load_route=False)
        test_graphs[tid] = RoadGraphIndex(trip)
    print(f"Built 90 road-graph indices in {time.time() - t_g0:.2f}s.")

    # --- Tune HMM decoder hyperparameters strictly on 80 training trips (OOF probabilities) ---
    # Use a diverse 20-trip training subset for coarse grid search, then verify top candidates on all 80 train trips
    all_train_ids = sorted(train_meta.keys())
    # Exclude the 2 extreme off-road stationary trips only from tuning objective ranking so signal isn't drowned
    # (final OOF evaluation still includes all 80 trips!)
    clean_train_ids = [tid for tid in all_train_ids if train_meta[tid]["candidate_miss_rate"] < 0.25]
    sub_tune_ids = clean_train_ids[::3]  # 26 diverse training trips

    print(f"\nTuning HMM decoder hyperparameters on {len(sub_tune_ids)} training trips (strictly OOF, zero test leakage)...")
    grid_records: List[Dict[str, Any]] = []
    param_grid = [
        HMMConfig(emit_weight=ew, beta=b, u_turn_penalty=ut, disconnect_penalty=dp, same_arc_bonus=sb)
        for ew in (0.6, 1.0, 1.5)
        for b in (6.0, 12.0, 22.0)
        for ut in (2.5, 5.0)
        for dp in (12.0, 20.0)
        for sb in (0.0, 0.35)
    ]
    # Focused structured search: first sweep (emit_weight, beta), then refine (u_turn_penalty, disconnect_penalty, same_arc_bonus)
    stage1_cfgs = [
        HMMConfig(emit_weight=ew, beta=b, u_turn_penalty=4.0, disconnect_penalty=15.0, same_arc_bonus=0.25)
        for ew in (0.4, 0.7, 1.0, 1.4, 2.0)
        for b in (5.0, 10.0, 18.0, 30.0)
    ]
    best_cfg = stage1_cfgs[0]
    best_score = -1.0

    for idx, cfg in enumerate(stage1_cfgs, 1):
        res = evaluate_hmm_on_trips("train", train_df, oof_probs, train_meta, train_graphs, cfg, trip_ids=sub_tune_ids)
        rm = res["route_metrics"]
        # Joint score balancing observation-level Top-1 accuracy and sequence-level LCS F1
        score = 0.5 * rm["top1_accuracy_covered"] + 0.5 * rm["seq_lcs_f1_mean"]
        rec = {
            "stage": 1,
            **asdict(cfg),
            "top1_cov": rm["top1_accuracy_covered"],
            "route_continuity": rm["route_continuity"],
            "switch_continuity": rm["transition_continuity_trip_mean"],
            "seq_lcs_f1": rm["seq_lcs_f1_mean"],
            "joint_score": score,
            "decode_ms_per_trip": rm["decode_ms_per_trip"],
        }
        grid_records.append(rec)
        if score > best_score:
            best_score = score
            best_cfg = cfg

    print(
        f"  Stage 1 best: emit_weight={best_cfg.emit_weight}, beta={best_cfg.beta} -> "
        f"joint={best_score:.4f}"
    )

    stage2_cfgs = [
        HMMConfig(
            emit_weight=best_cfg.emit_weight,
            beta=best_cfg.beta,
            u_turn_penalty=ut,
            disconnect_penalty=dp,
            same_arc_bonus=sb,
        )
        for ut in (1.5, 3.5, 6.0)
        for dp in (10.0, 16.0, 24.0)
        for sb in (0.0, 0.25, 0.6)
    ]
    for cfg in stage2_cfgs:
        res = evaluate_hmm_on_trips("train", train_df, oof_probs, train_meta, train_graphs, cfg, trip_ids=sub_tune_ids)
        rm = res["route_metrics"]
        score = 0.5 * rm["top1_accuracy_covered"] + 0.5 * rm["seq_lcs_f1_mean"]
        rec = {
            "stage": 2,
            **asdict(cfg),
            "top1_cov": rm["top1_accuracy_covered"],
            "route_continuity": rm["route_continuity"],
            "switch_continuity": rm["transition_continuity_trip_mean"],
            "seq_lcs_f1": rm["seq_lcs_f1_mean"],
            "joint_score": score,
            "decode_ms_per_trip": rm["decode_ms_per_trip"],
        }
        grid_records.append(rec)
        if score > best_score:
            best_score = score
            best_cfg = cfg

    print(f"  Stage 2 best HMM config: {asdict(best_cfg)} (train-subset joint={best_score:.4f})")

    # --- Full evaluation on ALL 80 train trips (OOF) and ALL 10 student_test trips ---
    print("\nEvaluating Classifier-Only vs Classifier+HMM on all 80 Train OOF trips...")
    clf_train = evaluate_clf_only_on_trips(train_df, oof_probs, train_meta)
    hmm_train = evaluate_hmm_on_trips("train", train_df, oof_probs, train_meta, train_graphs, best_cfg)

    print("Evaluating Classifier-Only vs Classifier+HMM on all 10 Student-Test trips...")
    clf_test = evaluate_clf_only_on_trips(test_df, test_probs, test_meta)
    hmm_test = evaluate_hmm_on_trips("student_test", test_df, test_probs, test_meta, test_graphs, best_cfg)

    # Print comparison summary
    for label, c_res, h_res in [
        ("TRAIN OOF (80 trips)", clf_train["route_metrics"], hmm_train["route_metrics"]),
        ("STUDENT-TEST (10 trips)", clf_test["route_metrics"], hmm_test["route_metrics"]),
    ]:
        print(f"\n================ {label}: CLASSIFIER-ONLY vs CLASSIFIER+HMM ================")
        print(f"Top-1 Acc (All)    : {c_res['top1_accuracy']*100:.2f}% -> {h_res['top1_accuracy']*100:.2f}% ({(h_res['top1_accuracy']-c_res['top1_accuracy'])*100:+.2f}%)")
        print(f"Top-1 Acc (Cov)    : {c_res['top1_accuracy_covered']*100:.2f}% -> {h_res['top1_accuracy_covered']*100:.2f}% ({(h_res['top1_accuracy_covered']-c_res['top1_accuracy_covered'])*100:+.2f}%)")
        print(f"Step Continuity    : {c_res['route_continuity']*100:.2f}% -> {h_res['route_continuity']*100:.2f}% ({(h_res['route_continuity']-c_res['route_continuity'])*100:+.2f}%)")
        print(f"Switch Continuity  : {c_res['transition_continuity_trip_mean']*100:.2f}% -> {h_res['transition_continuity_trip_mean']*100:.2f}% ({(h_res['transition_continuity_trip_mean']-c_res['transition_continuity_trip_mean'])*100:+.2f}%)")
        print(f"Seq Arc F1         : {c_res['seq_arc_f1_mean']:.4f} -> {h_res['seq_arc_f1_mean']:.4f} ({h_res['seq_arc_f1_mean']-c_res['seq_arc_f1_mean']:+.4f})")
        print(f"Seq LCS F1         : {c_res['seq_lcs_f1_mean']:.4f} -> {h_res['seq_lcs_f1_mean']:.4f} ({h_res['seq_lcs_f1_mean']-c_res['seq_lcs_f1_mean']:+.4f})")
        print(f"HMM Decode Latency : {h_res['decode_ms_per_trip']:.2f} ms/trip (total {h_res['decode_total_s']:.2f}s)")

    # Per-trip Student-Test comparison table
    c_pt = pd.DataFrame(clf_test["per_trip"]).sort_values("trip_id").reset_index(drop=True)
    h_pt = pd.DataFrame(hmm_test["per_trip"]).sort_values("trip_id").reset_index(drop=True)
    comp_pt = pd.DataFrame({
        "trip_id": c_pt["trip_id"],
        "n_obs": c_pt["n_obs"],
        "clf_top1": c_pt["top1_accuracy"],
        "hmm_top1": h_pt["top1_accuracy"],
        "delta_top1": h_pt["top1_accuracy"] - c_pt["top1_accuracy"],
        "clf_step_cont": c_pt["route_continuity"],
        "hmm_step_cont": h_pt["route_continuity"],
        "clf_switch_cont": c_pt["transition_continuity"],
        "hmm_switch_cont": h_pt["transition_continuity"],
        "clf_lcs_f1": c_pt["seq_lcs_f1"],
        "hmm_lcs_f1": h_pt["seq_lcs_f1"],
        "delta_lcs_f1": h_pt["seq_lcs_f1"] - c_pt["seq_lcs_f1"],
    })
    print("\n=== Student-Test Per-Trip Comparison (Classifier-Only vs Classifier+HMM) ===")
    print(comp_pt.to_string(index=False))

    # Save artifacts
    pd.DataFrame(grid_records).to_csv(OUT_DIR / "hmm_tuning_grid.csv", index=False)
    comp_pt.to_csv(OUT_DIR / "student_test_clf_vs_hmm.csv", index=False)

    hmm_summary = {
        "hmm_config": asdict(best_cfg),
        "tuning_subset_trips": len(sub_tune_ids),
        "tuning_grid_evaluated": len(grid_records),
        "train_oof_80_trips": {
            "classifier_only": clf_train["route_metrics"],
            "classifier_plus_hmm": hmm_train["route_metrics"],
        },
        "student_test_10_trips": {
            "classifier_only": clf_test["route_metrics"],
            "classifier_plus_hmm": hmm_test["route_metrics"],
        },
    }
    with open(OUT_DIR / "hmm_results.json", "w") as f:
        json.dump(hmm_summary, f, indent=2)
    with open(OUT_DIR / "hmm_config.json", "w") as f:
        json.dump(asdict(best_cfg), f, indent=2)

    # Plots & Folium maps
    print("\nGenerating HMM comparison plots and interactive Folium maps...")
    plot_clf_vs_hmm(clf_test, hmm_test, clf_train, hmm_train)

    c_lookup = {r["trip_id"]: r for r in clf_test["per_trip"]}
    h_lookup = {r["trip_id"]: r for r in hmm_test["per_trip"]}
    for tid in ["00000000", "00000056", "00000077", "00000076"]:
        create_comparison_folium_map(
            "student_test",
            tid,
            clf_test["pred_arcs_by_trip"][tid],
            hmm_test["pred_arcs_by_trip"][tid],
            c_lookup[tid],
            h_lookup[tid],
            MAPS_DIR / f"hmm_vs_clf_{tid}.html",
        )

    print(f"Saved all HMM/Viterbi results, config, plots, and Folium maps to {OUT_DIR}")


if __name__ == "__main__":
    main()
