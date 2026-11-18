"""
Unified Evaluation Metrics for PathSnap (Candidate-level, Observation-level, Route-level).

Implements every metric specified in Models and Training Guidelines.pdf and Deliverables.pdf:
- Candidate binary Accuracy, Precision, Recall, Macro F1-score, Positive F1-score, Log Loss, Confusion Matrix
- Observation-level Top-1 Accuracy (overall & on covered observations)
- Candidate Miss Rate
- Route Continuity (step-level & arc-transition-level directed graph connectivity)
- Sequence-level Performance (deduplicated route Precision/Recall/F1 and order-preserving LCS F1)
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def deduplicate_sequence(seq: np.ndarray) -> np.ndarray:
    """Collapse consecutive duplicate arc_ids into a deduplicated route sequence."""
    if len(seq) == 0:
        return seq
    mask = np.empty(len(seq), dtype=bool)
    mask[0] = True
    mask[1:] = seq[1:] != seq[:-1]
    return seq[mask]


def fast_lcs_length(a: np.ndarray, b: np.ndarray) -> int:
    """
    Compute exact Longest Common Subsequence length between two integer sequences
    in O(M log M) time using Hunt-Szymanski / patience sorting on matching indices.
    """
    if len(a) == 0 or len(b) == 0:
        return 0
    pos_b: Dict[int, List[int]] = defaultdict(list)
    for idx, val in enumerate(b.tolist()):
        pos_b[val].append(idx)

    tails: List[int] = []
    for val in a.tolist():
        positions = pos_b.get(val)
        if not positions:
            continue
        # Iterate in reverse so each element of `a` matches at most once
        for p in reversed(positions):
            i = bisect_left(tails, p)
            if i == len(tails):
                tails.append(p)
            else:
                tails[i] = p
    return len(tails)


def evaluate_trip_predictions(
    pred_arcs: np.ndarray,
    gt_arc_ids: np.ndarray,
    ref_route: np.ndarray,
    arcs: np.ndarray,
    covered_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute observation-level, route-continuity, and sequence-level metrics for a single trip.
    """
    n_obs = len(gt_arc_ids)
    correct = pred_arcs == gt_arc_ids
    top1_acc_all = float(np.mean(correct)) if n_obs > 0 else 0.0

    if covered_mask is not None:
        n_cov = int(np.sum(covered_mask))
        miss_rate = float(1.0 - n_cov / max(n_obs, 1))
        top1_acc_cov = float(np.mean(correct[covered_mask])) if n_cov > 0 else 0.0
    else:
        n_cov = n_obs
        miss_rate = 0.0
        top1_acc_cov = top1_acc_all

    # Directed road-network continuity
    u = arcs[:, 0]
    v = arcs[:, 1]
    if n_obs >= 2:
        prev_a = pred_arcs[:-1]
        next_a = pred_arcs[1:]
        same_arc = prev_a == next_a
        valid_edge = v[prev_a] == u[next_a]
        step_continuous = same_arc | valid_edge
        route_continuity_step = float(np.mean(step_continuous))

        # Continuity restricted to actual arc transitions (prev_a != next_a)
        switches = ~same_arc
        n_switches = int(np.sum(switches))
        if n_switches > 0:
            route_continuity_trans = float(np.mean(valid_edge[switches]))
        else:
            route_continuity_trans = 1.0
    else:
        route_continuity_step = 1.0
        route_continuity_trans = 1.0
        n_switches = 0

    # Deduplicated sequence-level metrics against .route
    pred_seq = deduplicate_sequence(pred_arcs)
    ref_seq = deduplicate_sequence(ref_route)

    # Set-level precision/recall/F1 on unique arcs
    pred_set = set(pred_seq.tolist())
    ref_set = set(ref_seq.tolist())
    inter = len(pred_set & ref_set)
    seq_prec = inter / len(pred_set) if pred_set else 0.0
    seq_rec = inter / len(ref_set) if ref_set else 0.0
    seq_f1 = (2 * seq_prec * seq_rec / (seq_prec + seq_rec)) if (seq_prec + seq_rec) > 0 else 0.0

    # Order-preserving LCS F1
    lcs_len = fast_lcs_length(pred_seq, ref_seq)
    lcs_prec = lcs_len / len(pred_seq) if len(pred_seq) > 0 else 0.0
    lcs_rec = lcs_len / len(ref_seq) if len(ref_seq) > 0 else 0.0
    lcs_f1 = (2 * lcs_prec * lcs_rec / (lcs_prec + lcs_rec)) if (lcs_prec + lcs_rec) > 0 else 0.0

    return {
        "n_obs": n_obs,
        "n_covered_obs": n_cov,
        "candidate_miss_rate": miss_rate,
        "top1_accuracy": top1_acc_all,
        "top1_accuracy_covered": top1_acc_cov,
        "route_continuity": route_continuity_step,
        "transition_continuity": route_continuity_trans,
        "n_arc_switches": n_switches,
        "pred_seq_len": int(len(pred_seq)),
        "ref_seq_len": int(len(ref_seq)),
        "seq_arc_precision": float(seq_prec),
        "seq_arc_recall": float(seq_rec),
        "seq_arc_f1": float(seq_f1),
        "seq_lcs_f1": float(lcs_f1),
    }


def select_top1_arcs(
    df_candidates: pd.DataFrame,
    scores: np.ndarray,
    n_obs: int,
) -> np.ndarray:
    """
    For a single trip's candidate DataFrame and predicted candidate scores,
    return array of shape (n_obs,) containing the top-1 predicted arc_id per observation.
    """
    obs_idx = df_candidates["obs_idx"].to_numpy(dtype=np.int64)
    arc_ids = df_candidates["arc_id"].to_numpy(dtype=np.int64)

    # Sort by (obs_idx asc, score desc) -> first occurrence per obs_idx is argmax
    order = np.lexsort((-scores, obs_idx))
    obs_s = obs_idx[order]
    arc_s = arc_ids[order]

    first_mask = np.empty(len(obs_s), dtype=bool)
    first_mask[0] = True
    first_mask[1:] = obs_s[1:] != obs_s[:-1]

    pred = np.full(n_obs, -1, dtype=np.int64)
    pred[obs_s[first_mask]] = arc_s[first_mask]
    return pred


def evaluate_candidate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    obs_groups: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Compute candidate-level binary classification metrics.
    """
    y_prob_clip = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
    y_pred = (y_prob_clip >= threshold).astype(np.int8)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    res: Dict[str, Any] = {
        "candidate_accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "positive_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, y_prob_clip, labels=[0, 1])),
        "confusion_matrix": cm.tolist(),
    }

    # Also compute Top-1 selection binary Macro F1 (setting exactly 1 candidate=1 per obs)
    if obs_groups is not None:
        trip_codes, obs_idx = obs_groups
        # Unique key per (trip, obs)
        keys = trip_codes.astype(np.int64) * 1_000_000 + obs_idx.astype(np.int64)
        order = np.lexsort((-y_prob_clip, keys))
        keys_s = keys[order]
        first = np.empty(len(keys_s), dtype=bool)
        first[0] = True
        first[1:] = keys_s[1:] != keys_s[:-1]
        y_pred_top1 = np.zeros(len(y_true), dtype=np.int8)
        y_pred_top1[order[first]] = 1
        res["top1_candidate_macro_f1"] = float(f1_score(y_true, y_pred_top1, average="macro", zero_division=0))
        res["top1_candidate_positive_f1"] = float(f1_score(y_true, y_pred_top1, pos_label=1, zero_division=0))

    return res


def aggregate_trip_metrics(trip_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate per-trip metrics both weighted by observation count (micro)
    and unweighted mean across trips (macro).
    """
    df_m = pd.DataFrame(trip_metrics)
    total_obs = float(df_m["n_obs"].sum())
    total_cov = float(df_m["n_covered_obs"].sum())

    weighted_top1 = float(np.sum(df_m["top1_accuracy"] * df_m["n_obs"]) / max(total_obs, 1.0))
    weighted_top1_cov = float(np.sum(df_m["top1_accuracy_covered"] * df_m["n_covered_obs"]) / max(total_cov, 1.0))
    weighted_miss = float(1.0 - total_cov / max(total_obs, 1.0))
    weighted_cont = float(np.sum(df_m["route_continuity"] * df_m["n_obs"]) / max(total_obs, 1.0))

    return {
        "n_trips": int(len(df_m)),
        "total_obs": int(total_obs),
        "total_covered_obs": int(total_cov),
        "candidate_miss_rate": weighted_miss,
        "candidate_miss_rate_trip_mean": float(df_m["candidate_miss_rate"].mean()),
        "top1_accuracy": weighted_top1,
        "top1_accuracy_trip_mean": float(df_m["top1_accuracy"].mean()),
        "top1_accuracy_covered": weighted_top1_cov,
        "route_continuity": weighted_cont,
        "route_continuity_trip_mean": float(df_m["route_continuity"].mean()),
        "transition_continuity_trip_mean": float(df_m["transition_continuity"].mean()),
        "seq_arc_precision_mean": float(df_m["seq_arc_precision"].mean()),
        "seq_arc_recall_mean": float(df_m["seq_arc_recall"].mean()),
        "seq_arc_f1_mean": float(df_m["seq_arc_f1"].mean()),
        "seq_lcs_f1_mean": float(df_m["seq_lcs_f1"].mean()),
    }
