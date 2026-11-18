"""
Build and Cache Candidate Feature Datasets for PathSnap (train & student_test).

Runs exact Shapely STRtree candidate generation, monotonic DP ground-truth alignment,
and feature engineering across all 80 training trips and 10 student_test trips.

Saves cached artifacts to outputs/features/:
- train_candidates.pkl
- student_test_candidates.pkl
- candidate_generation_summary.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from data_loader import list_trips, load_trip
from features import FEATURE_COLUMNS, extract_trip_candidates_and_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
FEATURES_DIR = PROJECT_ROOT / "outputs" / "features"

# Candidate generation hyperparameters (justified by EDA: 99.93% of student_test & 97.6% of train
# GPS points lie within 50m of reference route; bidirectional streets double directed arc count)
RADIUS_M = 60.0
MAX_K = 16
FALLBACK_K = 6


def build_partition(partition: str) -> Dict[str, Any]:
    t0 = time.time()
    trip_dirs = list_trips(DATASET_ROOT, partition)
    dfs: List[pd.DataFrame] = []
    trip_meta: Dict[str, Dict[str, Any]] = {}

    total_obs = 0
    total_covered = 0
    total_candidates = 0

    for i, td in enumerate(trip_dirs, 1):
        trip = load_trip(td, partition=partition, load_route=True)
        bundle = extract_trip_candidates_and_features(
            trip,
            radius_m=RADIUS_M,
            max_k=MAX_K,
            fallback_k=FALLBACK_K,
            include_labels=True,
        )
        dfs.append(bundle.df)
        total_obs += bundle.n_obs
        total_covered += bundle.n_covered_obs
        total_candidates += len(bundle.df)

        # Compute per-observation covered mask
        obs_idx = bundle.df["obs_idx"].to_numpy(dtype=np.int64)
        labels = bundle.df["label"].to_numpy(dtype=np.int8)
        covered_mask = np.bincount(obs_idx, weights=labels, minlength=bundle.n_obs) > 0

        trip_meta[trip.trip_id] = {
            "n_obs": bundle.n_obs,
            "n_candidates": len(bundle.df),
            "n_covered_obs": bundle.n_covered_obs,
            "candidate_miss_rate": float(1.0 - bundle.n_covered_obs / max(bundle.n_obs, 1)),
            "gt_arc_ids": bundle.gt_arc_ids,
            "gt_perp_dist_m": bundle.gt_perp_dist_m,
            "covered_mask": covered_mask,
            "ref_route": trip.route,
            "arcs": trip.arcs,
        }
        if i % 20 == 0 or i == len(trip_dirs):
            print(f"  [{partition}] processed {i}/{len(trip_dirs)} trips ({time.time() - t0:.1f}s)")

    full_df = pd.concat(dfs, ignore_index=True)
    elapsed = time.time() - t0

    summary = {
        "partition": partition,
        "n_trips": len(trip_dirs),
        "total_obs": int(total_obs),
        "total_covered_obs": int(total_covered),
        "candidate_miss_rate_weighted": float(1.0 - total_covered / max(total_obs, 1)),
        "candidate_miss_rate_trip_mean": float(
            np.mean([m["candidate_miss_rate"] for m in trip_meta.values()])
        ),
        "total_candidate_rows": int(total_candidates),
        "avg_candidates_per_obs": float(total_candidates / max(total_obs, 1)),
        "positive_rate": float(full_df["label"].mean()),
        "n_features": len(FEATURE_COLUMNS),
        "elapsed_s": round(elapsed, 2),
    }

    return {"df": full_df, "trip_meta": trip_meta, "summary": summary}


def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building candidate feature datasets (radius={RADIUS_M}m, max_k={MAX_K}, fallback_k={FALLBACK_K})...")

    all_summary: Dict[str, Any] = {
        "config": {
            "radius_m": RADIUS_M,
            "max_k": MAX_K,
            "fallback_k": FALLBACK_K,
            "features": FEATURE_COLUMNS,
        }
    }

    for part in ["train", "student_test"]:
        res = build_partition(part)
        out_path = FEATURES_DIR / f"{part}_candidates.joblib"
        joblib.dump({"df": res["df"], "trip_meta": res["trip_meta"]}, out_path, compress=3)
        all_summary[part] = res["summary"]
        print(f"Saved {part} -> {out_path}")
        print(json.dumps(res["summary"], indent=2))

    with open(FEATURES_DIR / "candidate_generation_summary.json", "w") as f:
        json.dump(all_summary, f, indent=2)


if __name__ == "__main__":
    main()
