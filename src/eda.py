"""
Comprehensive Exploratory Data Analysis (EDA) for PathSnap Dataset.

Inspects both 'train' (80 trips) and 'student_test' (10 trips) partitions:
1. Inventory: trip counts, file availability, file sizes, row counts.
2. Data quality: missing/NaN values, invalid coordinates, timestamp ordering,
   duplicate timestamps, irregular time gaps (dt != 1s), large GPS jumps/speeds.
3. Geographical extents: bounding boxes, trip spans (km), global/regional coverage.
4. Road network topology: node/arc counts, arc length distributions, bidirectional arcs,
   self-loops, duplicate arcs, degree distributions, isolated nodes, weakly & strongly
   connected components.
5. Reference route (.route) analysis: arc validity, route length vs GPS count,
   route connectivity (consecutive arc continuity), repeated arcs, distance from GPS
   observations to reference route vs full road network, and alignment behavior.

Saves structured artifacts to outputs/eda/.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import pandas as pd

from data_loader import (
    LocalMetricProjector,
    TripData,
    haversine_m,
    list_trips,
    load_trip,
    point_to_segments_distance_and_proj,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"


def analyze_trip(trip: TripData) -> Dict[str, Any]:
    """Compute comprehensive EDA statistics for a single trip."""
    rec: Dict[str, Any] = {
        "partition": trip.partition,
        "trip_id": trip.trip_id,
        "missing_files": ",".join(trip.missing_files) if trip.missing_files else "none",
        "size_track_bytes": trip.file_sizes["track"],
        "size_nodes_bytes": trip.file_sizes["nodes"],
        "size_arcs_bytes": trip.file_sizes["arcs"],
        "size_route_bytes": trip.file_sizes["route"],
    }

    # --- Track analysis ---
    df = trip.track
    n_obs = len(df)
    rec["n_obs"] = n_obs
    rec["n_nan_lon"] = int(df["lon"].isna().sum())
    rec["n_nan_lat"] = int(df["lat"].isna().sum())
    rec["n_nan_ts"] = int(df["timestamp"].isna().sum())
    rec["n_any_nan"] = int(df.isna().any(axis=1).sum())

    valid = df.dropna()
    lons = valid["lon"].to_numpy(dtype=np.float64)
    lats = valid["lat"].to_numpy(dtype=np.float64)
    ts = valid["timestamp"].to_numpy(dtype=np.float64)

    rec["n_invalid_coord"] = int(
        np.sum((lons < -180) | (lons > 180) | (lats < -90) | (lats > 90) | ((lons == 0) & (lats == 0)))
    )

    if len(valid) >= 2:
        dt = np.diff(ts)
        step_dist_m = haversine_m(lons[:-1], lats[:-1], lons[1:], lats[1:])
        valid_dt = dt[dt > 0]
        speeds_mps = np.where(dt > 0, step_dist_m / np.where(dt > 0, dt, 1.0), np.nan)
        valid_speeds = speeds_mps[np.isfinite(speeds_mps)]

        rec["t_start"] = float(ts[0])
        rec["t_end"] = float(ts[-1])
        rec["duration_s"] = float(ts[-1] - ts[0])
        rec["n_dt_neg"] = int(np.sum(dt < 0))
        rec["n_dt_zero"] = int(np.sum(dt == 0))
        rec["n_dt_eq1"] = int(np.sum(np.isclose(dt, 1.0, atol=1e-3)))
        rec["pct_dt_eq1"] = float(100.0 * rec["n_dt_eq1"] / len(dt))
        rec["n_dt_gt1"] = int(np.sum(dt > 1.001))
        rec["n_dt_gt5"] = int(np.sum(dt > 5.0))
        rec["n_dt_gt30"] = int(np.sum(dt > 30.0))
        rec["max_dt_s"] = float(np.max(dt))
        rec["median_dt_s"] = float(np.median(dt))
        rec["total_gps_dist_km"] = float(np.sum(step_dist_m) / 1000.0)
        rec["max_step_dist_m"] = float(np.max(step_dist_m))
        rec["n_jumps_gt100m"] = int(np.sum(step_dist_m > 100.0))
        rec["n_jumps_gt500m"] = int(np.sum(step_dist_m > 500.0))
        rec["median_speed_kmh"] = float(np.median(valid_speeds) * 3.6) if len(valid_speeds) else 0.0
        rec["p99_speed_kmh"] = float(np.percentile(valid_speeds, 99) * 3.6) if len(valid_speeds) else 0.0
        rec["max_speed_kmh"] = float(np.max(valid_speeds) * 3.6) if len(valid_speeds) else 0.0
        rec["n_speed_gt150kmh"] = int(np.sum(valid_speeds * 3.6 > 150.0))
    else:
        for k in [
            "t_start", "t_end", "duration_s", "n_dt_neg", "n_dt_zero", "n_dt_eq1",
            "pct_dt_eq1", "n_dt_gt1", "n_dt_gt5", "n_dt_gt30", "max_dt_s", "median_dt_s",
            "total_gps_dist_km", "max_step_dist_m", "n_jumps_gt100m", "n_jumps_gt500m",
            "median_speed_kmh", "p99_speed_kmh", "max_speed_kmh", "n_speed_gt150kmh",
        ]:
            rec[k] = 0

    # Geographic extent
    proj = LocalMetricProjector.from_trip(trip)
    tx, ty = proj.to_xy(lons, lats)
    rec["lon_min"] = float(np.min(lons))
    rec["lon_max"] = float(np.max(lons))
    rec["lat_min"] = float(np.min(lats))
    rec["lat_max"] = float(np.max(lats))
    rec["lon_center"] = float(np.median(lons))
    rec["lat_center"] = float(np.median(lats))
    rec["span_x_km"] = float((np.max(tx) - np.min(tx)) / 1000.0)
    rec["span_y_km"] = float((np.max(ty) - np.min(ty)) / 1000.0)

    # --- Road network analysis ---
    n_nodes = len(trip.nodes)
    n_arcs = len(trip.arcs)
    rec["n_nodes"] = n_nodes
    rec["n_arcs"] = n_arcs
    rec["n_nodes_nan"] = int(np.isnan(trip.nodes).sum())

    u = trip.arcs[:, 0]
    v = trip.arcs[:, 1]
    rec["n_invalid_arc_node_id"] = int(
        np.sum((u < 0) | (u >= n_nodes) | (v < 0) | (v >= n_nodes))
    )
    rec["n_self_loops"] = int(np.sum(u == v))

    # Duplicate directed arcs and bidirectional arcs
    arc_pairs = list(zip(u.tolist(), v.tolist()))
    pair_set = set(arc_pairs)
    rec["n_duplicate_arcs"] = n_arcs - len(pair_set)
    n_bidir = sum(1 for (a, b) in pair_set if (b, a) in pair_set)
    rec["pct_bidirectional_arcs"] = float(100.0 * n_bidir / max(len(pair_set), 1))

    # Arc lengths in meters
    nxy = proj.to_xy_array(trip.nodes)
    ax, ay = nxy[u, 0], nxy[u, 1]
    bx, by = nxy[v, 0], nxy[v, 1]
    arc_lens = np.hypot(bx - ax, by - ay)
    rec["arc_len_min_m"] = float(np.min(arc_lens))
    rec["arc_len_median_m"] = float(np.median(arc_lens))
    rec["arc_len_mean_m"] = float(np.mean(arc_lens))
    rec["arc_len_p95_m"] = float(np.percentile(arc_lens, 95))
    rec["arc_len_max_m"] = float(np.max(arc_lens))
    rec["n_zero_len_arcs"] = int(np.sum(arc_lens < 1e-6))

    # Graph connectivity via NetworkX
    g = nx.DiGraph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from(arc_pairs)
    in_deg = np.array([d for _, d in g.in_degree()])
    out_deg = np.array([d for _, d in g.out_degree()])
    tot_deg = in_deg + out_deg
    rec["n_isolated_nodes"] = int(np.sum(tot_deg == 0))
    rec["n_deadend_out0"] = int(np.sum((out_deg == 0) & (in_deg > 0)))
    rec["mean_out_degree"] = float(np.mean(out_deg))
    rec["max_out_degree"] = int(np.max(out_deg))

    wcc = list(nx.weakly_connected_components(g))
    scc = list(nx.strongly_connected_components(g))
    rec["n_wcc"] = len(wcc)
    rec["largest_wcc_pct"] = float(100.0 * max(len(c) for c in wcc) / max(n_nodes, 1))
    rec["n_scc"] = len(scc)
    rec["largest_scc_pct"] = float(100.0 * max(len(c) for c in scc) / max(n_nodes, 1))

    # --- Reference route analysis ---
    if trip.route is not None and len(trip.route) > 0:
        r = trip.route
        rec["n_route_arcs"] = len(r)
        rec["n_route_unique_arcs"] = len(set(r.tolist()))
        rec["route_has_repeats"] = int(len(r) != len(set(r.tolist())))
        rec["n_invalid_route_arc_id"] = int(np.sum((r < 0) | (r >= n_arcs)))

        valid_r = r[(r >= 0) & (r < n_arcs)]
        route_len_m = float(np.sum(arc_lens[valid_r]))
        rec["route_total_len_km"] = route_len_m / 1000.0
        rec["gps_to_route_len_ratio"] = (
            rec["total_gps_dist_km"] / rec["route_total_len_km"]
            if rec["route_total_len_km"] > 0
            else np.nan
        )

        # Consecutive connectivity in .route: to_node(r[i]) == from_node(r[i+1])
        r_from = u[valid_r]
        r_to = v[valid_r]
        if len(valid_r) >= 2:
            connected = r_to[:-1] == r_from[1:]
            rec["route_n_transitions"] = len(connected)
            rec["route_n_connected"] = int(np.sum(connected))
            rec["route_n_disconnected"] = int(np.sum(~connected))
            rec["route_continuity_pct"] = float(100.0 * np.mean(connected))
        else:
            rec["route_n_transitions"] = 0
            rec["route_n_connected"] = 0
            rec["route_n_disconnected"] = 0
            rec["route_continuity_pct"] = 100.0

        # Distance from each GPS point to closest arc in reference route
        r_ax, r_ay = ax[valid_r], ay[valid_r]
        r_bx, r_by = bx[valid_r], by[valid_r]
        gps_to_route_dists = np.empty(len(tx), dtype=np.float64)
        closest_route_idx = np.empty(len(tx), dtype=np.int64)
        for i in range(len(tx)):
            d, _, _, _ = point_to_segments_distance_and_proj(tx[i], ty[i], r_ax, r_ay, r_bx, r_by)
            idx = int(np.argmin(d))
            gps_to_route_dists[i] = d[idx]
            closest_route_idx[i] = idx

        rec["gps_to_route_dist_median_m"] = float(np.median(gps_to_route_dists))
        rec["gps_to_route_dist_mean_m"] = float(np.mean(gps_to_route_dists))
        rec["gps_to_route_dist_p95_m"] = float(np.percentile(gps_to_route_dists, 95))
        rec["gps_to_route_dist_max_m"] = float(np.max(gps_to_route_dists))
        rec["pct_gps_within_15m_route"] = float(100.0 * np.mean(gps_to_route_dists <= 15.0))
        rec["pct_gps_within_30m_route"] = float(100.0 * np.mean(gps_to_route_dists <= 30.0))
        rec["pct_gps_within_50m_route"] = float(100.0 * np.mean(gps_to_route_dists <= 50.0))
        rec["pct_gps_within_100m_route"] = float(100.0 * np.mean(gps_to_route_dists <= 100.0))

        # Check monotonicity of closest_route_idx along track (sanity check for alignment)
        idx_diffs = np.diff(closest_route_idx)
        rec["route_idx_nondecreasing_pct"] = float(100.0 * np.mean(idx_diffs >= 0))
    else:
        rec["n_route_arcs"] = 0

    return rec


def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    all_dt_values: Dict[str, List[float]] = {"train": [], "student_test": []}

    for partition in ["train", "student_test"]:
        trip_dirs = list_trips(DATASET_ROOT, partition)
        print(f"Scanning partition '{partition}': {len(trip_dirs)} trips...")
        for td in trip_dirs:
            trip = load_trip(td, partition=partition, load_route=True)
            rows.append(analyze_trip(trip))
            ts = trip.track["timestamp"].dropna().to_numpy()
            if len(ts) >= 2:
                all_dt_values[partition].extend(np.diff(ts).tolist())

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "trip_inventory_eda.csv", index=False)

    # Aggregate summary per partition
    summary: Dict[str, Any] = {}
    for partition in ["train", "student_test"]:
        sub = df[df["partition"] == partition]
        dts = np.array(all_dt_values[partition], dtype=np.float64)
        dt_counts = Counter(np.round(dts, 2).tolist())
        top_dt = dt_counts.most_common(10)

        summary[partition] = {
            "n_trips": int(len(sub)),
            "total_gps_observations": int(sub["n_obs"].sum()),
            "obs_per_trip": {
                "min": int(sub["n_obs"].min()),
                "median": float(sub["n_obs"].median()),
                "mean": float(sub["n_obs"].mean()),
                "max": int(sub["n_obs"].max()),
            },
            "missing_files_trips": int((sub["missing_files"] != "none").sum()),
            "total_nan_rows_track": int(sub["n_any_nan"].sum()),
            "total_invalid_coords": int(sub["n_invalid_coord"].sum()),
            "total_nan_nodes": int(sub["n_nodes_nan"].sum()),
            "total_invalid_arc_node_ids": int(sub["n_invalid_arc_node_id"].sum()),
            "total_invalid_route_arc_ids": int(sub["n_invalid_route_arc_id"].sum()),
            "dt_summary": {
                "total_intervals": int(len(dts)),
                "pct_dt_eq_1s": float(100.0 * np.mean(np.isclose(dts, 1.0, atol=1e-3))),
                "n_dt_neg": int(np.sum(dts < 0)),
                "n_dt_zero": int(np.sum(dts == 0)),
                "n_dt_gt_5s": int(np.sum(dts > 5.0)),
                "n_dt_gt_30s": int(np.sum(dts > 30.0)),
                "max_dt_s": float(np.max(dts)),
                "top_10_dt_values": top_dt,
            },
            "geo_bounds": {
                "lon_min": float(sub["lon_min"].min()),
                "lon_max": float(sub["lon_max"].max()),
                "lat_min": float(sub["lat_min"].min()),
                "lat_max": float(sub["lat_max"].max()),
                "span_x_km_median": float(sub["span_x_km"].median()),
                "span_x_km_max": float(sub["span_x_km"].max()),
                "span_y_km_median": float(sub["span_y_km"].median()),
                "span_y_km_max": float(sub["span_y_km"].max()),
            },
            "road_network": {
                "nodes_min": int(sub["n_nodes"].min()),
                "nodes_median": float(sub["n_nodes"].median()),
                "nodes_max": int(sub["n_nodes"].max()),
                "arcs_min": int(sub["n_arcs"].min()),
                "arcs_median": float(sub["n_arcs"].median()),
                "arcs_max": int(sub["n_arcs"].max()),
                "arc_len_median_m": float(sub["arc_len_median_m"].median()),
                "pct_bidirectional_median": float(sub["pct_bidirectional_arcs"].median()),
                "total_self_loops": int(sub["n_self_loops"].sum()),
                "total_duplicate_arcs": int(sub["n_duplicate_arcs"].sum()),
                "total_isolated_nodes": int(sub["n_isolated_nodes"].sum()),
                "wcc_median": float(sub["n_wcc"].median()),
                "wcc_max": int(sub["n_wcc"].max()),
                "largest_wcc_pct_min": float(sub["largest_wcc_pct"].min()),
                "largest_scc_pct_median": float(sub["largest_scc_pct"].median()),
            },
            "reference_route": {
                "route_arcs_min": int(sub["n_route_arcs"].min()),
                "route_arcs_median": float(sub["n_route_arcs"].median()),
                "route_arcs_max": int(sub["n_route_arcs"].max()),
                "trips_with_repeats": int(sub["route_has_repeats"].sum()),
                "trips_with_disconnected_route": int((sub["route_n_disconnected"] > 0).sum()),
                "total_route_disconnected_transitions": int(sub["route_n_disconnected"].sum()),
                "route_continuity_min_pct": float(sub["route_continuity_pct"].min()),
                "gps_to_route_median_m": float(sub["gps_to_route_dist_median_m"].median()),
                "gps_to_route_p95_m": float(sub["gps_to_route_dist_p95_m"].median()),
                "gps_to_route_max_m": float(sub["gps_to_route_dist_max_m"].max()),
                "pct_gps_within_15m": float(sub["pct_gps_within_15m_route"].mean()),
                "pct_gps_within_30m": float(sub["pct_gps_within_30m_route"].mean()),
                "pct_gps_within_50m": float(sub["pct_gps_within_50m_route"].mean()),
                "pct_gps_within_100m": float(sub["pct_gps_within_100m_route"].mean()),
            },
        }

    with open(OUTPUT_DIR / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ EDA SUMMARY ================")
    print(json.dumps(summary, indent=2))

    # Print notable anomalies (large dt gaps, large GPS jumps, large distance to route, disconnected routes)
    print("\n=== Top 10 Trips by Max Time Gap (dt > 5s) ===")
    cols_gap = ["partition", "trip_id", "n_obs", "max_dt_s", "n_dt_gt5", "max_step_dist_m", "max_speed_kmh", "gps_to_route_dist_max_m"]
    print(df.sort_values("max_dt_s", ascending=False)[cols_gap].head(10).to_string(index=False))

    print("\n=== Top 10 Trips by Max GPS Distance to Reference Route ===")
    cols_dist = ["partition", "trip_id", "n_obs", "n_route_arcs", "gps_to_route_dist_median_m", "gps_to_route_dist_p95_m", "gps_to_route_dist_max_m", "pct_gps_within_50m_route"]
    print(df.sort_values("gps_to_route_dist_max_m", ascending=False)[cols_dist].head(10).to_string(index=False))

    print("\n=== Trips with Disconnected Reference Routes (if any) ===")
    disc = df[df["route_n_disconnected"] > 0]
    if len(disc) > 0:
        print(disc[["partition", "trip_id", "n_route_arcs", "route_n_disconnected", "route_continuity_pct"]].to_string(index=False))
    else:
        print("None! All reference routes are 100% connected.")

    print("\n=== Geographic Regions (Center Lon/Lat Distribution) ===")
    print(df.groupby("partition")[["lon_center", "lat_center"]].describe().to_string())

    print(f"\nCompleted EDA in {time.time() - t0:.2f}s. Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
