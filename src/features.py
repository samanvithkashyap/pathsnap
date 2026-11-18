"""
Candidate Generation, Monotonic Ground-Truth Alignment, and Feature Engineering for PathSnap.

All features are computed strictly from (.track, .nodes, .arcs) with zero ground-truth leakage.
Ground-truth alignment (.route -> per-observation arc_id) is strictly separated and only invoked
when include_labels=True during training/evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shapely

from data_loader import LocalMetricProjector, TripData

FEATURE_COLUMNS: List[str] = [
    # Spatial / geometric features (GPS -> candidate arc)
    "perp_dist_m",
    "log_perp_dist",
    "dist_rank",
    "dist_diff_from_best",
    "dist_ratio_to_best",
    "proj_frac",
    "is_interior_proj",
    "dist_to_endpoints_min",
    # Heading / directional alignment features (critical for bidirectional roads)
    "cos_heading_diff",
    "sin_heading_diff",
    "abs_heading_diff_deg",
    "cos_heading_diff_prev",
    "cos_heading_diff_next",
    "cos_heading_diff_wide",
    "heading_rank",
    # Kinematic & temporal context features
    "speed_mps",
    "is_moving",
    "log_dt_prev",
    "log_dt_next",
    "turn_rate_deg",
    # Multi-point local trajectory & topological consistency features
    "prev_pt_dist_m",
    "next_pt_dist_m",
    "mean_3pt_dist_m",
    "proj_progress_m",
    "has_prev_connected_cand",
    "has_next_connected_cand",
    "min_prev_connected_dist_m",
    "min_next_connected_dist_m",
    # Road-network topology features
    "log_arc_length",
    "is_bidirectional",
    "from_node_deg",
    "to_node_deg",
    "num_candidates",
]


def _wrap_angle_rad(diff: np.ndarray) -> np.ndarray:
    """Wrap angle difference to [-pi, pi]."""
    return (diff + np.pi) % (2.0 * np.pi) - np.pi


def compute_track_kinematics(
    tx: np.ndarray,
    ty: np.ndarray,
    ts: np.ndarray,
    min_disp_m: float = 4.0,
) -> Dict[str, np.ndarray]:
    """
    Compute robust local headings, speeds, time gaps, and turn rates along a GPS track.

    Because stationary or slow-moving GPS tracks exhibit ~5-7 m random jitter,
    the primary heading expands its lookback/lookahead window until net displacement
    reaches >= min_disp_m (up to +/- 8 steps), preventing jitter from corrupting bearings.
    """
    n = len(tx)
    dt_prev = np.ones(n, dtype=np.float64)
    dt_next = np.ones(n, dtype=np.float64)
    if n >= 2:
        dts = np.maximum(np.diff(ts), 1e-3)
        dt_prev[1:] = dts
        dt_prev[0] = dts[0]
        dt_next[:-1] = dts
        dt_next[-1] = dts[-1]

    # Backward, forward, and wide (+/-3) headings
    idx_prev = np.maximum(np.arange(n) - 1, 0)
    idx_next = np.minimum(np.arange(n) + 1, n - 1)
    idx_w_prev = np.maximum(np.arange(n) - 3, 0)
    idx_w_next = np.minimum(np.arange(n) + 3, n - 1)

    dx_prev = tx - tx[idx_prev]
    dy_prev = ty - ty[idx_prev]
    dx_next = tx[idx_next] - tx
    dy_next = ty[idx_next] - ty
    dx_wide = tx[idx_w_next] - tx[idx_w_prev]
    dy_wide = ty[idx_w_next] - ty[idx_w_prev]

    # Fix boundary zero vectors
    if n >= 2:
        dx_prev[0], dy_prev[0] = dx_next[0], dy_next[0]
        dx_next[-1], dy_next[-1] = dx_prev[-1], dy_prev[-1]

    heading_prev = np.arctan2(dy_prev, dx_prev)
    heading_next = np.arctan2(dy_next, dx_next)
    heading_wide = np.arctan2(dy_wide, dx_wide)

    # Adaptive central heading: expand half-window k=1..8 until net displacement >= min_disp_m
    dx_main = tx[idx_next] - tx[idx_prev]
    dy_main = ty[idx_next] - ty[idx_prev]
    disp_main = np.hypot(dx_main, dy_main)

    for k in (2, 3, 5, 8):
        need = disp_main < min_disp_m
        if not np.any(need):
            break
        ip = np.maximum(np.arange(n) - k, 0)
        in_ = np.minimum(np.arange(n) + k, n - 1)
        dx_k = tx[in_] - tx[ip]
        dy_k = ty[in_] - ty[ip]
        disp_k = np.hypot(dx_k, dy_k)
        use = need & (disp_k > disp_main)
        dx_main = np.where(use, dx_k, dx_main)
        dy_main = np.where(use, dy_k, dy_main)
        disp_main = np.where(use, disp_k, disp_main)

    heading_main = np.arctan2(dy_main, dx_main)
    is_moving = (disp_main >= min_disp_m).astype(np.float64)

    # Smoothed speed over +/-1 step (clamped to physically plausible [0, 60] m/s)
    dt_span = np.maximum(ts[idx_next] - ts[idx_prev], 1e-3)
    disp_1 = np.hypot(tx[idx_next] - tx[idx_prev], ty[idx_next] - ty[idx_prev])
    speed_mps = np.clip(disp_1 / dt_span, 0.0, 60.0)

    # Turn rate between backward and forward headings (degrees)
    turn_rate_deg = np.rad2deg(np.abs(_wrap_angle_rad(heading_next - heading_prev)))

    return {
        "heading_main": heading_main,
        "heading_prev": heading_prev,
        "heading_next": heading_next,
        "heading_wide": heading_wide,
        "speed_mps": speed_mps,
        "is_moving": is_moving,
        "dt_prev": dt_prev,
        "dt_next": dt_next,
        "turn_rate_deg": turn_rate_deg,
    }


def _batch_point_to_segment_dist(
    px: np.ndarray,
    py: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized distance & projection fraction t in [0, 1] for paired arrays
    (px[i], py[i]) against segment (ax[i], ay[i]) -> (bx[i], by[i]).
    """
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    safe_len_sq = np.where(seg_len_sq > 1e-12, seg_len_sq, 1.0)
    t = ((px - ax) * dx + (py - ay) * dy) / safe_len_sq
    t = np.where(seg_len_sq > 1e-12, np.clip(t, 0.0, 1.0), 0.0)
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    dist = np.hypot(px - proj_x, py - proj_y)
    return dist, t


def align_track_to_route(
    trip: TripData,
    proj: LocalMetricProjector,
    kinematics: Dict[str, np.ndarray],
    step_penalty_m: float = 0.05,
    heading_weight_m: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align GPS observations track[0..T-1] chronologically to reference route[0..M-1]
    using globally optimal monotonic Dynamic Programming (non-decreasing route index).

    Objective minimized over all non-decreasing paths 0 <= m_0 <= m_1 <= ... <= m_{T-1} < M:
        sum_t [ perp_dist(p_t, route[m_t]) + heading_weight_m * is_moving_t * (1 - cos(theta_t - phi_{m_t})) ]
        + sum_{t>=1} step_penalty_m * (m_t - m_{t-1})

    Using prefix-minimum decomposition, each time step runs in O(M) vectorized numpy.

    Returns
    -------
    gt_arc_ids : np.ndarray of shape (T,), int64
        Ground-truth arc_id aligned to each GPS observation.
    gt_route_pos : np.ndarray of shape (T,), int64
        Index m in [0, M-1] along trip.route.
    gt_perp_dist_m : np.ndarray of shape (T,), float64
        Perpendicular distance (m) from each GPS point to its aligned reference arc.
    """
    if trip.route is None or len(trip.route) == 0:
        raise ValueError(f"Trip {trip.trip_id} has no reference route for alignment.")

    tx, ty = proj.to_xy(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy())
    nxy = proj.to_xy_array(trip.nodes)
    r = trip.route
    u = trip.arcs[r, 0]
    v = trip.arcs[r, 1]
    ax, ay = nxy[u, 0], nxy[u, 1]
    bx, by = nxy[v, 0], nxy[v, 1]
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    safe_len_sq = np.where(seg_len_sq > 1e-12, seg_len_sq, 1.0)
    arc_heading = np.arctan2(dy, dx)

    T = len(tx)
    M = len(r)
    m_idx = np.arange(M, dtype=np.float64)
    step_offset = step_penalty_m * m_idx

    heading_main = kinematics["heading_main"]
    is_moving = kinematics["is_moving"]

    # Backpointer table (T, M) stores best previous route index m' <= m
    backptr = np.empty((T, M), dtype=np.int32)
    dp_prev = np.zeros(M, dtype=np.float64)
    dist_rows = np.empty((T, M), dtype=np.float32)

    for t in range(T):
        # Perpendicular distance from p_t to all M route arcs
        proj_t = ((tx[t] - ax) * dx + (ty[t] - ay) * dy) / safe_len_sq
        proj_t = np.where(seg_len_sq > 1e-12, np.clip(proj_t, 0.0, 1.0), 0.0)
        d = np.hypot(tx[t] - (ax + proj_t * dx), ty[t] - (ay + proj_t * dy))
        dist_rows[t] = d

        # Mild heading tie-breaker when vehicle is moving (distinguishes u->v vs v->u)
        if is_moving[t] > 0.5:
            h_cost = heading_weight_m * (1.0 - np.cos(heading_main[t] - arc_heading))
            emit = d + h_cost
        else:
            emit = d

        if t == 0:
            dp_cur = emit
            backptr[0] = np.arange(M, dtype=np.int32)
        else:
            # min_{m' <= m} (dp_prev[m'] + step_penalty * (m - m'))
            # = step_penalty * m + min_{m' <= m} (dp_prev[m'] - step_penalty * m')
            shifted = dp_prev - step_offset
            # Prefix min and prefix argmin in O(M)
            pref_min = np.minimum.accumulate(shifted)
            # Track where new strict minimum occurs
            is_new_min = np.empty(M, dtype=bool)
            is_new_min[0] = True
            is_new_min[1:] = pref_min[1:] < pref_min[:-1]
            idx_arr = np.where(is_new_min, np.arange(M, dtype=np.int32), -1)
            pref_argmin = np.maximum.accumulate(idx_arr)

            dp_cur = emit + step_offset + pref_min
            backptr[t] = pref_argmin

        dp_prev = dp_cur

    # Backtrack optimal monotonic alignment
    gt_route_pos = np.empty(T, dtype=np.int64)
    gt_route_pos[T - 1] = int(np.argmin(dp_prev))
    for t in range(T - 2, -1, -1):
        gt_route_pos[t] = backptr[t + 1, gt_route_pos[t + 1]]

    gt_arc_ids = r[gt_route_pos].astype(np.int64)
    gt_perp_dist_m = dist_rows[np.arange(T), gt_route_pos].astype(np.float64)
    return gt_arc_ids, gt_route_pos, gt_perp_dist_m


@dataclass
class TripCandidateBundle:
    trip_id: str
    partition: str
    n_obs: int
    df: pd.DataFrame  # candidate rows with obs_idx, arc_id, features, and optional label
    gt_arc_ids: Optional[np.ndarray]  # shape (n_obs,) if include_labels
    gt_perp_dist_m: Optional[np.ndarray]  # shape (n_obs,) if include_labels
    n_covered_obs: int  # observations whose gt_arc_id appears in candidate set


def extract_trip_candidates_and_features(
    trip: TripData,
    radius_m: float = 60.0,
    max_k: int = 16,
    fallback_k: int = 6,
    include_labels: bool = True,
) -> TripCandidateBundle:
    """
    Generate candidate road segments per GPS observation using exact Shapely STRtree
    and engineer all spatial, directional, kinematic, local-trajectory, and topological features.
    """
    proj = LocalMetricProjector.from_trip(trip)
    tx, ty = proj.to_xy(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy())
    ts = trip.track["timestamp"].to_numpy(dtype=np.float64)
    n_obs = len(tx)

    nxy = proj.to_xy_array(trip.nodes)
    u = trip.arcs[:, 0]
    v = trip.arcs[:, 1]
    n_nodes = len(trip.nodes)
    n_arcs = len(trip.arcs)

    ax, ay = nxy[u, 0], nxy[u, 1]
    bx, by = nxy[v, 0], nxy[v, 1]
    arc_dx = bx - ax
    arc_dy = by - ay
    arc_len = np.hypot(arc_dx, arc_dy)
    arc_heading = np.arctan2(arc_dy, arc_dx)

    # Precompute node degrees and bidirectionality flags (vectorized / fast)
    in_deg = np.bincount(v, minlength=n_nodes)
    out_deg = np.bincount(u, minlength=n_nodes)
    tot_deg = (in_deg + out_deg).astype(np.float32)

    # Bidirectionality: check if (v, u) exists using packed int64 keys
    keys_fwd = u.astype(np.int64) * n_nodes + v.astype(np.int64)
    keys_rev = v.astype(np.int64) * n_nodes + u.astype(np.int64)
    key_set = set(keys_fwd.tolist())
    is_bidir = np.fromiter((k in key_set for k in keys_rev.tolist()), dtype=np.float32, count=n_arcs)

    # Track kinematics
    kin = compute_track_kinematics(tx, ty, ts)

    # Optional ground-truth alignment (strictly never used in features!)
    gt_arc_ids: Optional[np.ndarray] = None
    gt_perp_dist_m: Optional[np.ndarray] = None
    if include_labels and trip.route is not None and len(trip.route) > 0:
        gt_arc_ids, _, gt_perp_dist_m = align_track_to_route(trip, proj, kin)

    # Build exact Shapely STRtree over all road arcs
    arc_coords = np.stack([nxy[u], nxy[v]], axis=1)  # (N_arcs, 2, 2)
    lines = shapely.linestrings(arc_coords)
    tree = shapely.STRtree(lines)
    pts = shapely.points(tx, ty)

    # Vectorized spatial query within radius_m
    obs_q, arc_q = tree.query(pts, predicate="dwithin", distance=radius_m)

    # Compute exact perpendicular distances for all query matches
    d_q, t_q = _batch_point_to_segment_dist(
        tx[obs_q], ty[obs_q], ax[arc_q], ay[arc_q], bx[arc_q], by[arc_q]
    )

    # Group matches by obs_idx, sort by exact perpendicular distance, keep top max_k
    # Sort lexicographically by (obs_idx, dist)
    order = np.lexsort((d_q, obs_q))
    obs_q = obs_q[order]
    arc_q = arc_q[order]
    d_q = d_q[order]
    t_q = t_q[order]

    # Identify slice boundaries per observation
    counts = np.bincount(obs_q, minlength=n_obs)
    starts = np.zeros(n_obs + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])

    # Collect trimmed candidate arrays; use nearest fallback for observations with 0 hits
    obs_list: List[np.ndarray] = []
    arc_list: List[np.ndarray] = []
    dist_list: List[np.ndarray] = []
    proj_list: List[np.ndarray] = []

    missing_obs = np.where(counts == 0)[0]
    fallback_map: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if len(missing_obs) > 0:
        # Query nearest arcs via expanding radius or tree.query_nearest
        for oi in missing_obs:
            # Progressively search wider radius (150m -> 400m -> 1200m -> nearest)
            found = np.empty(0, dtype=np.int64)
            for r_fb in (150.0, 400.0, 1200.0):
                hits = tree.query(pts[oi], predicate="dwithin", distance=r_fb)
                if len(hits) >= fallback_k:
                    found = np.asarray(hits, dtype=np.int64)
                    break
            if len(found) == 0:
                nearest_idx = tree.query_nearest(pts[oi], all_matches=True)
                found = np.asarray(nearest_idx, dtype=np.int64).ravel()
            df_fb, tf_fb = _batch_point_to_segment_dist(
                np.full(len(found), tx[oi]),
                np.full(len(found), ty[oi]),
                ax[found],
                ay[found],
                bx[found],
                by[found],
            )
            ord_fb = np.argsort(df_fb)[:fallback_k]
            fallback_map[int(oi)] = (found[ord_fb], df_fb[ord_fb], tf_fb[ord_fb])

    # Fast trim mask for non-empty observations
    if len(obs_q) > 0:
        rank_in_obs = np.arange(len(obs_q), dtype=np.int64) - np.repeat(starts[:-1], counts)
        keep_mask = rank_in_obs < max_k
        obs_kept = obs_q[keep_mask]
        arc_kept = arc_q[keep_mask]
        dist_kept = d_q[keep_mask]
        proj_kept = t_q[keep_mask]
    else:
        obs_kept = np.empty(0, dtype=np.int64)
        arc_kept = np.empty(0, dtype=np.int64)
        dist_kept = np.empty(0, dtype=np.float64)
        proj_kept = np.empty(0, dtype=np.float64)

    if len(fallback_map) > 0:
        fb_obs = []
        fb_arc = []
        fb_dist = []
        fb_proj = []
        for oi, (fa, fd, ft) in fallback_map.items():
            fb_obs.append(np.full(len(fa), oi, dtype=np.int64))
            fb_arc.append(fa)
            fb_dist.append(fd)
            fb_proj.append(ft)
        obs_all = np.concatenate([obs_kept, np.concatenate(fb_obs)])
        arc_all = np.concatenate([arc_kept, np.concatenate(fb_arc)])
        dist_all = np.concatenate([dist_kept, np.concatenate(fb_dist)])
        proj_all = np.concatenate([proj_kept, np.concatenate(fb_proj)])
        # Re-sort by (obs_idx, dist)
        ord2 = np.lexsort((dist_all, obs_all))
        obs_all = obs_all[ord2]
        arc_all = arc_all[ord2]
        dist_all = dist_all[ord2]
        proj_all = proj_all[ord2]
    else:
        obs_all = obs_kept
        arc_all = arc_kept
        dist_all = dist_kept
        proj_all = proj_kept

    # Final per-observation slices after trimming + fallback
    cand_counts = np.bincount(obs_all, minlength=n_obs)
    cand_starts = np.zeros(n_obs + 1, dtype=np.int64)
    np.cumsum(cand_counts, out=cand_starts[1:])
    n_rows = len(obs_all)

    # --- Spatial / geometric features ---
    dist_rank = (np.arange(n_rows, dtype=np.int64) - np.repeat(cand_starts[:-1], cand_counts)).astype(np.float32)
    min_dist_per_obs = dist_all[cand_starts[:-1]]  # first candidate per obs is closest
    min_dist_rep = np.repeat(min_dist_per_obs, cand_counts)
    dist_diff_from_best = dist_all - min_dist_rep
    dist_ratio_to_best = (dist_all + 1.0) / (min_dist_rep + 1.0)
    is_interior_proj = ((proj_all > 1e-4) & (proj_all < 1.0 - 1e-4)).astype(np.float32)

    d_to_a = np.hypot(tx[obs_all] - ax[arc_all], ty[obs_all] - ay[arc_all])
    d_to_b = np.hypot(tx[obs_all] - bx[arc_all], ty[obs_all] - by[arc_all])
    dist_to_endpoints_min = np.minimum(d_to_a, d_to_b)

    # --- Heading / directional alignment features ---
    cand_arc_h = arc_heading[arc_all]
    h_diff_main = _wrap_angle_rad(kin["heading_main"][obs_all] - cand_arc_h)
    cos_h_main = np.cos(h_diff_main)
    sin_h_main = np.sin(h_diff_main)
    abs_h_deg = np.rad2deg(np.abs(h_diff_main))
    cos_h_prev = np.cos(_wrap_angle_rad(kin["heading_prev"][obs_all] - cand_arc_h))
    cos_h_next = np.cos(_wrap_angle_rad(kin["heading_next"][obs_all] - cand_arc_h))
    cos_h_wide = np.cos(_wrap_angle_rad(kin["heading_wide"][obs_all] - cand_arc_h))

    # Heading rank within each observation
    heading_rank = np.empty(n_rows, dtype=np.float32)
    for oi in range(n_obs):
        s, e = cand_starts[oi], cand_starts[oi + 1]
        if e > s:
            heading_rank[s:e] = np.argsort(np.argsort(abs_h_deg[s:e]))

    # --- Multi-point local trajectory support ---
    obs_prev = np.maximum(obs_all - 1, 0)
    obs_next = np.minimum(obs_all + 1, n_obs - 1)
    prev_pt_dist, prev_proj = _batch_point_to_segment_dist(
        tx[obs_prev], ty[obs_prev], ax[arc_all], ay[arc_all], bx[arc_all], by[arc_all]
    )
    next_pt_dist, next_proj = _batch_point_to_segment_dist(
        tx[obs_next], ty[obs_next], ax[arc_all], ay[arc_all], bx[arc_all], by[arc_all]
    )
    mean_3pt_dist = (prev_pt_dist + dist_all + next_pt_dist) / 3.0
    proj_progress_m = (next_proj - prev_proj) * arc_len[arc_all]

    # --- Local topological connectivity with candidates at t-1 and t+1 ---
    cand_u = u[arc_all]
    cand_v = v[arc_all]
    has_prev_conn = np.zeros(n_rows, dtype=np.float32)
    has_next_conn = np.zeros(n_rows, dtype=np.float32)
    min_prev_conn_dist = np.full(n_rows, radius_m, dtype=np.float32)
    min_next_conn_dist = np.full(n_rows, radius_m, dtype=np.float32)

    for oi in range(n_obs):
        s, e = cand_starts[oi], cand_starts[oi + 1]
        if e <= s:
            continue
        cur_arcs = arc_all[s:e]
        cur_u = cand_u[s:e]
        cur_v = cand_v[s:e]

        if oi > 0:
            ps, pe = cand_starts[oi - 1], cand_starts[oi]
            p_arcs = arc_all[ps:pe]
            p_v = cand_v[ps:pe]
            p_d = dist_all[ps:pe]
            # Connected if same arc OR prev_arc.to_node == cur_arc.from_node
            conn = (cur_arcs[:, None] == p_arcs[None, :]) | (cur_u[:, None] == p_v[None, :])
            any_c = np.any(conn, axis=1)
            has_prev_conn[s:e] = any_c.astype(np.float32)
            if np.any(any_c):
                masked_d = np.where(conn, p_d[None, :], np.inf)
                md = np.min(masked_d, axis=1)
                min_prev_conn_dist[s:e] = np.where(any_c, np.minimum(md, radius_m), radius_m)
        else:
            has_prev_conn[s:e] = 1.0
            min_prev_conn_dist[s:e] = dist_all[s:e]

        if oi < n_obs - 1:
            ns, ne = cand_starts[oi + 1], cand_starts[oi + 2]
            n_arcs_next = arc_all[ns:ne]
            n_u = cand_u[ns:ne]
            n_d = dist_all[ns:ne]
            # Connected if same arc OR cur_arc.to_node == next_arc.from_node
            conn = (cur_arcs[:, None] == n_arcs_next[None, :]) | (cur_v[:, None] == n_u[None, :])
            any_c = np.any(conn, axis=1)
            has_next_conn[s:e] = any_c.astype(np.float32)
            if np.any(any_c):
                masked_d = np.where(conn, n_d[None, :], np.inf)
                md = np.min(masked_d, axis=1)
                min_next_conn_dist[s:e] = np.where(any_c, np.minimum(md, radius_m), radius_m)
        else:
            has_next_conn[s:e] = 1.0
            min_next_conn_dist[s:e] = dist_all[s:e]

    # Assemble DataFrame
    data: Dict[str, np.ndarray] = {
        "obs_idx": obs_all.astype(np.int32),
        "arc_id": arc_all.astype(np.int32),
        "perp_dist_m": dist_all.astype(np.float32),
        "log_perp_dist": np.log1p(dist_all).astype(np.float32),
        "dist_rank": dist_rank,
        "dist_diff_from_best": dist_diff_from_best.astype(np.float32),
        "dist_ratio_to_best": dist_ratio_to_best.astype(np.float32),
        "proj_frac": proj_all.astype(np.float32),
        "is_interior_proj": is_interior_proj,
        "dist_to_endpoints_min": dist_to_endpoints_min.astype(np.float32),
        "cos_heading_diff": cos_h_main.astype(np.float32),
        "sin_heading_diff": sin_h_main.astype(np.float32),
        "abs_heading_diff_deg": abs_h_deg.astype(np.float32),
        "cos_heading_diff_prev": cos_h_prev.astype(np.float32),
        "cos_heading_diff_next": cos_h_next.astype(np.float32),
        "cos_heading_diff_wide": cos_h_wide.astype(np.float32),
        "heading_rank": heading_rank,
        "speed_mps": kin["speed_mps"][obs_all].astype(np.float32),
        "is_moving": kin["is_moving"][obs_all].astype(np.float32),
        "log_dt_prev": np.log1p(kin["dt_prev"][obs_all]).astype(np.float32),
        "log_dt_next": np.log1p(kin["dt_next"][obs_all]).astype(np.float32),
        "turn_rate_deg": kin["turn_rate_deg"][obs_all].astype(np.float32),
        "prev_pt_dist_m": prev_pt_dist.astype(np.float32),
        "next_pt_dist_m": next_pt_dist.astype(np.float32),
        "mean_3pt_dist_m": mean_3pt_dist.astype(np.float32),
        "proj_progress_m": np.clip(proj_progress_m, -200.0, 200.0).astype(np.float32),
        "has_prev_connected_cand": has_prev_conn,
        "has_next_connected_cand": has_next_conn,
        "min_prev_connected_dist_m": min_prev_conn_dist,
        "min_next_connected_dist_m": min_next_conn_dist,
        "log_arc_length": np.log1p(arc_len[arc_all]).astype(np.float32),
        "is_bidirectional": is_bidir[arc_all],
        "from_node_deg": tot_deg[cand_u],
        "to_node_deg": tot_deg[cand_v],
        "num_candidates": np.repeat(cand_counts, cand_counts).astype(np.float32),
    }

    n_covered = 0
    if gt_arc_ids is not None:
        labels = (arc_all == gt_arc_ids[obs_all]).astype(np.int8)
        data["label"] = labels
        # Count covered observations (those with at least one positive candidate)
        covered_mask = np.bincount(obs_all, weights=labels, minlength=n_obs) > 0
        n_covered = int(np.sum(covered_mask))

    df = pd.DataFrame(data)
    df.insert(0, "trip_id", trip.trip_id)

    return TripCandidateBundle(
        trip_id=trip.trip_id,
        partition=trip.partition,
        n_obs=n_obs,
        df=df,
        gt_arc_ids=gt_arc_ids,
        gt_perp_dist_m=gt_perp_dist_m,
        n_covered_obs=n_covered,
    )
