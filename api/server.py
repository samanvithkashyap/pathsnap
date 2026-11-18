"""
PathSnap FastAPI Backend Server (Laptop 1).

Hosts:
- Trained Optuna-Tuned XGBoost candidate classifier (artifacts/tuned_xgboost.joblib)
- HMM + Viterbi road-network route reconstruction decoder (src/hmm_viterbi.py)
- Road-network & evaluation trip discovery (PATHSNAP_EVAL_DIR -> dataset/evaluator_test -> dataset/student_test)
- Strictly server-side evaluation against internal .route reference files (never exposed to model or UI)

Run on Laptop 1:
    uvicorn api.server:app --host 0.0.0.0 --port 8000
    # Or with an explicit evaluation folder:
    PATHSNAP_EVAL_DIR=/path/to/evaluator_test uvicorn api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import LocalMetricProjector, load_trip
from features import align_track_to_route, compute_track_kinematics, extract_trip_candidates_and_features
from hmm_viterbi import HMMConfig, RoadGraphIndex, viterbi_decode_trip
from metrics import deduplicate_sequence, evaluate_trip_predictions, select_top1_arcs

# Configure clear terminal logging for live evaluator demonstration on Laptop 1
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pathsnap.api")


class PredictRequest(BaseModel):
    trip_id: str = Field(..., description="Trip directory ID (e.g., '00000000')")
    include_clf_only: bool = Field(
        default=True,
        description="Also return classifier-only route geometry & metrics for visual comparison",
    )


class AppState:
    def __init__(self) -> None:
        self.model: Any = None
        self.features: List[str] = []
        self.config: Dict[str, Any] = {}
        self.hmm_cfg: HMMConfig = HMMConfig()
        self.eval_dir: Path = PROJECT_ROOT / "dataset" / "student_test"
        self.partition_name: str = "student_test"
        self.trip_dirs: Dict[str, Path] = {}
        self.trip_summaries: List[Dict[str, Any]] = []
        self.request_count: int = 0


STATE = AppState()


def resolve_eval_dir() -> Path:
    """Resolve active evaluation directory: PATHSNAP_EVAL_DIR > evaluator_test > student_test."""
    env_dir = os.environ.get("PATHSNAP_EVAL_DIR")
    if env_dir:
        p = Path(env_dir).expanduser().resolve()
        if p.exists() and p.is_dir():
            return p
        logger.warning("PATHSNAP_EVAL_DIR=%s does not exist; falling back to default dataset paths.", env_dir)

    for candidate in (
        PROJECT_ROOT / "dataset" / "evaluator_test",
        PROJECT_ROOT / "dataset" / "student_test",
    ):
        if candidate.exists() and any(d.is_dir() for d in candidate.iterdir()):
            return candidate
    raise RuntimeError("No valid evaluation trip directory found (checked evaluator_test and student_test).")


def discover_trips(eval_dir: Path, partition_name: str) -> tuple[Dict[str, Path], List[Dict[str, Any]]]:
    """Scan trip directories WITHOUT loading .route files."""
    trip_dirs: Dict[str, Path] = {}
    summaries: List[Dict[str, Any]] = []

    for td in sorted([p for p in eval_dir.iterdir() if p.is_dir()]):
        tid = td.name
        # Strictly load_route=False when building public trip metadata
        trip = load_trip(td, partition=partition_name, load_route=False)
        if len(trip.track) == 0 or len(trip.nodes) == 0 or len(trip.arcs) == 0:
            continue
        ts = trip.track["timestamp"].to_numpy(dtype=np.float64)
        lons = trip.track["lon"].to_numpy(dtype=np.float64)
        lats = trip.track["lat"].to_numpy(dtype=np.float64)
        trip_dirs[tid] = td
        summaries.append({
            "trip_id": tid,
            "n_observations": int(len(trip.track)),
            "n_nodes": int(len(trip.nodes)),
            "n_arcs": int(len(trip.arcs)),
            "duration_s": round(float(ts[-1] - ts[0]), 1) if len(ts) >= 2 else 0.0,
            "center_lat": round(float(np.median(lats)), 6),
            "center_lon": round(float(np.median(lons)), 6),
        })
    return trip_dirs, summaries


def _arcs_to_latlon_segments(arc_ids: np.ndarray, arcs: np.ndarray, nodes: np.ndarray) -> List[List[List[float]]]:
    """Convert arc IDs to [[lat1, lon1], [lat2, lon2]] segments for Leaflet PolyLine rendering."""
    valid = arc_ids[arc_ids >= 0]
    if len(valid) == 0:
        return []
    u = arcs[valid, 0]
    v = arcs[valid, 1]
    return [
        [
            [round(float(nodes[u_i, 1]), 6), round(float(nodes[u_i, 0]), 6)],
            [round(float(nodes[v_i, 1]), 6), round(float(nodes[v_i, 0]), 6)],
        ]
        for u_i, v_i in zip(u.tolist(), v.tolist())
    ]


def _extract_nearby_road_segments(
    lons: np.ndarray,
    lats: np.ndarray,
    arcs: np.ndarray,
    nodes: np.ndarray,
    max_segments: int = 1800,
) -> List[List[List[float]]]:
    """Return a lightweight subset of road-network geometry around the GPS bounding box."""
    u, v = arcs[:, 0], arcs[:, 1]
    nlon, nlat = nodes[:, 0], nodes[:, 1]
    in_box = (
        (nlon[u] >= float(lons.min()) - 0.003)
        & (nlon[u] <= float(lons.max()) + 0.003)
        & (nlat[u] >= float(lats.min()) - 0.0025)
        & (nlat[u] <= float(lats.max()) + 0.0025)
    )
    arc_ids = np.where(in_box)[0]
    if len(arc_ids) > max_segments:
        arc_ids = arc_ids[:: max(1, len(arc_ids) // max_segments)]
    return _arcs_to_latlon_segments(arc_ids, arcs, nodes)


def _evaluate_server_side(
    trip_dir: Path,
    partition_name: str,
    cand_df: Any,
    pred_arcs_hmm: np.ndarray,
    pred_arcs_clf: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    Strictly server-side evaluation against internal .route file (if present).
    Returns aggregate scalar evaluation metrics ONLY — never exposes raw ground-truth
    arc IDs or reference route geometry to the client or UI.
    """
    route_file = trip_dir / f"{trip_dir.name}.route"
    if not route_file.exists() or route_file.stat().st_size == 0:
        return None

    trip_ref = load_trip(trip_dir, partition=partition_name, load_route=True)
    if trip_ref.route is None or len(trip_ref.route) == 0:
        return None

    proj = LocalMetricProjector.from_trip(trip_ref)
    tx, ty = proj.to_xy(trip_ref.track["lon"].to_numpy(), trip_ref.track["lat"].to_numpy())
    ts = trip_ref.track["timestamp"].to_numpy(dtype=np.float64)
    kin = compute_track_kinematics(tx, ty, ts)
    gt_arc_ids, _, _ = align_track_to_route(trip_ref, proj, kin)

    # Determine candidate coverage mask strictly server-side
    obs_idx = cand_df["obs_idx"].to_numpy(dtype=np.int64)
    arc_id = cand_df["arc_id"].to_numpy(dtype=np.int64)
    is_gt = (arc_id == gt_arc_ids[obs_idx]).astype(np.int8)
    covered_mask = np.bincount(obs_idx, weights=is_gt, minlength=len(gt_arc_ids)) > 0

    hmm_m = evaluate_trip_predictions(
        pred_arcs=pred_arcs_hmm,
        gt_arc_ids=gt_arc_ids,
        ref_route=trip_ref.route,
        arcs=trip_ref.arcs,
        covered_mask=covered_mask,
    )
    clf_m = evaluate_trip_predictions(
        pred_arcs=pred_arcs_clf,
        gt_arc_ids=gt_arc_ids,
        ref_route=trip_ref.route,
        arcs=trip_ref.arcs,
        covered_mask=covered_mask,
    )

    def _clean(m: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "top1_accuracy": round(float(m["top1_accuracy"]), 4),
            "top1_accuracy_covered": round(float(m["top1_accuracy_covered"]), 4),
            "candidate_miss_rate": round(float(m["candidate_miss_rate"]), 4),
            "route_continuity": round(float(m["route_continuity"]), 4),
            "transition_continuity": round(float(m["transition_continuity"]), 4),
            "seq_arc_precision": round(float(m["seq_arc_precision"]), 4),
            "seq_arc_recall": round(float(m["seq_arc_recall"]), 4),
            "seq_arc_f1": round(float(m["seq_arc_f1"]), 4),
            "seq_lcs_f1": round(float(m["seq_lcs_f1"]), 4),
            "pred_route_arc_count": int(m["pred_seq_len"]),
        }

    return {
        "reference_available_server_side": True,
        "hmm_viterbi": _clean(hmm_m),
        "classifier_only": _clean(clf_m),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    cfg_path = ARTIFACTS_DIR / "pipeline_config.json"
    model_path = ARTIFACTS_DIR / "tuned_xgboost.joblib"

    if not cfg_path.exists() or not model_path.exists():
        raise RuntimeError(f"Missing inference artifacts in {ARTIFACTS_DIR}. Run src/export_artifacts.py first.")

    with open(cfg_path) as f:
        STATE.config = json.load(f)

    bundle = joblib.load(model_path)
    STATE.model = bundle["model"]
    STATE.features = list(bundle["features"])
    STATE.hmm_cfg = HMMConfig(**STATE.config["hmm_config"])

    STATE.eval_dir = resolve_eval_dir()
    STATE.partition_name = STATE.eval_dir.name
    STATE.trip_dirs, STATE.trip_summaries = discover_trips(STATE.eval_dir, STATE.partition_name)

    logger.info("=" * 72)
    logger.info("PATHSNAP INFERENCE API READY (Laptop 1)")
    logger.info("  Model        : %s (%d features)", STATE.config["model_name"], len(STATE.features))
    logger.info("  Eval Folder  : %s (%d trips)", STATE.eval_dir, len(STATE.trip_dirs))
    logger.info("  Trip IDs     : %s", ", ".join(sorted(STATE.trip_dirs.keys())))
    logger.info("  Startup Time : %.2fs", time.time() - t0)
    logger.info("=" * 72)
    yield


app = FastAPI(
    title="PathSnap Map-Matching & Route Reconstruction API",
    description="Distributed map-matching inference backend (Tuned XGBoost + HMM/Viterbi)",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow cross-origin requests from Laptop 2's browser UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health(request: Request) -> Dict[str, Any]:
    client_ip = request.client.host if request.client else "unknown"
    logger.info("[GET /health] client=%s -> ok (%d trips ready)", client_ip, len(STATE.trip_dirs))
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": STATE.config.get("model_name"),
        "n_features": len(STATE.features),
        "partition": STATE.partition_name,
        "n_trips": len(STATE.trip_dirs),
        "requests_served": STATE.request_count,
    }


@app.get("/trips")
async def list_available_trips(request: Request) -> Dict[str, Any]:
    client_ip = request.client.host if request.client else "unknown"
    logger.info("[GET /trips] client=%s -> returning %d evaluation trips", client_ip, len(STATE.trip_summaries))
    return {
        "partition": STATE.partition_name,
        "count": len(STATE.trip_summaries),
        "trips": STATE.trip_summaries,
    }


@app.post("/predict")
async def predict_trip(req: PredictRequest, request: Request) -> Dict[str, Any]:
    client_ip = request.client.host if request.client else "unknown"
    STATE.request_count += 1
    req_id = STATE.request_count

    if req.trip_id not in STATE.trip_dirs:
        logger.warning("[REQ #%d] client=%s trip=%s -> 404 Not Found", req_id, client_ip, req.trip_id)
        raise HTTPException(status_code=404, detail=f"Trip '{req.trip_id}' not found in '{STATE.partition_name}'.")

    trip_dir = STATE.trip_dirs[req.trip_id]
    logger.info("-" * 72)
    logger.info("[REQ #%d] POST /predict | client=%s | trip_id=%s", req_id, client_ip, req.trip_id)

    t_total0 = time.perf_counter()

    # Stage 1: Load raw GPS track & road network ONLY (strictly load_route=False!)
    t0 = time.perf_counter()
    trip = load_trip(trip_dir, partition=STATE.partition_name, load_route=False)
    t_load_ms = (time.perf_counter() - t0) * 1000.0

    # Stage 2: Exact Shapely STRtree candidate generation & 33 feature extraction (include_labels=False!)
    t0 = time.perf_counter()
    cand_cfg = STATE.config["candidate_generation"]
    bundle = extract_trip_candidates_and_features(
        trip,
        radius_m=float(cand_cfg["radius_m"]),
        max_k=int(cand_cfg["max_k"]),
        fallback_k=int(cand_cfg["fallback_k"]),
        include_labels=False,
    )
    t_feat_ms = (time.perf_counter() - t0) * 1000.0

    # Stage 3: Tuned XGBoost candidate probability scoring
    t0 = time.perf_counter()
    X = bundle.df[STATE.features].to_numpy(dtype=np.float32)
    cand_probs = STATE.model.predict_proba(X)[:, 1]
    pred_arcs_clf = select_top1_arcs(bundle.df, cand_probs, bundle.n_obs)
    t_clf_ms = (time.perf_counter() - t0) * 1000.0

    # Stage 4: HMM + Viterbi road-network sequential route reconstruction
    t0 = time.perf_counter()
    graph = RoadGraphIndex(trip)
    pred_arcs_hmm, viterbi_diag = viterbi_decode_trip(
        trip_df=bundle.df,
        cand_probs=cand_probs,
        graph=graph,
        n_obs=bundle.n_obs,
        cfg=STATE.hmm_cfg,
    )
    t_hmm_ms = (time.perf_counter() - t0) * 1000.0
    t_total_ms = (time.perf_counter() - t_total0) * 1000.0

    # Deduplicated predicted route geometry & route statistics
    dedup_hmm = deduplicate_sequence(pred_arcs_hmm[pred_arcs_hmm >= 0])
    pred_route_len_km = float(np.sum(graph.arc_len[dedup_hmm]) / 1000.0) if len(dedup_hmm) > 0 else 0.0

    lons = trip.track["lon"].to_numpy(dtype=np.float64)
    lats = trip.track["lat"].to_numpy(dtype=np.float64)
    ts = trip.track["timestamp"].to_numpy(dtype=np.float64)

    # Downsample raw GPS markers slightly only if >1200 points for smooth browser rendering
    gps_step = max(1, len(lats) // 1000)
    raw_gps_pts = [
        [round(float(lats[i]), 6), round(float(lons[i]), 6), round(float(ts[i]), 1)]
        for i in range(0, len(lats), gps_step)
    ]

    hmm_segments = _arcs_to_latlon_segments(dedup_hmm, trip.arcs, trip.nodes)
    clf_segments: Optional[List[List[List[float]]]] = None
    if req.include_clf_only:
        dedup_clf = np.unique(pred_arcs_clf[pred_arcs_clf >= 0])
        clf_segments = _arcs_to_latlon_segments(dedup_clf, trip.arcs, trip.nodes)

    nearby_roads = _extract_nearby_road_segments(lons, lats, trip.arcs, trip.nodes, max_segments=1800)

    # Stage 5: Strictly server-side evaluation (never exposes .route arcs or geometry!)
    eval_metrics = _evaluate_server_side(
        trip_dir=trip_dir,
        partition_name=STATE.partition_name,
        cand_df=bundle.df,
        pred_arcs_hmm=pred_arcs_hmm,
        pred_arcs_clf=pred_arcs_clf,
    )

    # Rich live log output for Laptop 1 terminal/CMD demonstration
    if eval_metrics is not None:
        hm = eval_metrics["hmm_viterbi"]
        cm = eval_metrics["classifier_only"]
        logger.info(
            "[REQ #%d] DONE trip=%s | obs=%d cands=%d | time=%.1fms "
            "(load=%.1f feat=%.1f xgb=%.1f hmm=%.1f) | "
            "Top-1: %.2f%% (Clf: %.2f%%) | Cont: %.2f%% (Switch: %.2f%%) | LCS-F1: %.4f",
            req_id,
            req.trip_id,
            bundle.n_obs,
            len(bundle.df),
            t_total_ms,
            t_load_ms,
            t_feat_ms,
            t_clf_ms,
            t_hmm_ms,
            hm["top1_accuracy"] * 100.0,
            cm["top1_accuracy"] * 100.0,
            hm["route_continuity"] * 100.0,
            hm["transition_continuity"] * 100.0,
            hm["seq_lcs_f1"],
        )
    else:
        logger.info(
            "[REQ #%d] DONE trip=%s | obs=%d cands=%d | time=%.1fms | predicted_arcs=%d (no server .route file)",
            req_id,
            req.trip_id,
            bundle.n_obs,
            len(bundle.df),
            t_total_ms,
            len(dedup_hmm),
        )

    return {
        "request_id": req_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "trip_id": req.trip_id,
        "partition": STATE.partition_name,
        "model_name": STATE.config["model_name"],
        "trip_stats": {
            "n_observations": int(bundle.n_obs),
            "n_candidates_evaluated": int(len(bundle.df)),
            "n_road_nodes": int(len(trip.nodes)),
            "n_road_arcs": int(len(trip.arcs)),
            "duration_s": round(float(ts[-1] - ts[0]), 1) if len(ts) >= 2 else 0.0,
            "predicted_route_arcs": int(len(dedup_hmm)),
            "predicted_route_km": round(pred_route_len_km, 2),
        },
        "viterbi_diagnostics": viterbi_diag,
        "timings_ms": {
            "load": round(t_load_ms, 2),
            "features": round(t_feat_ms, 2),
            "classifier": round(t_clf_ms, 2),
            "hmm_viterbi": round(t_hmm_ms, 2),
            "total": round(t_total_ms, 2),
        },
        "map_data": {
            "bounds": [
                [round(float(lats.min()), 6), round(float(lons.min()), 6)],
                [round(float(lats.max()), 6), round(float(lons.max()), 6)],
            ],
            "start_point": [round(float(lats[0]), 6), round(float(lons[0]), 6)],
            "end_point": [round(float(lats[-1]), 6), round(float(lons[-1]), 6)],
            "raw_gps_points": raw_gps_pts,
            "predicted_route_segments": hmm_segments,
            "clf_only_route_segments": clf_segments,
            "nearby_road_segments": nearby_roads,
        },
        "evaluation": eval_metrics,
    }


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("PATHSNAP_HOST", "0.0.0.0")
    port = int(os.environ.get("PATHSNAP_PORT", "8000"))
    uvicorn.run("api.server:app", host=host, port=port, reload=False)
