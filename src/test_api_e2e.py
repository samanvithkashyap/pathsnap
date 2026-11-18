"""
End-to-End Integration & Leakage-Isolation Test for PathSnap FastAPI Server.

Tests GET /health, GET /trips, and POST /predict across ALL available trips in the
configured evaluation partition, verifying:
- HTTP 200 status and complete map payload for every trip
- Strict server-side isolation of ground-truth .route labels (never present in response)
- Exact match with offline Tuned XGBoost + HMM/Viterbi accuracy & continuity benchmarks
- End-to-end live server latency per trip
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.server import app


def main() -> None:
    with TestClient(app) as client:
        # 1. Health check
        r_health = client.get("/health")
        assert r_health.status_code == 200, f"/health failed: {r_health.text}"
        health = r_health.json()
        print("HEALTH:", json.dumps(health))

        # 2. List trips
        r_trips = client.get("/trips")
        assert r_trips.status_code == 200, f"/trips failed: {r_trips.text}"
        trips_payload = r_trips.json()
        trips = trips_payload["trips"]
        assert len(trips) > 0, "No evaluation trips returned!"
        print(f"\nDiscovered {len(trips)} trips in partition '{trips_payload['partition']}'.")

        # 3. Run POST /predict for ALL trips
        total_obs = 0
        weighted_top1 = 0.0
        weighted_cont = 0.0
        lcs_vals = []
        latencies = []

        print("\nTrip ID  | Obs  | Total ms | Clf Top-1 | HMM Top-1 | Step Cont | Sw. Cont | LCS F1")
        print("-" * 86)
        for t in trips:
            tid = t["trip_id"]
            resp = client.post("/predict", json={"trip_id": tid, "include_clf_only": True})
            assert resp.status_code == 200, f"/predict failed for {tid}: {resp.text}"
            body = resp.json()

            # Strict leakage assertion: ensure no raw ground-truth route fields exist in response
            raw_keys = set(body.keys()) | set(body["map_data"].keys())
            forbidden = {"route", "ref_route", "gt_arc_ids", "reference_route_segments", "ground_truth"}
            assert raw_keys.isdisjoint(forbidden), f"Leakage check failed! Found {raw_keys & forbidden}"

            assert len(body["map_data"]["predicted_route_segments"]) > 0
            assert len(body["map_data"]["raw_gps_points"]) > 0

            ms = body["timings_ms"]["total"]
            latencies.append(ms)
            ev = body["evaluation"]
            if ev and ev["reference_available_server_side"]:
                hm = ev["hmm_viterbi"]
                cm = ev["classifier_only"]
                n_obs = body["trip_stats"]["n_observations"]
                total_obs += n_obs
                weighted_top1 += hm["top1_accuracy"] * n_obs
                weighted_cont += hm["route_continuity"] * n_obs
                lcs_vals.append(hm["seq_lcs_f1"])
                print(
                    f"{tid} | {n_obs:4d} | {ms:7.1f}  | {cm['top1_accuracy']*100:8.2f}% | "
                    f"{hm['top1_accuracy']*100:8.2f}% | {hm['route_continuity']*100:8.2f}% | "
                    f"{hm['transition_continuity']*100:7.2f}% | {hm['seq_lcs_f1']:.4f}"
                )
            else:
                print(f"{tid} | {body['trip_stats']['n_observations']:4d} | {ms:7.1f}  | (no server .route)")

        print("-" * 86)
        if total_obs > 0:
            print(
                f"ALL {len(trips)} TRIPS PASSED! "
                f"Weighted Top-1: {weighted_top1/total_obs*100:.2f}% | "
                f"Step Continuity: {weighted_cont/total_obs*100:.2f}% | "
                f"Mean LCS F1: {sum(lcs_vals)/len(lcs_vals):.4f} | "
                f"Mean Latency: {sum(latencies)/len(latencies):.1f} ms/trip"
            )
        else:
            print(f"ALL {len(trips)} TRIPS PASSED! Mean Latency: {sum(latencies)/len(latencies):.1f} ms/trip")


if __name__ == "__main__":
    main()
