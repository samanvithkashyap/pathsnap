"""
Export Final Self-Contained Inference Artifacts for PathSnap Deployment (Laptop 1).

Packages:
- artifacts/tuned_xgboost.joblib   : CPU-portable Tuned XGBoost classifier + feature list + metadata
- artifacts/pipeline_config.json   : Candidate generation, feature list, HMM config & benchmark summary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
TUNING_DIR = PROJECT_ROOT / "outputs" / "tuning"
HMM_DIR = PROJECT_ROOT / "outputs" / "hmm"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features import FEATURE_COLUMNS
from prepare_features import FALLBACK_K, MAX_K, RADIUS_M


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load tuned XGBoost bundle and guarantee CPU-portable configuration
    bundle = joblib.load(TUNING_DIR / "tuned_xgboost.joblib")
    clf = bundle["model"]
    clf.set_params(device="cpu")
    clf.get_booster().set_param({"device": "cpu"})

    out_model = ARTIFACTS_DIR / "tuned_xgboost.joblib"
    joblib.dump(
        {
            "model": clf,
            "model_key": "xgboost_tuned",
            "display_name": "XGBoost (Optuna Tuned)",
            "features": list(FEATURE_COLUMNS),
            "best_params": bundle["best_params"],
            "random_seed": bundle["random_seed"],
        },
        out_model,
        compress=3,
    )

    with open(HMM_DIR / "hmm_config.json") as f:
        hmm_cfg = json.load(f)
    with open(HMM_DIR / "hmm_results.json") as f:
        hmm_res = json.load(f)
    with open(TUNING_DIR / "tuning_results.json") as f:
        tuning_res = json.load(f)

    pipeline_cfg = {
        "system": "PathSnap: Distributed Map-Matching & Route Reconstruction",
        "model_artifact": "tuned_xgboost.joblib",
        "model_name": "XGBoost (Optuna Tuned) + HMM/Viterbi",
        "candidate_generation": {
            "radius_m": RADIUS_M,
            "max_k": MAX_K,
            "fallback_k": FALLBACK_K,
        },
        "features": list(FEATURE_COLUMNS),
        "n_features": len(FEATURE_COLUMNS),
        "xgboost_best_params": bundle["best_params"],
        "hmm_config": hmm_cfg,
        "student_test_benchmarks": {
            "classifier_only": tuning_res["untuned_vs_tuned"]["tuned_xgboost"],
            "classifier_plus_hmm": hmm_res["student_test_10_trips"]["classifier_plus_hmm"],
        },
    }

    out_cfg = ARTIFACTS_DIR / "pipeline_config.json"
    with open(out_cfg, "w") as f:
        json.dump(pipeline_cfg, f, indent=2)

    print(f"Exported final model  -> {out_model} ({out_model.stat().st_size / 1024:.1f} KB)")
    print(f"Exported pipeline cfg -> {out_cfg}")


if __name__ == "__main__":
    main()
