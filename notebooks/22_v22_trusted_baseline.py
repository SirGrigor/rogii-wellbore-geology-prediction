"""v22 — trusted residual-GBDT baseline, re-established under the discovery-first lens.

WHY THIS EXISTS (2026-05-31, after the leaderboard reality-check):
  The live LB shows top-1 = 6.69 RMSE vs the public pack ~9.25 and our sacred ~9.16
  — a 2.5 ft BREAKAWAY. So the "9.16 is a proven ceiling" claim is FALSE: signal worth
  ~2.5 ft is demonstrably extractable and we left all of it on the table. (Contrast
  S6E5, where the top-20 sat within 0.0004 AUC = a real, benign noise floor.)
  Before chasing the 2.5 ft, we need a CLEAN, TRUSTED sacred number built the disciplined
  way (verified API, static-tested, sacred-gated) to anchor the discovery work. This is
  that anchor — NOT the medal attempt. It mirrors the proven 03_baseline_lgb.py contract.

Hypothesis: residual GBDT (model Δ=TVT−anchor) on the full feature set WITH alignment,
  evaluated ONLY on the 150-well sacred holdout (the 3-well LB is noise — it swung 0.44 ft).
  Predicted sacred RMSE ~14–15 (at/near the carry-forward floor 14.08) — prior baselines
  did NOT beat the floor on sacred; this re-confirms the honest starting point, trustworthy.
Predicted Δ vs floor: ~0 (we expect to be AT the floor; beating it is the discovery job).
Most relevant lesson: judge ONLY on sacred; high feature-gain ≠ generalizing signal.

NO porting, NO blend, NO public OOF. One model, one honest number.

Colab: set SPRINT_ACTIVE.txt -> notebooks/22_v22_trusted_baseline.py, run bootstrap.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import joblib

from src import cv, data, features, submission, train
from src.evaluate import rmse
from src.observer import Experiment

VERSION = "v22_trusted_baseline"


def main() -> None:
    t0 = time.time()
    algo = train.default_algo()
    print("=" * 70)
    print(f"{VERSION} — residual GBDT ({algo}), sacred-gated")
    print("=" * 70)

    # 1) sacred split — 150 wells never tuned on (3-well LB is noise)
    dev_wells, sacred_wells = cv.sacred_split(data.list_well_ids("train"))
    print(f"dev wells: {len(dev_wells)}   sacred wells: {len(sacred_wells)}")

    # 2) dev features — residual target Δ=TVT−anchor (absolute TVT can't extrapolate)
    tr = features.build_dataset("train", well_ids=dev_wells, target="residual")
    X, y_drift, groups, anchor = tr["X"], tr["y"], tr["groups"], tr["anchor"]
    tvt_true = y_drift + anchor
    floor_rmse = rmse(tvt_true, anchor)
    print(f"dev: X={X.shape}  floor(dev carry-forward) RMSE={floor_rmse:.4f}")

    # 3) sacred + test features (same pipeline)
    sac = features.build_dataset("train", well_ids=sacred_wells, target="residual")
    X_sac, y_sac, sac_anchor = sac["X"], sac["y"], sac["anchor"]
    sacred_floor = rmse(y_sac + sac_anchor, sac_anchor)
    te = features.build_dataset("test", target="residual")
    X_test, test_anchor, test_groups = te["X"], te["anchor"], te["groups"]
    print(f"sacred: X={X_sac.shape}  floor(sacred) RMSE={sacred_floor:.4f}")
    print(f"test:   X={X_test.shape}")

    # 4) diary BEFORE training
    exp = Experiment.start(
        version=VERSION,
        parent="v5_ensemble",
        hypothesis=(
            "Trusted residual-GBDT baseline re-established under the discovery-first "
            "lens after the LB reality-check (top-1 6.69 vs pack 9.25 => ceiling claim "
            "FALSE, 2.5ft signal extractable). Expect sacred ~14-15 (at floor); this is "
            "the honest anchor for discovery, NOT the medal attempt. Judge ONLY on sacred."
        ),
        predicted_delta=0.0,
        confidence="high",
        feature_changes=[],
        pipeline_changes=["verified-API rebuild of the residual baseline; sacred-gated"],
        cloud_or_local="cloud",
    )

    # 5) train residual GBDT, GroupKFold-by-well, fit_full for sacred/test inference
    res = train.train_variant(
        VERSION, algo, X, y_drift, groups, X_test=X_test,
        save=True, fit_full=True, use_gpu="auto",
    )
    tvt_oof = res.oof + anchor
    oof_rmse = rmse(tvt_true, tvt_oof)

    # 6) THE HONEST NUMBER — sacred RMSE from the full-fit model
    full = joblib.load(train.PROBS / VERSION / "model_full.pkl")
    tvt_sac = full.predict(X_sac) + sac_anchor
    sacred_rmse = rmse(y_sac + sac_anchor, tvt_sac)

    print()
    print(f"OOF drift-RMSE   : {res.oof_rmse:.4f}")
    print(f"dev OOF TVT-RMSE : {oof_rmse:.4f}   (dev floor {floor_rmse:.4f})")
    print(f"SACRED TVT-RMSE  : {sacred_rmse:.4f}   (sacred floor {sacred_floor:.4f}, "
          f"Δ {sacred_rmse - sacred_floor:+.4f})")
    print(f"public pack ~9.25 | top-1 LB 6.69 | runtime {time.time()-t0:.0f}s")

    # 7) submission (de-residualize + per-well dicts)
    tvt_test = res.test_pred + test_anchor
    well_preds = {w: tvt_test[test_groups == w] for w in dict.fromkeys(test_groups)}
    ss = submission.build_submission_from_wells(well_preds, split="test")
    out = submission.save_submission(ss, VERSION)
    print(f"submission -> {out}")

    # 8) commit diary
    exp.record(
        oof_score_mean=oof_rmse,
        oof_score_per_fold=res.fold_rmses,
        holdout_score=sacred_rmse,
        runtime_sec=time.time() - t0,
        extra={"floor_rmse": float(floor_rmse), "sacred_rmse": float(sacred_rmse),
               "sacred_floor": float(sacred_floor), "algo": algo,
               "n_features": int(X.shape[1]), "blend": False, "ported": False},
    )
    exp.commit()
    print(f"\n{VERSION} committed. Flags: {exp.flags or '(none)'}")


if __name__ == "__main__":
    main()
