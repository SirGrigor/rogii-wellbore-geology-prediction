"""v1_lgb_resid — first GBDT baseline (RESIDUAL target). COLAB ONLY (full-data training).

Run on Colab via colab_runner (SPRINT_ACTIVE.txt → this script). Do NOT run locally:
all model fits go to Colab (feedback_cloud-first-compute, 2026-05-25 tightening).

Models the DRIFT (TVT - last_known_TVT), not absolute TVT — absolute fails because
between-well level variance dominates and trees can't extrapolate the level.

Honest expectation: the alignment feature is still the NAIVE global DTW (known weak),
so this may only modestly beat — or even trail — the 15.91 ft carry-forward floor.
The point is to establish the real GroupKFold OOF RMSE + the feature-gain ranking that
tells Tier 1 where the signal is (and whether the alignment recipe is the bottleneck).
"""
from __future__ import annotations

import time

import numpy as np

from src import features, submission, train
from src.evaluate import rmse
from src.observer import Experiment


def main() -> None:
    t0 = time.time()
    exp = Experiment.start(
        version="v1_lgb_resid",
        parent="v0_floor",
        hypothesis=("Residual LGB (drift = TVT - last_known_TVT) on trajectory + GR-sequence + "
                    "naive-DTW-alignment features. Alignment recipe is weak, so expect at most a "
                    "modest gain over the 15.91 ft floor; establishes real OOF + feature gains."),
        predicted_delta=1.0,          # ft of RMSE improvement vs floor — low confidence
        confidence="low",
        feature_changes=["+ residual target", "+ trajectory", "+ GR sequence/deviation", "+ naive align"],
        pipeline_changes=["GroupKFold-by-well LGB"],
        cloud_or_local="cloud",
    )

    print("building train features (residual target)...")
    tr = features.build_dataset("train", with_alignment=True, target="residual")
    X, y_drift, groups, anchor = tr["X"], tr["y"], tr["groups"], tr["anchor"]
    tvt_true = y_drift + anchor
    floor_rmse = rmse(tvt_true, anchor)                 # Δ=0 carry-forward floor on full data
    print(f"X {X.shape} | floor RMSE {floor_rmse:.4f} ft")

    print("building test features...")
    te = features.build_dataset("test", with_alignment=True, target="residual")
    X_test, test_anchor, test_groups = te["X"], te["anchor"], te["groups"]

    res = train.train_variant("v1_lgb_resid", "lgb", X, y_drift, groups,
                              X_test=X_test, save=True, fit_full=True)

    # de-residualize: TVT = drift + anchor; metric RMSE on absolute TVT
    tvt_oof = res.oof + anchor
    oof_rmse = rmse(tvt_true, tvt_oof)
    # per-fold metric RMSE (de-residualized) is what train logged as drift-RMSE; recompute on TVT
    print(f"OOF drift-RMSE {res.oof_rmse:.4f} | OOF TVT-RMSE {oof_rmse:.4f} (floor {floor_rmse:.4f})")

    # feature importances → Tier-1 guidance (where is the signal? is alignment the bottleneck?)
    import joblib
    full = joblib.load(train.PROBS / "v1_lgb_resid" / "model_full.pkl")
    imp = sorted(zip(X.columns, full.feature_importances_), key=lambda kv: -kv[1])
    print("top 12 features by gain:")
    for name, g in imp[:12]:
        print(f"  {name:24s} {g}")

    # test submission (precomputed path) — de-residualize per row
    tvt_test = res.test_pred + test_anchor
    well_preds = {wid: tvt_test[test_groups == wid] for wid in dict.fromkeys(test_groups)}
    ss = submission.build_submission_from_wells(well_preds, split="test")
    out = submission.save_submission(ss, "v1_lgb_resid")
    print(f"wrote {out}")

    exp.record(
        oof_score_mean=oof_rmse,
        oof_score_per_fold=res.fold_rmses,   # NOTE: drift-RMSE per fold (≈ TVT-RMSE since anchor const/well)
        holdout_score=oof_rmse,
        runtime_sec=time.time() - t0,
        extra={"floor_rmse": floor_rmse, "n_train_wells": len(tr["well_ids"]),
               "submission": out.name, "model": "probs/v1_lgb_resid/model_full.pkl"},
    )
    exp.note(f"floor {floor_rmse:.3f} -> OOF {oof_rmse:.3f} ft ({'beats' if oof_rmse < floor_rmse else 'TRAILS'} floor)")
    exp.commit()
    print("diary: committed v1_lgb_resid")


if __name__ == "__main__":
    main()
