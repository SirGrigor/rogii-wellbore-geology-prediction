"""v2 — GATE the (naive-DTW) alignment features on the SACRED holdout. COLAB.

v1 finding: the model leans hardest on align_* (top gains) but does NOT beat the
carry-forward floor on the sacred wells (14.45 vs 14.08) — it overfits naive-DTW
alignment noise. This is the definitive with/without-fit gate (handover §5):
train geometry-only vs geometry+alignment, judge on sacred. ONE variable changed.

Verdict rule: if geometry-only's sacred RMSE ≤ geometry+alignment's → the naive
alignment HURTS → drop it until the alignment RECIPE is fixed (the real 6.7ft lever).

Run on Colab via colab_runner. Set ROGII_ALGO=xgb for the GPU path.
"""
from __future__ import annotations

import os
import time

import joblib
import numpy as np

from src import cv, data, features, train
from src.evaluate import rmse
from src.observer import Experiment

ALGO = os.environ.get("ROGII_ALGO", "lgb")


def _sacred_rmse(version: str, X_sac, anchor_sac, y_tvt_sac) -> float:
    full = joblib.load(train.PROBS / version / "model_full.pkl")
    return rmse(y_tvt_sac, full.predict(X_sac) + anchor_sac)


def main() -> None:
    print(f"GATE alignment — algo={ALGO}")
    dev, sacred = cv.sacred_split(data.list_well_ids("train"))

    # Build WITH alignment once; geometry-only = drop align_* columns from the same matrix.
    tr = features.build_dataset("train", well_ids=dev, with_alignment=True, target="residual")
    X, y, groups, anchor = tr["X"], tr["y"], tr["groups"], tr["anchor"]
    sac = features.build_dataset("train", well_ids=sacred, with_alignment=True, target="residual")
    Xs, ys, anchor_s = sac["X"], sac["y"], sac["anchor"]
    y_tvt_dev, y_tvt_sac = y + anchor, ys + anchor_s
    dev_floor, sac_floor = rmse(y_tvt_dev, anchor), rmse(y_tvt_sac, anchor_s)
    align_cols = [c for c in X.columns if c.startswith("align_")]
    print(f"align cols gated: {align_cols} | dev_floor {dev_floor:.3f} sac_floor {sac_floor:.3f}")

    results = {}
    for name, drop, parent in [("v2_geom", align_cols, "v0_floor"),
                               ("v2_geom_align", [], "v2_geom")]:
        t0 = time.time()
        Xv, Xsv = X.drop(columns=drop), Xs.drop(columns=drop)
        exp = Experiment.start(
            version=name, parent=parent,
            hypothesis=("Geometry-only residual model (no alignment)" if not drop == []
                        else "Add naive-DTW alignment back on top of geometry") +
                       " — judged on the sacred holdout.",
            predicted_delta=0.3, confidence="low",
            feature_changes=(["- align_*"] if drop else ["+ align_*"]),
            cloud_or_local="cloud",
        )
        res = train.train_variant(name, ALGO, Xv, y, groups, save=True, fit_full=True, use_gpu="auto")
        dev_oof = rmse(y_tvt_dev, res.oof + anchor)
        sac_rmse = _sacred_rmse(name, Xsv, anchor_s, y_tvt_sac)
        results[name] = (dev_oof, sac_rmse)
        exp.record(oof_score_mean=sac_rmse, oof_score_per_fold=res.fold_rmses,
                   holdout_score=sac_rmse, runtime_sec=time.time() - t0,
                   extra={"dev_oof": dev_oof, "dev_floor": dev_floor, "sac_floor": sac_floor,
                          "sacred_minus_floor": sac_rmse - sac_floor, "n_features": Xv.shape[1]})
        exp.note(f"sacred {sac_rmse:.3f} vs floor {sac_floor:.3f} "
                 f"({'beats' if sac_rmse < sac_floor else 'TRAILS'} by {abs(sac_rmse-sac_floor):.3f})")
        exp.commit()
        print(f"  {name}: dev_oof {dev_oof:.3f} | SACRED {sac_rmse:.3f} (floor {sac_floor:.3f}, "
              f"Δ {sac_rmse - sac_floor:+.3f})")

    g, ga = results["v2_geom"][1], results["v2_geom_align"][1]
    print("\n=== GATE VERDICT (sacred RMSE) ===")
    print(f"  geometry-only:      {g:.4f}")
    print(f"  geometry+alignment: {ga:.4f}")
    print(f"  → naive alignment {'HELPS' if ga < g else 'HURTS'} "
          f"({ga - g:+.4f} ft on sacred). "
          f"{'Keep it.' if ga < g else 'DROP it; fix the alignment recipe (the real lever).'}")
    print(f"  best vs sacred floor {sac_floor:.3f}: {min(g, ga) - sac_floor:+.3f} ft")


if __name__ == "__main__":
    main()
