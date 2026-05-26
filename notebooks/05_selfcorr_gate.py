"""v3 — GATE the known-prefix self-correlation features (P1) on the SACRED holdout. COLAB.

Self-correlation alone trails the floor (~20 vs ~13 ft) — as expected; no single alignment
method beats carry-forward (that's why the field sits at ~14ft, winners stack many). The
real test is whether the GBDT extracts signal from it IN COMBINATION. One variable: train
on (geometry + naive-align) vs (+ self-correlation), judged on sacred.

Verdict: if +selfcorr lowers sacred RMSE → keep building the ensemble; if not, P1 doesn't
help and we move to the anchored backbone (beam/PF-with-momentum, P2). Set ROGII_ALGO to override.
"""
from __future__ import annotations

import time

import joblib
import numpy as np

from src import cv, data, features, train
from src.evaluate import rmse
from src.observer import Experiment

ALGO = train.default_algo()


def _sacred_rmse(version, X_sac, anchor_sac, y_tvt_sac):
    full = joblib.load(train.PROBS / version / "model_full.pkl")
    return rmse(y_tvt_sac, full.predict(X_sac) + anchor_sac)


def main() -> None:
    print(f"GATE self-correlation — algo={ALGO}")
    dev, sacred = cv.sacred_split(data.list_well_ids("train"))
    tr = features.build_dataset("train", well_ids=dev, with_alignment=True, target="residual")
    X, y, groups, anchor = tr["X"], tr["y"], tr["groups"], tr["anchor"]
    sac = features.build_dataset("train", well_ids=sacred, with_alignment=True, target="residual")
    Xs, ys, anchor_s = sac["X"], sac["y"], sac["anchor"]
    y_tvt_dev, y_tvt_sac = y + anchor, ys + anchor_s
    sac_floor = rmse(y_tvt_sac, anchor_s)
    sc_cols = [c for c in X.columns if c.startswith("selfcorr_")]
    print(f"selfcorr cols gated: {sc_cols} | sac_floor {sac_floor:.3f}")

    results = {}
    for name, drop, parent in [("v3_noselfcorr", sc_cols, "v0_floor"),
                               ("v3_selfcorr", [], "v3_noselfcorr")]:
        t0 = time.time()
        Xv, Xsv = X.drop(columns=drop), Xs.drop(columns=drop)
        exp = Experiment.start(
            version=name, parent=parent,
            hypothesis=("Geometry + naive-align, NO self-correlation" if drop
                        else "Add known-prefix self-correlation features (P1)") + " — judged on sacred.",
            predicted_delta=0.3, confidence="low",
            feature_changes=(["- selfcorr_*"] if drop else ["+ selfcorr_*"]), cloud_or_local="cloud")
        res = train.train_variant(name, ALGO, Xv, y, groups, save=True, fit_full=True, use_gpu="auto")
        dev_oof = rmse(y_tvt_dev, res.oof + anchor)
        sac_rmse = _sacred_rmse(name, Xsv, anchor_s, y_tvt_sac)
        results[name] = sac_rmse
        exp.record(oof_score_mean=sac_rmse, oof_score_per_fold=res.fold_rmses, holdout_score=sac_rmse,
                   runtime_sec=time.time() - t0,
                   extra={"dev_oof": dev_oof, "sac_floor": sac_floor, "sacred_minus_floor": sac_rmse - sac_floor})
        exp.note(f"sacred {sac_rmse:.3f} vs floor {sac_floor:.3f}")
        exp.commit()
        print(f"  {name}: dev_oof {dev_oof:.3f} | SACRED {sac_rmse:.3f} (Δfloor {sac_rmse - sac_floor:+.3f})")

    no, yes = results["v3_noselfcorr"], results["v3_selfcorr"]
    print("\n=== GATE VERDICT (sacred RMSE) ===")
    print(f"  without selfcorr: {no:.4f}\n  with selfcorr:    {yes:.4f}")
    print(f"  → self-correlation {'HELPS' if yes < no else 'does NOT help'} ({yes - no:+.4f} ft)")
    print(f"  best vs sacred floor {sac_floor:.3f}: {min(no, yes) - sac_floor:+.3f} ft "
          f"({'BEATS floor' if min(no, yes) < sac_floor else 'still trails floor'})")


if __name__ == "__main__":
    main()
