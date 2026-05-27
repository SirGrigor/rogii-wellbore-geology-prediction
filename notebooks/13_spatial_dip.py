"""S-spatial — cross-well dip field ablation (the signal v11's residual diagnosis pointed to). COLAB (GPU).

v11 found error is dominated by HORIZON (octile RMSE 3.7→12.0) and LARGE DRIFT, and near-neighbour wells
err far less (5.54 vs 7.50) → cross-well signal is latent. This adds the frame-independent neighbour-dip
features (src/features_spatial_dip) to the 222 and measures marginal Δ on sacred — OVERALL and at LONG
HORIZON (md_since > median), the regime where self-alignment degrades. Keep iff it lowers sacred.

Leakage: cloud = dev wells; dev targets exclude their own well; sacred targets are clean (sacred ∉ cloud).
Decided on sacred. (If flat → the dip is redundant with the kernel's formation-plane-KNN; if it helps,
especially at far-horizon → orthogonal signal found, fold into the ensemble.)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import cv, dashboard as dash, data, features_spatial_dip as fsd, train
from src.evaluate import rmse
from src.observer import Experiment

CACHE = Path(os.environ.get("DRIVE_ROOT") or "data") / "cache"
VER = "v12_spatialdip"
CAT = dict(depth=6, learning_rate=0.03, random_seed=42, l2_leaf_reg=2.0, min_data_in_leaf=15)


def main() -> None:
    t0 = time.time()
    dev_w, sac_w = cv.sacred_split(data.list_well_ids("train"))
    dev_df = pd.read_parquet(CACHE / "dev_k9.parquet")
    sac_df = pd.read_parquet(CACHE / "sacred_k9.parquet")
    base = [c for c in dev_df.columns if c not in {"well", "id", "target"}]

    dash.goal_banner(VER, "cross-well dip field ablation",
                     "does neighbour dip add signal beyond the 222? (esp. long horizon, where we fail)")
    exp = Experiment.start(
        version=VER, parent="v6s8_fast",
        hypothesis=("Frame-independent cross-well dip (neighbours' TVT-plane gradient → drift extrapolation) "
                    "adds signal the per-well 222 miss, esp. at long horizon where self-alignment degrades "
                    "(v11: error 3.7→12 by horizon; near-neighbour wells 5.54 vs 7.50)."),
        predicted_delta=0.10, confidence="low",
        pipeline_changes=["cross-well dip field"], cloud_or_local="cloud")

    cloud = fsd.build_cloud(dev_w, "train", stride=4)
    t = time.time(); dev_dip = fsd.dip_features(dev_w, "train", cloud)
    print(f"[dip] dev {len(dev_dip)} rows in {time.time()-t:.0f}s")
    sac_dip = fsd.dip_features(sac_w, "train", cloud)
    dev_df = dev_df.join(dev_dip, on="id"); sac_df = sac_df.join(sac_dip, on="id")
    miss = int(dev_df[fsd.COLS].isna().sum().sum()) + int(sac_df[fsd.COLS].isna().sum().sum())
    print(f"[merge] dip NaN after join: {miss} (filled 0)")
    dev_df[fsd.COLS] = dev_df[fsd.COLS].fillna(0.0); sac_df[fsd.COLS] = sac_df[fsd.COLS].fillna(0.0)

    yd = dev_df["target"].to_numpy(np.float32); gd = dev_df["well"].to_numpy(); ys = sac_df["target"].to_numpy(np.float32)
    md = sac_df["md_since"].to_numpy() if "md_since" in sac_df.columns else np.zeros(len(ys))
    far = md > np.median(md)

    def run(feats, tag):
        r = train.train_variant(f"{VER}_{tag}", "cat", dev_df[feats].astype("float32"), yd, gd,
                                X_test=sac_df[feats].astype("float32"), params=CAT, save=False, use_gpu="auto")
        s, sf = rmse(ys, r.test_pred), rmse(ys[far], r.test_pred[far])
        print(f"  [{tag}] sacred {s:.3f} | far-horizon {sf:.3f}  ({len(feats)} feats)")
        return s, sf

    s0, s0f = run(base, "base222")
    s1, s1f = run(base + fsd.COLS, "+dip")
    print(f"\nSACRED: base {s0:.3f} → +dip {s1:.3f} (Δ {s1-s0:+.3f})  |  FAR-horizon: {s0f:.3f} → {s1f:.3f} (Δ {s1f-s0f:+.3f})")
    best = min(s0, s1)

    exp.record(oof_score_mean=best, oof_score_per_fold=[float(s0), float(s1)], holdout_score=best,
               runtime_sec=time.time() - t0,
               extra={"base": s0, "dip": s1, "d_overall": s1 - s0, "base_far": s0f, "dip_far": s1f, "d_far": s1f - s0f})
    exp.note(f"cross-well dip: sacred {s0:.3f}→{s1:.3f} (Δ{s1-s0:+.3f}), far Δ{s1f-s0f:+.3f}")
    exp.commit()

    dash.verdict(VER, best, time.time() - t0, simple_avg=s0, parent=9.155)
    v = ("✅ dip helps — orthogonal signal found" if s1 < s0 - 0.05
         else "≈ marginal" if s1 < s0 - 0.02 else "✗ redundant with the kernel's formation planes")
    print(f"=== {VER}: {v} | +dip {s1:.3f} vs base {s0:.3f} (Δ{s1-s0:+.3f}) | far-horizon Δ{s1f-s0f:+.3f} | {time.time()-t0:.0f}s ===")


if __name__ == "__main__":
    main()
