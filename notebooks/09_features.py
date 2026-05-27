"""Strategy-2 — sequential GBDT feature ablation. COLAB (GPU).

GBDT is capped at sacred ~9.16 on the 222 align features. Add cheap, leakage-free feature
waves and measure each one's MARGINAL Δ on sacred with ONE consistent cat model (so deltas are
comparable). Keep a wave only if it lowers sacred; drop it otherwise (roadmap G-rule).

  base222  → the cached kernel features (our ~9.16 reference, single-cat)
  +W1      → horizon + geometry (md-since-PS, ▲Z, inclination)
  +W2      → GR texture (gradient, curvature, local roughness/level)
  +W2only  → isolates W2 (so we don't credit W2 for W1's lift)

Reuses the Drive-cached features; extra feats computed from raw + merged by id.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import cv, dashboard as dash, data, features_extra as fx, train
from src.evaluate import rmse
from src.observer import Experiment

CACHE = Path(os.environ.get("DRIVE_ROOT") or "data") / "cache"
CAT = dict(depth=6, learning_rate=0.03, random_seed=42, l2_leaf_reg=2.0, min_data_in_leaf=15)
VER = "v8_feats"


def main() -> None:
    t0 = time.time()
    dev_w, sac_w = cv.sacred_split(data.list_well_ids("train"))
    dev_df = pd.read_parquet(CACHE / "dev_k9.parquet")
    sac_df = pd.read_parquet(CACHE / "sacred_k9.parquet")
    base = [c for c in dev_df.columns if c not in {"well", "id", "target"}]

    # extra features from raw, merged onto the cached frames by id
    dev_df = dev_df.join(fx.build_extra(dev_w, "train"), on="id")
    sac_df = sac_df.join(fx.build_extra(sac_w, "train"), on="id")
    extra = fx.W1 + fx.W2
    miss = int(dev_df[extra].isna().sum().sum()) + int(sac_df[extra].isna().sum().sum())
    print(f"[merge] extra-feat NaN after join: {miss} (filled with 0)")
    dev_df[extra] = dev_df[extra].fillna(0.0); sac_df[extra] = sac_df[extra].fillna(0.0)

    yd = dev_df["target"].to_numpy(np.float32); gd = dev_df["well"].to_numpy()
    ys = sac_df["target"].to_numpy(np.float32)

    def sacred_of(feats, tag):
        r = train.train_variant(f"{VER}_{tag}", "cat", dev_df[feats].astype("float32"), yd, gd,
                                 X_test=sac_df[feats].astype("float32"), params=CAT, save=False, use_gpu="auto")
        s = rmse(ys, r.test_pred)
        print(f"  [{tag}] dev_oof {r.oof_rmse:.3f} | sacred {s:.3f}  ({len(feats)} feats)")
        return s

    dash.goal_banner(VER, "Strategy-2 feature ablation (cat depth-6)",
                     "do horizon/geom + GR-texture feats add signal beyond the 222? (marginal Δ on sacred)")
    exp = Experiment.start(
        version=VER, parent="v6s8_fast",
        hypothesis=("Horizon+geometry (▲Z since PS, md-since-PS, inclination) and GR-texture (gradient, "
                    "curvature, roughness) carry signal the 222 alignment feats miss → sacred < the ~9.16 floor."),
        predicted_delta=0.10, confidence="low",
        pipeline_changes=["+W1 horizon/geom", "+W2 GR texture"], cloud_or_local="cloud")

    s0 = sacred_of(base, "base222")
    s1 = sacred_of(base + fx.W1, "+W1")
    s2 = sacred_of(base + fx.W1 + fx.W2, "+W1W2")
    s2o = sacred_of(base + fx.W2, "+W2only")
    best = min(s0, s1, s2, s2o)
    print(f"\nbase {s0:.3f} | +W1 {s1:.3f} (Δ {s1-s0:+.3f}) | +W1W2 {s2:.3f} (Δ {s2-s0:+.3f}) "
          f"| +W2only {s2o:.3f} (Δ {s2o-s0:+.3f})  ← keep a wave iff Δ < 0")

    exp.record(oof_score_mean=best, oof_score_per_fold=[float(s0), float(s1), float(s2), float(s2o)],
               holdout_score=best, runtime_sec=time.time() - t0,
               extra={"base222": s0, "W1": s1, "W1W2": s2, "W2only": s2o,
                      "dW1": s1 - s0, "dW2only": s2o - s0, "dW1W2": s2 - s0})
    exp.note(f"feat ablation: base {s0:.3f} | +W1 {s1:.3f} | +W1W2 {s2:.3f} | +W2only {s2o:.3f}")
    exp.commit()

    dash.verdict(VER, best, time.time() - t0, simple_avg=s0, parent=9.155)
    verdict = ("✅ features help" if best < s0 - 0.05 else "≈ flat" if best < s0 - 0.01
               else "✗ no lift — GBDT signal-saturated")
    print(f"=== {VER}: {verdict} | best {best:.3f} vs base {s0:.3f} | {time.time()-t0:.0f}s ===")


if __name__ == "__main__":
    main()
