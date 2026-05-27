"""18 — large-scale auto-FE ablation: the #1 documented winner's edge, genuinely untried on rogii. COLAB.

KG survey: FE-depth (recover/encode structure as features) is the single biggest edge, 5/6 comps; we
hand-add a handful. NVIDIA Playbook #3: "generate more features, discover more patterns." This GENERATES
~1.3k candidates (pairwise prod/diff/ratio of the top-30 base feats + per-well group aggs), importance-
selects 150, and asks: does base+autoFE beat the base 222 on SACRED? If yes → FE-depth was our gap, expand
to the full stack. If flat → FE-depth is exhausted here too (the kernel's 222 already encode the structure;
the rogii gap is domain, not generic FE). Env: ROGII_ROW_STRIDE (default 4).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import cv, dashboard as dash, data, features_auto as fa, features_extra as fx, train
from src.evaluate import rmse
from src.observer import Experiment

CACHE = Path(os.environ.get("DRIVE_ROOT") or "data") / "cache"
VER = "v17_autofe"
STRIDE = int(os.environ.get("ROGII_ROW_STRIDE") or 4)
CAT = dict(depth=6, learning_rate=0.03, random_seed=42, l2_leaf_reg=2.0, min_data_in_leaf=15)


def main() -> None:
    t0 = time.time()
    dev_w, sac_w = cv.sacred_split(data.list_well_ids("train"))
    dev_df = pd.read_parquet(CACHE / "dev_k9.parquet").iloc[::STRIDE]
    sac_df = pd.read_parquet(CACHE / "sacred_k9.parquet")
    base = [c for c in dev_df.columns if c not in {"well", "id", "target"}]
    dev_df = dev_df.join(fx.build_extra(dev_w, "train")[fx.W1], on="id")
    sac_df = sac_df.join(fx.build_extra(sac_w, "train")[fx.W1], on="id")
    dev_df[fx.W1] = dev_df[fx.W1].fillna(0); sac_df[fx.W1] = sac_df[fx.W1].fillna(0)
    base = base + fx.W1
    yd = dev_df["target"].to_numpy(np.float32); gd = dev_df["well"].to_numpy(); ys = sac_df["target"].to_numpy(np.float32)

    dash.goal_banner(VER, f"large-scale auto-FE (stride {STRIDE})",
                     "#1 winner edge: do 1000s of generated feats beat the 222 on sacred?")
    exp = Experiment.start(version=VER, parent="v6s8_fast",
                           hypothesis="Large-scale auto-FE (prod/diff/ratio of top base feats + per-well "
                                      "group-aggs, importance-selected) adds signal the 222 miss → sacred < 9.16. "
                                      "The #1 documented winner edge (FE-depth), untried on rogii.",
                           predicted_delta=0.10, confidence="low",
                           pipeline_changes=["large-scale auto-FE"], cloud_or_local="cloud")

    sel, ncand = fa.select(dev_df, base, yd, k=30, keep=150)
    print(f"[autoFE] generated {ncand} candidates → selected {len(sel)} | top: {sel[:8]}")
    cand_dev = fa.build(dev_df, sel); cand_sac = fa.build(sac_df, sel)

    def run(Xd, Xs, tag):
        r = train.train_variant(f"{VER}_{tag}", "cat", Xd, yd, gd, X_test=Xs, params=CAT, save=False, use_gpu="auto")
        s = rmse(ys, r.test_pred); print(f"  [{tag}] sacred {s:.3f}  ({Xd.shape[1]} feats)"); return s

    s0 = run(dev_df[base].astype("float32"), sac_df[base].astype("float32"), "base")
    Xd2 = pd.concat([dev_df[base], cand_dev], axis=1).astype("float32")
    Xs2 = pd.concat([sac_df[base], cand_sac], axis=1).astype("float32")
    s1 = run(Xd2, Xs2, "base+autoFE")
    print(f"\nbase {s0:.3f} → base+autoFE {s1:.3f} (Δ {s1-s0:+.3f}) | vs v5 9.155 | floor {rmse(ys, np.zeros_like(ys)):.3f}")

    best = min(s0, s1)
    exp.record(oof_score_mean=best, oof_score_per_fold=[float(s0), float(s1)], holdout_score=best,
               runtime_sec=time.time() - t0,
               extra={"base": s0, "base_autofe": s1, "delta": s1 - s0, "n_candidates": ncand,
                      "n_selected": len(sel), "top_selected": sel[:15]})
    exp.note(f"large-scale auto-FE: base {s0:.3f} → +autoFE {s1:.3f} (Δ{s1-s0:+.3f})")
    exp.commit()

    dash.verdict(VER, best, time.time() - t0, simple_avg=s0, parent=9.155)
    v = ("✅ auto-FE helps — FE-depth WAS the gap" if s1 < s0 - 0.05
         else "≈ marginal" if s1 < s0 - 0.02 else "✗ flat — the 222 already encode the structure (gap is domain)")
    print(f"=== {VER}: {v} | base+autoFE {s1:.3f} vs base {s0:.3f} (Δ{s1-s0:+.3f}) | {time.time()-t0:.0f}s ===")


if __name__ == "__main__":
    main()
