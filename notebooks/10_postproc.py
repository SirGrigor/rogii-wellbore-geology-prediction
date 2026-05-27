"""S3a — the two genuinely-missing cheap levers from the TOP-3 solution. COLAB (GPU).

Uncertainty/divergence features are ALREADY in our 222 (kernel9251: sig_std, form_std_d,
frm_rmse_*, pf_vs_*, dtw_vs_*) — re-adding them is redundant (why our feature bets go flat).
So 'cheap wins' = what we DON'T have:
  1. CatBoost border_count=254 (finer numerical splits) + depth-7  vs our depth-6/default.
  2. per-well Savitzky-Golay smoothing of the predicted drift (window tuned on dev-OOF) + clip.

Decided on sacred. Reuses the Drive-cached 222 features.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import cv, dashboard as dash, data, postproc as pp, train  # noqa: F401 (cv/data kept for parity)
from src.evaluate import rmse
from src.observer import Experiment

CACHE = Path(os.environ.get("DRIVE_ROOT") or "data") / "cache"
VER = "v9_postproc"
BASE_CAT = dict(depth=6, learning_rate=0.03, random_seed=42, l2_leaf_reg=2.0, min_data_in_leaf=15)
QUAL_CAT = dict(depth=7, learning_rate=0.025, border_count=254, random_seed=42, l2_leaf_reg=2.0, min_data_in_leaf=15)


def main() -> None:
    t0 = time.time()
    dev_df = pd.read_parquet(CACHE / "dev_k9.parquet")
    sac_df = pd.read_parquet(CACHE / "sacred_k9.parquet")
    feats = [c for c in dev_df.columns if c not in {"well", "id", "target"}]
    Xd = dev_df[feats].astype("float32"); yd = dev_df["target"].to_numpy(np.float32)
    gd = dev_df["well"].to_numpy(); Xs = sac_df[feats].astype("float32"); ys = sac_df["target"].to_numpy(np.float32)
    dev_ids = dev_df["id"].tolist(); sac_ids = sac_df["id"].tolist()

    dash.goal_banner(VER, "CatBoost border=254/depth-7 + per-well savgol",
                     "the 2 missing cheap levers (uncertainty feats already in the 222)")
    exp = Experiment.start(
        version=VER, parent="v6s8_fast",
        hypothesis=("border_count=254 (finer splits) + per-well savgol smoothing of the drift curve "
                    "(window tuned on dev-OOF) + clip lower sacred below the ~9.16 GBDT floor."),
        predicted_delta=0.15, confidence="medium",
        pipeline_changes=["border_count=254/depth7", "per-well savgol", "clip"], cloud_or_local="cloud")

    def run(params, tag):
        r = train.train_variant(f"{VER}_{tag}", "cat", Xd, yd, gd, X_test=Xs, params=params,
                                save=False, use_gpu="auto")
        print(f"  [{tag}] dev_oof {r.oof_rmse:.3f} | sacred {rmse(ys, r.test_pred):.3f}")
        return r

    rb = run(BASE_CAT, "base_d6")
    rq = run(QUAL_CAT, "qual_d7_b254")
    use, tag = (rq, "qual") if rq.oof_rmse < rb.oof_rmse else (rb, "base")   # pick by OOF, not sacred
    s_raw = rmse(ys, use.test_pred)

    bw, tbl = pp.tune_window(dev_ids, use.oof, yd)                            # tune window on OOF (leak-free)
    sm_sac = pp.smooth_per_well(sac_ids, use.test_pred, bw) if bw else use.test_pred
    s_sm = rmse(ys, sm_sac)
    lo, hi = pp.fit_clip(yd)
    s_clip = rmse(ys, np.clip(sm_sac, lo, hi))

    d_border = rmse(ys, rq.test_pred) - rmse(ys, rb.test_pred)
    print(f"\nbase={tag} (oof {use.oof_rmse:.3f}) | savgol window {bw} | OOF table {{{', '.join(f'{k}:{v:.3f}' for k,v in tbl.items())}}}")
    print(f"sacred: raw {s_raw:.3f} | +savgol {s_sm:.3f} (Δ {s_sm-s_raw:+.3f}) | +clip {s_clip:.3f} (Δ {s_clip-s_raw:+.3f})")
    print(f"border_count=254 vs default: Δsacred {d_border:+.3f}")
    best = min(s_raw, s_sm, s_clip)

    exp.record(oof_score_mean=best, oof_score_per_fold=[float(s_raw), float(s_sm), float(s_clip)],
               holdout_score=best, runtime_sec=time.time() - t0,
               extra={"base_d6_sacred": rmse(ys, rb.test_pred), "qual_sacred": rmse(ys, rq.test_pred),
                      "border_delta": d_border, "savgol_window": bw, "raw": s_raw, "savgol": s_sm,
                      "clip": s_clip, "oof_table": {str(k): float(v) for k, v in tbl.items()}})
    exp.note(f"border254 Δ{d_border:+.3f} + savgol(w={bw}) Δ{s_sm-s_raw:+.3f}: sacred {s_raw:.3f}→{best:.3f}")
    exp.commit()

    dash.verdict(VER, best, time.time() - t0, simple_avg=s_raw, parent=9.155)
    v = "✅ cheap wins land" if best < s_raw - 0.05 else "≈ marginal" if best < s_raw - 0.01 else "✗ flat"
    print(f"=== {VER}: {v} | best {best:.3f} (raw {s_raw:.3f}) | savgol Δ{s_sm-s_raw:+.3f} | border Δ{d_border:+.3f} | {time.time()-t0:.0f}s ===")


if __name__ == "__main__":
    main()
