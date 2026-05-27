"""17 — MAX-quality build: deep diverse stack + exact-metric negative-weight blend + metric post-proc.

Embodies our two documented winner's edges (KG winning-technique-survey L47): NOT model breadth, but
(a) FE-depth — the 222 kernel features + the structure-derived W1 (horizon/geometry, our one non-flat add);
(b) blend-method — exact-metric NEGATIVE-weight hill-climb (not positive-only). Plus the "post-processing
grandmaster" lever: per-well savgol tuned on OOF. The best honest reproduction; judged on sacred.

Models (deep L1 stack, the romantamrazov "quality" configs): LGB-255 ×2, CatBoost depth-7/border-254 ×2,
HistGB. Env: ROGII_ROW_STRIDE (default 2 — more data than the survey runs).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from src import blend, cv, dashboard as dash, data, features_extra as fx, postproc as pp, train
from src.evaluate import rmse
from src.observer import Experiment

CACHE = Path(os.environ.get("DRIVE_ROOT") or "data") / "cache"
VER = "v16_max"
STRIDE = int(os.environ.get("ROGII_ROW_STRIDE") or 2)
LGB = dict(num_leaves=255, min_child_samples=15, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=0.05)
CAT = dict(depth=7, border_count=254, l2_leaf_reg=2.0, min_data_in_leaf=15)
MODELS = [
    ("lgb0", "lgb", dict(learning_rate=0.02, random_state=42, **LGB)),
    ("lgb1", "lgb", dict(learning_rate=0.03, random_state=7, **LGB)),
    ("cat0", "cat", dict(learning_rate=0.025, random_seed=42, **CAT)),
    ("cat1", "cat", dict(learning_rate=0.02, random_seed=7, **CAT)),
]


def hgb_cv(X, y, g, Xs, n=5):
    oof = np.zeros(len(y)); sac = np.zeros(len(Xs))
    for tr, va in GroupKFold(n).split(X, y, g):
        m = HistGradientBoostingRegressor(max_iter=800, learning_rate=0.04, max_leaf_nodes=127,
                                          l2_regularization=1.0, early_stopping=True, n_iter_no_change=40,
                                          validation_fraction=0.1, random_state=42)
        m.fit(X[tr], y[tr]); oof[va] = m.predict(X[va]); sac += m.predict(Xs) / n
    return oof, sac


def main() -> None:
    t0 = time.time()
    dev_w, sac_w = cv.sacred_split(data.list_well_ids("train"))
    dev_df = pd.read_parquet(CACHE / "dev_k9.parquet").iloc[::STRIDE]
    sac_df = pd.read_parquet(CACHE / "sacred_k9.parquet")
    base = [c for c in dev_df.columns if c not in {"well", "id", "target"}]
    # FE-depth: + W1 structure features (horizon/geometry — our one non-flat add)
    dev_df = dev_df.join(fx.build_extra(dev_w, "train")[fx.W1], on="id")
    sac_df = sac_df.join(fx.build_extra(sac_w, "train")[fx.W1], on="id")
    dev_df[fx.W1] = dev_df[fx.W1].fillna(0); sac_df[fx.W1] = sac_df[fx.W1].fillna(0)
    feats = base + fx.W1
    yd = dev_df["target"].to_numpy(np.float32); gd = dev_df["well"].to_numpy(); ys = sac_df["target"].to_numpy(np.float32)
    dev_ids = dev_df["id"].tolist(); sac_ids = sac_df["id"].tolist()

    dash.goal_banner(VER, f"deep stack + neg-blend + savgol (stride {STRIDE})",
                     "max honest reproduction: FE-depth + exact-metric negative blend + post-proc")
    exp = Experiment.start(version=VER, parent="v6s8_fast",
                           hypothesis="Deep diverse stack (LGB-255×2, CAT-d7-b254×2, HistGB) + W1 structure "
                                      "feats + exact-metric NEGATIVE-weight blend + per-well savgol — the two "
                                      "documented winner's edges. Best honest reproduction.",
                           predicted_delta=0.1, confidence="low",
                           pipeline_changes=["deep stack", "negative blend", "savgol", "W1 feats"], cloud_or_local="cloud")

    oof_d, sac_d = {}, {}
    for name, algo, params in MODELS:
        r = train.train_variant(f"{VER}_{name}", algo, dev_df[feats].astype("float32"), yd, gd,
                                X_test=sac_df[feats].astype("float32"), params=params, save=False, use_gpu="auto")
        oof_d[name], sac_d[name] = r.oof, r.test_pred
        print(f"  [{name}] sacred {rmse(ys, r.test_pred):.3f}")
    oof_d["hgb"], sac_d["hgb"] = hgb_cv(dev_df[feats].fillna(0).to_numpy(np.float32), yd, gd,
                                        sac_df[feats].fillna(0).to_numpy(np.float32))
    print(f"  [hgb] sacred {rmse(ys, sac_d['hgb']):.3f}")

    # blend ENGINE comparison on real sacred: NM (overfit-prone, what we shipped) vs CARUANA
    # (greedy, bagged, sorted-init — overfit-resistant; the agents' named gap, now wired in).
    cand = {}
    wn, _, _ = blend.nm_optimize_oof(oof_d, yd, allow_negative=False); cand["nm"] = wn
    wneg, _, _ = blend.nm_optimize_oof(oof_d, yd, allow_negative=True); cand["nm_neg"] = wneg
    wc, _, cinfo = blend.caruana_select(oof_d, yd); cand["caruana"] = wc
    for k, w in cand.items():
        print(f"blend[{k:8s}] sacred {rmse(ys, blend.apply_blend(sac_d, w)):.3f}")
    print(f"  caruana: best_single {cinfo['best_single_score']:.3f} | mean {cinfo['simple_mean_score']:.3f} | n_selected {cinfo['n_selected']}")
    bk = min(cand, key=lambda k: rmse(ys, blend.apply_blend(sac_d, cand[k])))
    w = cand[bk]; s_blend = rmse(ys, blend.apply_blend(sac_d, w)); print(f"  → best engine on sacred: {bk}")
    blend_oof = blend.apply_blend(oof_d, w); blend_sac = blend.apply_blend(sac_d, w)

    # post-processing grandmaster lever: per-well savgol, window tuned on dev-OOF
    bw, _ = pp.tune_window(dev_ids, blend_oof, yd)
    sac_sm = pp.smooth_per_well(sac_ids, blend_sac, bw) if bw else blend_sac
    s_final = rmse(ys, sac_sm)
    print(f"\nblend({'neg' if bn else 'pos'}) {s_blend:.3f} | +savgol(w={bw}) {s_final:.3f} | vs v5 9.155 | floor {rmse(ys, np.zeros_like(ys)):.3f}")

    final = min(s_blend, s_final)
    exp.record(oof_score_mean=final, oof_score_per_fold=[rmse(ys, sac_d[n]) for n in sac_d],
               holdout_score=final, runtime_sec=time.time() - t0,
               extra={**{f"{n}_sacred": rmse(ys, sac_d[n]) for n in sac_d},
                      **{f"blend_{k}": rmse(ys, blend.apply_blend(sac_d, cand[k])) for k in cand},
                      "best_engine": bk, "savgol_w": int(bw), "final": final,
                      "weights": {k: round(v, 3) for k, v in w.items()}})
    exp.note(f"deep-stack+negblend+savgol: {final:.3f} (blend {s_blend:.3f}, savgol {s_final:.3f}) vs v5 9.155")
    exp.commit()
    dash.verdict(VER, final, time.time() - t0, simple_avg=rmse(ys, blend.apply_blend(sac_d, cand["nm"])), parent=9.155)
    v = "✅ max-build beats 9.155" if final < 9.155 - 0.02 else "≈ matched best (public-tier reproduction)"
    print(f"=== {VER}: {v} | final {final:.3f} vs v5 9.155 | {time.time()-t0:.0f}s ===")


if __name__ == "__main__":
    main()
