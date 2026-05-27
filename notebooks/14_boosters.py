"""Booster overview — which gradient booster is most suitable, and does a diverse blend help. COLAB (GPU).

Within the (now-understood-saturated) 222-feature paradigm — so expectations are MARGINAL. But the user
asked, and a decorrelated booster family can add a small blend gain (unlike the CNN/TabPFN, which predicted
poorly). Compares LightGBM / CatBoost / XGBoost / HistGradientBoosting: individual sacred, the residual-
correlation matrix (diversity), and the supervised dev-OOF blend vs the best single + our lgb+cat baseline.
stride-4 for a fast relative survey. Decided on sacred.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from src import blend, dashboard as dash, train
from src.evaluate import rmse
from src.observer import Experiment

CACHE = Path(os.environ.get("DRIVE_ROOT") or "data") / "cache"
VER = "v13_boosters"
STRIDE = int(os.environ.get("ROGII_ROW_STRIDE") or 4)
PARAMS = {
    "lgb": dict(num_leaves=63, learning_rate=0.03, random_state=42, min_child_samples=15,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=0.05),
    "cat": dict(depth=6, learning_rate=0.03, random_seed=42, l2_leaf_reg=2.0, min_data_in_leaf=15),
    "xgb": dict(max_depth=7, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0),
}


def hgb_cv(X, y, g, Xs, n=5):
    """HistGradientBoosting (sklearn, native NaN) — honest by-well OOF + bagged sacred."""
    oof = np.zeros(len(y)); sac = np.zeros(len(Xs))
    for tr, va in GroupKFold(n).split(X, y, g):
        m = HistGradientBoostingRegressor(max_iter=700, learning_rate=0.05, max_leaf_nodes=63,
                                          l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.1, n_iter_no_change=30, random_state=42)
        m.fit(X[tr], y[tr]); oof[va] = m.predict(X[va]); sac += m.predict(Xs) / n
    return oof, sac


def main() -> None:
    t0 = time.time()
    dev_df = pd.read_parquet(CACHE / "dev_k9.parquet")
    sac_df = pd.read_parquet(CACHE / "sacred_k9.parquet")
    feats = [c for c in dev_df.columns if c not in {"well", "id", "target"}]
    X = dev_df[feats].iloc[::STRIDE].astype("float32"); y = dev_df["target"].to_numpy(np.float32)[::STRIDE]
    g = dev_df["well"].to_numpy()[::STRIDE]
    Xs = sac_df[feats].astype("float32"); ys = sac_df["target"].to_numpy(np.float32)

    dash.goal_banner(VER, f"booster overview (stride {STRIDE})", "most suitable booster + does a diverse blend help?")
    exp = Experiment.start(version=VER, parent="v6s8_fast",
                           hypothesis="A more diverse booster set (lgb/cat/xgb/HistGB) gives a decorrelated "
                                      "blend member that lowers sacred below the lgb+cat ~9.16 — marginal, within-paradigm.",
                           predicted_delta=0.05, confidence="low",
                           pipeline_changes=["booster survey + diverse blend"], cloud_or_local="cloud")

    oof_d, sac_d = {}, {}
    for algo in ("lgb", "cat", "xgb"):
        r = train.train_variant(f"{VER}_{algo}", algo, X, y, g, X_test=Xs, params=PARAMS[algo], save=False, use_gpu="auto")
        oof_d[algo], sac_d[algo] = r.oof, r.test_pred
        print(f"  [{algo}] sacred {rmse(ys, r.test_pred):.3f}")
    t = time.time(); oof_d["hgb"], sac_d["hgb"] = hgb_cv(X.fillna(0).to_numpy(np.float32), y, g, Xs.fillna(0).to_numpy(np.float32))
    print(f"  [hgb] sacred {rmse(ys, sac_d['hgb']):.3f}  ({time.time()-t:.0f}s)")

    names = list(sac_d)
    print("\nresidual-corr matrix (sacred):    " + "  ".join(f"{n:>6}" for n in names))
    res = {n: ys - sac_d[n] for n in names}
    for a in names:
        print(f"  {a:>6}  " + "  ".join(f"{np.corrcoef(res[a], res[b])[0,1]:6.3f}" for b in names))

    w_lc, _, _ = blend.nm_optimize_oof({k: oof_d[k] for k in ("lgb", "cat")}, y)
    s_lc = rmse(ys, blend.apply_blend({k: sac_d[k] for k in ("lgb", "cat")}, w_lc))
    w_all, _, _ = blend.nm_optimize_oof(oof_d, y)
    s_all = rmse(ys, blend.apply_blend(sac_d, w_all))
    best_single = min((rmse(ys, sac_d[n]), n) for n in names)
    print(f"\nbest single: {best_single[1]} {best_single[0]:.3f} | lgb+cat blend {s_lc:.3f} | "
          f"all-4 blend {s_all:.3f} (Δ {s_all-s_lc:+.3f})")
    print(f"all-4 weights: {dict((k, round(v,3)) for k,v in w_all.items())}")
    best = min(s_lc, s_all)

    exp.record(oof_score_mean=best, oof_score_per_fold=[rmse(ys, sac_d[n]) for n in names], holdout_score=best,
               runtime_sec=time.time() - t0,
               extra={**{f"{n}_sacred": rmse(ys, sac_d[n]) for n in names}, "lgbcat_blend": s_lc,
                      "all4_blend": s_all, "d_diverse": s_all - s_lc, "weights": {k: round(v, 3) for k, v in w_all.items()}})
    exp.note(f"boosters: best single {best_single[1]} {best_single[0]:.3f}; lgb+cat {s_lc:.3f} → all-4 {s_all:.3f}")
    exp.commit()

    dash.verdict(VER, best, time.time() - t0, simple_avg=s_lc, parent=9.155)
    v = "✅ diverse blend helps" if s_all < s_lc - 0.03 else "≈ flat — boosters are correlated"
    print(f"=== {VER}: {v} | all-4 {s_all:.3f} vs lgb+cat {s_lc:.3f} (Δ{s_all-s_lc:+.3f}) | {time.time()-t0:.0f}s ===")


if __name__ == "__main__":
    main()
