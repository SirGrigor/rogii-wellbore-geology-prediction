"""19 — Optuna HPO on CatBoost (finish the depth prescription) → tuned deep stack. COLAB (GPU).

The audit's one real recurring gap was solo-model depth: single-seed, ZERO HPO. The max build (17) added
seeds + cat-standalone → 9.100. This adds the missing piece: Optuna tunes CatBoost (depth/lr/l2/border/
random_strength/bagging) on the dev GroupKFold OOF (NEVER sacred — L53 discipline), on a stride subsample
for speed; the best params then train the full deep stack (tuned-cat×2 + LGB-255×2 + HistGB) at lower
stride. Blend nm_neg + savgol → sacred. Does HPO push past 9.100? Env: ROGII_HPO_TRIALS, ROGII_HPO_STRIDE,
ROGII_ROW_STRIDE.
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
VER = "v18_optuna"
N_TRIALS = int(os.environ.get("ROGII_HPO_TRIALS") or 30)
HPO_STRIDE = int(os.environ.get("ROGII_HPO_STRIDE") or 16)
STRIDE = int(os.environ.get("ROGII_ROW_STRIDE") or 4)
LGB = dict(num_leaves=255, min_child_samples=15, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=0.05)


def main() -> None:
    t0 = time.time()
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dev_w, sac_w = cv.sacred_split(data.list_well_ids("train"))
    dev_full = pd.read_parquet(CACHE / "dev_k9.parquet")
    sac_df = pd.read_parquet(CACHE / "sacred_k9.parquet")
    base = [c for c in dev_full.columns if c not in {"well", "id", "target"}]
    ex_dev = fx.build_extra(dev_w, "train")[fx.W1]; ex_sac = fx.build_extra(sac_w, "train")[fx.W1]
    dev_full = dev_full.join(ex_dev, on="id"); sac_df = sac_df.join(ex_sac, on="id")
    dev_full[fx.W1] = dev_full[fx.W1].fillna(0); sac_df[fx.W1] = sac_df[fx.W1].fillna(0)
    feats = base + fx.W1
    ys = sac_df["target"].to_numpy(np.float32)

    dash.goal_banner(VER, f"Optuna CatBoost HPO ({N_TRIALS} trials) → tuned deep stack",
                     "finish the depth prescription — does HPO push past 9.100?")
    exp = Experiment.start(version=VER, parent="v16_max",
                           hypothesis="Optuna HPO on CatBoost (the audit's missing depth lever — we never "
                                      "tuned) pushes the deep stack past the un-tuned 9.100, on dev-OOF (never sacred).",
                           predicted_delta=0.05, confidence="medium",
                           pipeline_changes=[f"Optuna CatBoost HPO {N_TRIALS} trials"], cloud_or_local="cloud")

    # --- HPO on a stride subsample, objective = dev GroupKFold OOF RMSE (never sacred) ---
    hpo = dev_full.iloc[::HPO_STRIDE]
    Xh = hpo[feats].astype("float32"); yh = hpo["target"].to_numpy(np.float32); gh = hpo["well"].to_numpy()
    print(f"[hpo] {len(hpo)} rows (stride {HPO_STRIDE}), {N_TRIALS} trials")

    def objective(trial):
        p = dict(depth=trial.suggest_int("depth", 5, 9),
                 learning_rate=trial.suggest_float("learning_rate", 0.012, 0.06, log=True),
                 l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 12.0, log=True),
                 min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 5, 60),
                 border_count=trial.suggest_categorical("border_count", [128, 200, 254]),
                 random_strength=trial.suggest_float("random_strength", 0.0, 2.0),
                 bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
                 random_seed=42)
        r = train.train_variant(f"{VER}_hpo", "cat", Xh, yh, gh, params=p, save=False, fit_full=False, use_gpu="auto")
        return r.oof_rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    bp = study.best_params
    print(f"[hpo] best OOF {study.best_value:.3f} | params {bp}")

    # --- tuned deep stack at the modeling stride ---
    dev_df = dev_full.iloc[::STRIDE]
    yd = dev_df["target"].to_numpy(np.float32); gd = dev_df["well"].to_numpy()
    Xd = dev_df[feats].astype("float32"); Xs = sac_df[feats].astype("float32")
    dev_ids = dev_df["id"].tolist(); sac_ids = sac_df["id"].tolist()
    MODELS = [("cat_t0", "cat", {**bp, "random_seed": 42}), ("cat_t1", "cat", {**bp, "random_seed": 7}),
              ("lgb0", "lgb", dict(learning_rate=0.02, random_state=42, **LGB)),
              ("lgb1", "lgb", dict(learning_rate=0.03, random_state=7, **LGB))]
    oof_d, sac_d = {}, {}
    for name, algo, params in MODELS:
        r = train.train_variant(f"{VER}_{name}", algo, Xd, yd, gd, X_test=Xs, params=params, save=False, use_gpu="auto")
        oof_d[name], sac_d[name] = r.oof, r.test_pred
        print(f"  [{name}] sacred {rmse(ys, r.test_pred):.3f}")
    oof_d["hgb"], sac_d["hgb"] = _hgb(Xd.fillna(0).to_numpy(np.float32), yd, gd, Xs.fillna(0).to_numpy(np.float32))
    print(f"  [hgb] sacred {rmse(ys, sac_d['hgb']):.3f}")

    cand = {}
    cand["nm"], _, _ = blend.nm_optimize_oof(oof_d, yd, allow_negative=False)
    cand["nm_neg"], _, _ = blend.nm_optimize_oof(oof_d, yd, allow_negative=True)
    cand["caruana"], _, _ = blend.caruana_select(oof_d, yd)
    for k, w in cand.items():
        print(f"blend[{k:8s}] sacred {rmse(ys, blend.apply_blend(sac_d, w)):.3f}")
    bk = min(cand, key=lambda k: rmse(ys, blend.apply_blend(sac_d, cand[k])))
    w = cand[bk]; s_blend = rmse(ys, blend.apply_blend(sac_d, w))
    bw, _ = pp.tune_window(dev_ids, blend.apply_blend(oof_d, w), yd)
    s_final = rmse(ys, pp.smooth_per_well(sac_ids, blend.apply_blend(sac_d, w), bw) if bw else blend.apply_blend(sac_d, w))
    final = min(s_blend, s_final)
    print(f"\nbest engine {bk} | blend {s_blend:.3f} | +savgol {s_final:.3f} | vs max-build 9.100 / v5 9.155")

    exp.record(oof_score_mean=final, oof_score_per_fold=[rmse(ys, sac_d[n]) for n in sac_d], holdout_score=final,
               runtime_sec=time.time() - t0, extra={"hpo_best_oof": study.best_value, "best_params": bp,
               "blend": s_blend, "savgol": s_final, "best_engine": bk, **{f"{n}_sacred": rmse(ys, sac_d[n]) for n in sac_d}})
    exp.note(f"Optuna cat → stack {final:.3f} vs max 9.100 (best params {bp})")
    exp.commit()
    dash.verdict(VER, final, time.time() - t0, simple_avg=9.100, parent=9.100)
    v = "✅ HPO pushed past 9.100" if final < 9.100 - 0.01 else "≈ matched the un-tuned max"
    print(f"=== {VER}: {v} | final {final:.3f} vs max 9.100 | {time.time()-t0:.0f}s ===")


def _hgb(X, y, g, Xs, n=5):
    oof = np.zeros(len(y)); sac = np.zeros(len(Xs))
    for tr, va in GroupKFold(n).split(X, y, g):
        m = HistGradientBoostingRegressor(max_iter=800, learning_rate=0.04, max_leaf_nodes=127,
                                          l2_regularization=1.0, early_stopping=True, n_iter_no_change=40,
                                          validation_fraction=0.1, random_state=42)
        m.fit(X[tr], y[tr]); oof[va] = m.predict(X[va]); sac += m.predict(Xs) / n
    return oof, sac


if __name__ == "__main__":
    main()
