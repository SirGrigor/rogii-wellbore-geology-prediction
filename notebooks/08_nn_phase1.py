"""S2 Phase-1 — go/no-go for the sequence NN. COLAB (GPU + high-RAM).

Question (decided on SACRED, never the 3-well LB): does a 1D-CNN on the RAW GR window +
context carry signal ORTHOGONAL to the GBDT's 222 alignment features? Test = add the NN as
one more member of the supervised dev-OOF blend and compare sacred WITH vs WITHOUT it.

  G-NN1: if NN's blend weight ≈ 0 and sacred doesn't drop ≥ ~0.1 → raw GR is saturated;
         don't build the attention model (Phase 2). Fall back to Strategy 2 (new features).

Reuses the Drive-cached GBDT features (dev_k9/sacred_k9). NN trains on raw GR (nn_data/nn_model).
Honest by-well OOF for both sides → blend is leak-free (L48/L53). Env: ROGII_NN_WINDOW, ROGII_NN_EPOCHS.
"""
from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import blend, cv, dashboard as dash, data, nn_data, nn_model, train
from src.config import TRAIN_DIR
from src.evaluate import rmse
from src.observer import Experiment

CACHE = Path(os.environ.get("DRIVE_ROOT") or "data") / "cache"
WINDOW = int(os.environ.get("ROGII_NN_WINDOW") or 128)
EPOCHS = int(os.environ.get("ROGII_NN_EPOCHS") or 40)
VER = "v7_nn_p1"

# lean GBDT (cat depth-6 + one lgb), matching the v6s4_fast recipe at sacred ~9.16
GBDT = [
    ("lgb0", "lgb", dict(num_leaves=63, learning_rate=0.025, random_state=42,
                         min_child_samples=15, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=0.05)),
    ("cat3", "cat", dict(depth=6, learning_rate=0.025, random_seed=42, l2_leaf_reg=2.0, min_data_in_leaf=15)),
    ("cat4", "cat", dict(depth=6, learning_rate=0.020, random_seed=7, l2_leaf_reg=2.0, min_data_in_leaf=15)),
    ("cat5", "cat", dict(depth=6, learning_rate=0.030, random_seed=123, l2_leaf_reg=2.0, min_data_in_leaf=15)),
]


def main() -> None:
    t0 = time.time()
    dev_w, sac_w = cv.sacred_split(data.list_well_ids("train"))
    test_w = data.list_well_ids("test")

    # ---------- NN: raw-GR window + context, by-well CV ----------
    dd = nn_data.build(dev_w, "train")
    sd = nn_data.build(sac_w, "train", gr_stats=dd.gr_stats, ctx_stats=dd.ctx_stats)
    td = nn_data.build(test_w, "test", gr_stats=dd.gr_stats, ctx_stats=dd.ctx_stats)
    dash.goal_banner(VER, f"1D-CNN GR±{WINDOW} + {len(nn_data.CTX_COLS)} ctx, {EPOCHS}ep",
                     "does raw-GR add signal orthogonal to the 222 feats? NN⊕GBDT vs GBDT-only on sacred")
    exp = Experiment.start(
        version=VER, parent="v6s8_fast",
        hypothesis=("A 1D-CNN on the raw GR window + geometry context is decorrelated from the GBDT's "
                    "222 alignment features, so adding it to the dev-OOF blend lowers SACRED below the "
                    "~9.16 GBDT floor. Go/no-go for the Phase-2 attention build."),
        predicted_delta=0.15, confidence="low",
        pipeline_changes=[f"1D-CNN window={WINDOW}, {EPOCHS} epochs"], cloud_or_local="cloud")

    print(f"[{VER}] NN dev {len(dd.samples)} / sacred {len(sd.samples)} / test {len(td.samples)} samples")
    nn_oof, nn_sac, nn_test, nn_fr = nn_model.train_cv(dd, sd, td, window=WINDOW, epochs=EPOCHS)
    nn_oof_by = dict(zip(dd.ids, nn_oof)); nn_sac_by = dict(zip(sd.ids, nn_sac))

    # ---------- GBDT: cached feats, lean recipe, honest OOF + bagged sacred ----------
    dev_df = pd.read_parquet(CACHE / "dev_k9.parquet")
    sac_df = pd.read_parquet(CACHE / "sacred_k9.parquet")
    feats = [c for c in dev_df.columns if c not in {"well", "id", "target"}]
    Xd = dev_df[feats].astype("float32"); yd = dev_df["target"].to_numpy(np.float32)
    gd = dev_df["well"].to_numpy(); Xs = sac_df[feats].astype("float32")
    gb_oof, gb_sac = {}, {}
    for name, algo, params in GBDT:
        r = train.train_variant(f"{VER}_{name}", algo, Xd, yd, gd, X_test=Xs, params=params,
                                save=False, use_gpu="auto")
        gb_oof[name] = dict(zip(dev_df["id"], r.oof))
        gb_sac[name] = dict(zip(sac_df["id"], r.test_pred))
        del r; gc.collect()

    # ---------- align everything by id ----------
    dev_ids = [i for i in dev_df["id"].tolist() if i in nn_oof_by]
    sac_ids = [i for i in sac_df["id"].tolist() if i in nn_sac_by]
    print(f"[align] dev ids matched {len(dev_ids)}/{len(dev_df)} | sacred {len(sac_ids)}/{len(sac_df)}"
          f"  (low overlap ⇒ id/iloc mismatch between nn_data and the kernel cache — investigate)")
    y_dev = dev_df.set_index("id").loc[dev_ids, "target"].to_numpy(np.float32)
    y_sac = sac_df.set_index("id").loc[sac_ids, "target"].to_numpy(np.float32)
    pick = lambda by, ids: np.array([by[i] for i in ids], np.float32)
    oof_gbdt = {n: pick(gb_oof[n], dev_ids) for n, _, _ in GBDT}
    sac_gbdt = {n: pick(gb_sac[n], sac_ids) for n, _, _ in GBDT}
    oof_nn = pick(nn_oof_by, dev_ids); sac_nn = pick(nn_sac_by, sac_ids)

    # ---------- the go/no-go: blend WITH vs WITHOUT the NN, on sacred ----------
    w_wo, _, _ = blend.nm_optimize_oof(oof_gbdt, y_dev, allow_negative=False)
    sac_wo = rmse(y_sac, blend.apply_blend(sac_gbdt, w_wo))
    oof_all = {"nn": oof_nn, **oof_gbdt}; sac_all = {"nn": sac_nn, **sac_gbdt}
    w_w, _, _ = blend.nm_optimize_oof(oof_all, y_dev, allow_negative=False)
    sac_w = rmse(y_sac, blend.apply_blend(sac_all, w_w))

    nn_sac_rmse = rmse(y_sac, sac_nn)
    # residual decorrelation of NN vs the GBDT blend (lower |corr| = more orthogonal)
    gb_blend_sac = blend.apply_blend(sac_gbdt, w_wo)
    corr = float(np.corrcoef(y_sac - sac_nn, y_sac - gb_blend_sac)[0, 1])

    print(f"\n[{VER}] NN-alone sacred {nn_sac_rmse:.3f} | GBDT-only sacred {sac_wo:.3f} | "
          f"NN⊕GBDT sacred {sac_w:.3f} | NN weight {w_w.get('nn', 0):.3f} | resid corr {corr:.3f}")
    print(f"blend WITH nn: {dict((k, round(v,3)) for k,v in w_w.items())}")

    exp.record(oof_score_mean=sac_w, oof_score_per_fold=[float(x) for x in nn_fr],
               holdout_score=sac_w, runtime_sec=time.time() - t0,
               extra={"nn_alone_sacred": nn_sac_rmse, "gbdt_only_sacred": sac_wo,
                      "ens_sacred": sac_w, "nn_weight": w_w.get("nn", 0.0),
                      "resid_corr": corr, "nn_fold_rmse": nn_fr, "window": WINDOW})
    exp.note(f"NN⊕GBDT {sac_w:.3f} vs GBDT-only {sac_wo:.3f} (NN w={w_w.get('nn',0):.3f}, corr={corr:.2f})")
    exp.commit()

    # save NN preds for a later ensemble/submission
    out = CACHE / "nn_p1"; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "preds.npz", dev_ids=np.array(dev_ids), sac_ids=np.array(sac_ids),
             test_ids=np.array(td.ids), nn_oof=oof_nn, nn_sac=sac_nn, nn_test=nn_test)

    dash.verdict(VER, sac_w, time.time() - t0, simple_avg=sac_wo, parent=9.155)
    verdict = ("✅ GO — NN adds orthogonal signal" if sac_w < sac_wo - 0.1
               else "⚠ marginal" if sac_w < sac_wo - 0.02 else "✗ NO-GO — raw GR saturated")
    print(f"=== {VER}: {verdict} | NN⊕GBDT {sac_w:.3f} vs GBDT-only {sac_wo:.3f} "
          f"| NN-alone {nn_sac_rmse:.3f} | corr {corr:.2f} | {time.time()-t0:.0f}s ===")


if __name__ == "__main__":
    main()
