"""20 — Soft-DTW signal lever (proof-of-life). COLAB GPU (CPU fallback OK for tiny runs).

Conservative config: stride-16 query / band W=80 / d=32 / 3 folds / 8 epochs (~2h GPU). Trains the
soft-DTW model (siamese 1D-CNN GR encoder + Cuturi recursion + autograd-α readout) end-to-end on
TVT drift, gets OOF/sacred/test predictions, then Caruana-blends with a strong single-model kernel
baseline (cat depth-7, the 9.120 solo from 17_max). Decision gate (only basis for the next chunk):

  delta = 9.100 − sacred(Caruana[sdtw, cat])
    ≥ +0.05  → REAL SIGNAL → build Cuturi closed-form α-backward + quality config (~5× faster)
    ≈ 0     → soft-DTW honest end at conservative config; harvest lessons
    < 0     → soft-DTW hurts the blend; drop the lever

Env: ROGII_SDTW_EPOCHS (8), ROGII_SDTW_FOLDS (3), ROGII_SDTW_BAND (80).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import blend, cv, dashboard as dash, data, sdtw_data as sd, sdtw_model as sm, train
from src.evaluate import rmse
from src.observer import Experiment

CACHE = Path(os.environ.get("DRIVE_ROOT") or "data") / "cache"
VER = "v19_sdtw"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# proof-of-life config (compute-conscious; quality config behind Cuturi-backward gate)
SDTW_CFG = dict(d=32, gamma=1.0, tau=0.5,
                epochs=int(os.environ.get("ROGII_SDTW_EPOCHS") or 8),
                lr=3e-3, query_stride=16)
N_FOLDS = int(os.environ.get("ROGII_SDTW_FOLDS") or 3)
BAND = int(os.environ.get("ROGII_SDTW_BAND") or 80)
WIN = 32


def _align(d: dict[str, float], ids: list[str], name: str) -> np.ndarray:
    miss = sum(1 for i in ids if i not in d)
    print(f"    {name}: {len(d)} sdtw preds | aligning {len(ids)} cache ids | missing {miss}")
    return np.array([d.get(i, 0.0) for i in ids], dtype=np.float32)


def main() -> None:
    t0 = time.time()
    print(f"=== {VER} (soft-DTW proof-of-life) | device={DEVICE} | epochs={SDTW_CFG['epochs']} ===")
    dev_w, sac_w = cv.sacred_split(data.list_well_ids("train"))
    test_w = data.list_well_ids("test")

    dash.goal_banner(VER, "soft-DTW signal lever (Cuturi recursion + siamese GR encoder + α readout)",
                     "is there NEW alignment signal beyond the kernel GBDT? sacred(sdtw ⊕ cat) vs 9.100")
    exp = Experiment.start(version=VER, parent="v16_max",
                           hypothesis="Differentiable soft-DTW with learned cost finds horizontal-vs-typewell "
                                      "alignment signal the kernel GBDT lacks; Caruana blend pushes sacred below 9.100.",
                           predicted_delta=0.05, confidence="medium",
                           pipeline_changes=["soft-DTW (Cuturi 2017) + siamese 1D-CNN GR encoder + α readout"],
                           cloud_or_local="cloud")

    # --- (1) build per-well episodes ---
    print(f"[1] building episodes (win={WIN} band={BAND})")
    t = time.time()
    dev_eps = sd.build(dev_w, "train", win=WIN, band=BAND)
    sac_eps = sd.build(sac_w, "train", win=WIN, band=BAND)
    tst_eps = sd.build(test_w, "test", win=WIN, band=BAND)
    print(f"    dev {len(dev_eps)} | sac {len(sac_eps)} | test {len(tst_eps)} | {time.time()-t:.0f}s")

    # --- (2) train soft-DTW with GroupKFold by well ---
    print(f"[2] training soft-DTW ({N_FOLDS} folds, {SDTW_CFG['epochs']} epochs, stride={SDTW_CFG['query_stride']})")
    t = time.time()
    res = sm.train_cv(dev_eps, sacred_eps=sac_eps, test_eps=tst_eps,
                      n_folds=N_FOLDS, device=DEVICE, **SDTW_CFG)
    print(f"    trained in {(time.time()-t)/60:.1f} min | fold_rmse {[round(r,3) for r in res.fold_rmse]}")

    # --- (3) align sdtw drift preds to kernel cache id order ---
    print("[3] aligning sdtw preds to kernel cache id order")
    dev_cache = pd.read_parquet(CACHE / "dev_k9.parquet")
    sac_cache = pd.read_parquet(CACHE / "sacred_k9.parquet")
    tst_cache = pd.read_parquet(CACHE / "test_k9.parquet")
    sdtw_dev = _align(res.oof, dev_cache["id"].tolist(), "dev")
    sdtw_sac = _align(res.sac or {}, sac_cache["id"].tolist(), "sacred")
    sdtw_tst = _align(res.test or {}, tst_cache["id"].tolist(), "test")

    # sdtw predicts DRIFT (TVT − last_known). Convert back to TVT for blending with the kernel by
    # adding the per-well last_known (from the SDTWEpisode, not the cache — kernel cache stores 222
    # features but not last_tvt). Build well→last_tvt lookup from all episodes, then add by id.
    lt = {e.well: e.last_tvt for e in (*dev_eps, *sac_eps, *tst_eps)}
    def _to_tvt(arr: np.ndarray, ids: list[str]) -> np.ndarray:
        return arr + np.array([lt.get(i.rsplit("_", 1)[0], 0.0) for i in ids], dtype=np.float32)
    sdtw_dev_tvt = _to_tvt(sdtw_dev, dev_cache["id"].tolist())
    sdtw_sac_tvt = _to_tvt(sdtw_sac, sac_cache["id"].tolist())
    sdtw_tst_tvt = _to_tvt(sdtw_tst, tst_cache["id"].tolist())

    y_dev = dev_cache["target"].to_numpy(np.float32)
    y_sac = sac_cache["target"].to_numpy(np.float32)
    sdtw_alone = rmse(y_sac, sdtw_sac_tvt)
    print(f"    sdtw alone sacred RMSE = {sdtw_alone:.3f}")

    # --- (4) kernel baseline (cat depth-7, the 9.120 solo from 17_max) — one fit, both eval sets ---
    print("[4] training kernel baseline (cat depth-7, single seed)")
    base = [c for c in dev_cache.columns if c not in {"well", "id", "target"}]
    Xd = dev_cache[base].astype("float32")
    Xs = sac_cache[base].astype("float32")
    Xt = tst_cache[base].astype("float32")
    Xst = pd.concat([Xs, Xt], axis=0, ignore_index=True)              # one fit, then split predictions
    gd = dev_cache["well"].to_numpy()
    cat_p = dict(depth=7, learning_rate=0.04, l2_leaf_reg=3.0, border_count=254, random_seed=42)
    r = train.train_variant(f"{VER}_cat", "cat", Xd, y_dev, gd, X_test=Xst, params=cat_p, save=False, use_gpu="auto")
    cat_dev_oof = r.oof
    cat_sac = r.test_pred[:len(Xs)]
    cat_tst = r.test_pred[len(Xs):]
    print(f"    cat sacred RMSE = {rmse(y_sac, cat_sac):.3f}")

    # --- (5) Caruana blend (fit on dev OOF, verdict on sacred) ---
    print("[5] Caruana blend (fit on dev OOF, verdict on sacred)")
    oof_d = {"sdtw": sdtw_dev_tvt, "cat": cat_dev_oof}
    sac_d = {"sdtw": sdtw_sac_tvt, "cat": cat_sac}
    tst_d = {"sdtw": sdtw_tst_tvt, "cat": cat_tst}
    w_car, oof_score, info = blend.caruana_select(oof_d, y_dev)
    w_named = dict(zip(["sdtw", "cat"], w_car))
    s_blend = rmse(y_sac, blend.apply_blend(sac_d, w_car))
    print(f"    Caruana weights = {w_named}  | dev-OOF blend = {oof_score:.3f}")
    print(f"    sacred blend = {s_blend:.3f}  |  cat-alone {rmse(y_sac, cat_sac):.3f}  |  sdtw-alone {sdtw_alone:.3f}")
    print(f"    vs max-build 9.100 / v5 9.155")

    delta = 9.100 - s_blend
    exp.record(oof_score_mean=s_blend, oof_score_per_fold=[float(x) for x in res.fold_rmse], holdout_score=s_blend,
               runtime_sec=time.time() - t0, extra={
                   "sdtw_sacred_alone": float(sdtw_alone),
                   "cat_sacred_alone": float(rmse(y_sac, cat_sac)),
                   "blend_sacred": float(s_blend),
                   "caruana_weights": {k: float(v) for k, v in w_named.items()},
                   "config": {**SDTW_CFG, "n_folds": N_FOLDS, "band": BAND, "win": WIN},
               })
    exp.note(f"Soft-DTW proof-of-life: alone {sdtw_alone:.3f}, blend {s_blend:.3f} vs 9.100 (Δ {delta:+.3f})")
    exp.commit()
    dash.verdict(VER, s_blend, time.time() - t0, simple_avg=9.100, parent=9.100)

    verdict = ("✅ REAL SIGNAL — build Cuturi closed-form α-backward + quality config" if delta >= 0.05
               else "≈ flat — soft-DTW honest end at conservative config; harvest" if abs(delta) < 0.05
               else "✗ REGRESSION — soft-DTW hurts the blend; drop the lever")
    print(f"\n=== {VER}: {verdict} | sacred {s_blend:.3f} | Δ vs 9.100 = {delta:+.3f} | {(time.time()-t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()
