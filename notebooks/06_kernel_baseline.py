"""v4 — faithful 9.251 kernel feature engine → single-LGB baseline. COLAB.

Ports the public 9.251 build_well (src/kernel9251.py): PF-with-momentum, multi-scale/
stochastic DTW, beam search, multi-scale NCC, GR affine cal, formation/ANCC spatial
imputers. Target = drift (TVT − last_known), so RMSE-on-target == TVT-RMSE.

This run measures whether the PORTED FEATURES (with one LGB) clear the floor and approach
the recipe's level — the GBDT ensemble + hill-climb blend (M3) add the last bit toward 9.25.
Imputers fit on DEV only (clean sacred gate). FE is expensive (numba over ~776 wells) →
cached to data/cache. Set ROGII_ALGO to override the model.
"""
from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import cv, data, kernel9251 as k9, submission, train
from src.config import TRAIN_DIR
from src.evaluate import rmse
from src.observer import Experiment

# Cache the expensive FE in DRIVE (survives the bootstrap's repo re-clone); local fallback.
CACHE = Path(os.environ.get("DRIVE_ROOT") or "data") / "cache"
ALGO = train.default_algo()   # xgb on GPU (T4) — avoids the lgb-CPU OOM + actually uses the GPU


def _build(label, wells, is_train):
    """build_dataset for a well list, cached to data/cache/<label>.parquet."""
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / f"{label}.parquet"
    if fp.exists():
        print(f"  load cached {label} ({fp})")
        return pd.read_parquet(fp)
    t = time.time()
    paths = [data.horizontal_path(w, "train" if is_train else "test") for w in wells]
    df = k9.build_dataset(paths, is_train=is_train, label=label)
    df.to_parquet(fp)
    print(f"  built {label}: {df.shape} in {time.time()-t:.0f}s")
    return df


def main() -> None:
    t0 = time.time()
    dev, sacred = cv.sacred_split(data.list_well_ids("train"))
    maxw = int(os.environ.get("ROGII_MAX_WELLS") or 0)   # quick pipeline validation on a subset
    sfx = ""
    if maxw:
        dev, sacred = dev[:maxw], sacred[:max(20, maxw // 4)]
        sfx = f"_{maxw}"
        print(f"SUBSET mode: dev {len(dev)} | sacred {len(sacred)} (ROGII_MAX_WELLS={maxw})")
    print(f"dev {len(dev)} | sacred {len(sacred)} | fitting imputers on DEV (clean gate)")
    k9.fit_imputers(dev, TRAIN_DIR)

    dev_df = _build("dev_k9" + sfx, dev, True)
    sac_df = _build("sacred_k9" + sfx, sacred, True)
    feats = [c for c in dev_df.columns if c not in {"well", "id", "target"}]
    # float32 to halve RAM; subsample dev TRAIN rows (consecutive 1-ft samples in a well are
    # near-duplicates → every-Nth keeps signal, fixes the 3M-row xgb-DMatrix OOM on 12.7GB).
    stride = int(os.environ.get("ROGII_ROW_STRIDE") or 2)
    X = dev_df[feats].iloc[::stride].astype("float32")
    y = dev_df["target"].to_numpy(np.float32)[::stride]
    g = dev_df["well"].to_numpy()[::stride]
    Xs, ys = sac_df[feats].astype("float32"), sac_df["target"].to_numpy(np.float32)  # sacred FULL
    full_y = dev_df["target"].to_numpy(np.float32)
    del dev_df, sac_df; gc.collect()
    dev_floor, sac_floor = rmse(np.zeros_like(full_y), full_y), rmse(np.zeros_like(ys), ys)
    print(f"X {X.shape} (stride {stride}, {ALGO}) | {len(feats)} features | "
          f"dev_floor {dev_floor:.3f} sac_floor {sac_floor:.3f}")

    exp = Experiment.start(
        version="v4_kernel9251", parent="v0_floor",
        hypothesis=("Ported 9.251 feature engine (PF/DTW/beam/NCC/affine/spatial-imputers) + single LGB. "
                    "Expect to clear the floor decisively and approach ~10-11 (ensemble+blend reach 9.25)."),
        predicted_delta=3.0, confidence="medium",
        feature_changes=["+ full 9.251 build_well features"], pipeline_changes=[f"port baseline ({ALGO})"],
        cloud_or_local="cloud")

    res = train.train_variant("v4_kernel9251", ALGO, X, y, g,
                              save=True, fit_full=True, use_gpu="auto")
    full = joblib.load(train.PROBS / "v4_kernel9251" / "model_full.pkl")
    sac_pred = full.predict(Xs)
    sac_rmse = rmse(ys, sac_pred)
    print(f"\ndev OOF RMSE {res.oof_rmse:.3f} (floor {dev_floor:.3f}) | "
          f"SACRED RMSE {sac_rmse:.3f} (floor {sac_floor:.3f}, Δ {sac_rmse - sac_floor:+.3f})")
    imp = sorted(zip(feats, full.feature_importances_), key=lambda kv: -kv[1])[:15]
    print("top 15 features:", [f"{n}:{int(v)}" for n, v in imp])

    exp.record(oof_score_mean=sac_rmse, oof_score_per_fold=res.fold_rmses, holdout_score=sac_rmse,
               runtime_sec=time.time() - t0,
               extra={"dev_oof": res.oof_rmse, "sac_floor": sac_floor, "sacred_minus_floor": sac_rmse - sac_floor,
                      "n_features": len(feats), "target": "9.25 (full recipe)"})
    exp.note(f"ported 9.251 features; sacred {sac_rmse:.3f} vs floor {sac_floor:.3f} "
             f"({'BEATS' if sac_rmse < sac_floor else 'trails'})")
    exp.commit()

    # test submission (de-residualize: predicted drift + last_known_tvt). Uses the same
    # DEV imputers as training (consistent features). id = "{well}_{iloc}" matches the sample.
    test_df = _build("test_k9", data.list_well_ids("test"), False)
    tvt = full.predict(test_df[feats].astype("float32")) + test_df["last_known_tvt"].to_numpy(float)
    ss = submission.build_submission(dict(zip(test_df["id"], tvt)))
    out = submission.save_submission(ss, "v4_kernel9251")
    print(f"wrote {out}")
    print(f"\n=== v4 done in {time.time()-t0:.0f}s | SACRED {sac_rmse:.3f} (target 9.25, floor {sac_floor:.3f}) ===")


if __name__ == "__main__":
    main()
