"""v5 — M3: LGB×3 + CatBoost×3 ensemble + supervised blend. COLAB. Target ≤ 9.25.

The kernel's full recipe on top of v4's features. CatBoost runs task_type="GPU" (works on
Colab T4 — finally real GBDT GPU use + a decorrelated model); LGB stays CPU (binned,
memory-safe). Blend weights are optimized on the dev GroupKFold OOF and judged on the
SACRED holdout (never tuned to sacred — L48/L53). Stride kept at 8 (same as v4) so this
isolates the ENSEMBLE lift vs v4's 9.49; lower ROGII_ROW_STRIDE later for the data lift.

Reuses the Drive-cached features (no FE rebuild). Set ROGII_ROW_STRIDE to override.
"""
from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import blend, cv, data, kernel9251 as k9, submission, train
from src.config import TRAIN_DIR
from src.evaluate import rmse
from src.observer import Experiment

CACHE = Path(os.environ.get("DRIVE_ROOT") or "data") / "cache"
STRIDE = int(os.environ.get("ROGII_ROW_STRIDE") or 8)
# ROGII_FAST=1 → low-capacity EXPLORATION proxy (LGB num_leaves 63 / cat depth 6): ~2-3x faster on the
# CPU-bound LGBs. The QUALITY config (255 / depth 7) is the DEFAULT and the finalize-on-submission config.
# First use = a FAITHFULNESS check: run FAST at stride-8 and compare sacred to v5_ensemble's 9.155 (same
# recipe at 255). If they track → iterate in FAST mode, finalize on the full config. If 63 ≥ 255 → 255 was
# overfitting, retire it. (num_leaves is HPO — it won't break the wall to 8.2; the point is cheap iteration.)
FAST = os.environ.get("ROGII_FAST") == "1"
_LEAVES, _DEPTH = (63, 6) if FAST else (255, 7)
VER = f"v6s{STRIDE}" + ("_fast" if FAST else "")  # stride/capacity-aware version; v5_ensemble was 255/stride-8
# hypothesis-first predicted improvement over v5's 9.155 (diary rule). FAST: expect ~comparable (capacity
# probe — 63 may even generalize better). Full ladder (8→4→2→1): more data → lower sacred.
PRED_DELTA = 0.0 if FAST else {4: 0.20, 2: 0.35, 1: 0.45}.get(STRIDE, 0.10)

LGB_BASE = dict(num_leaves=_LEAVES, min_child_samples=15, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=0.05)
CAT_BASE = dict(depth=_DEPTH, l2_leaf_reg=2.0, min_data_in_leaf=15)
MODELS = [
    ("lgb", dict(learning_rate=0.025, random_state=42, **LGB_BASE)),
    ("lgb", dict(learning_rate=0.020, random_state=7, **LGB_BASE)),
    ("lgb", dict(learning_rate=0.030, random_state=123, **LGB_BASE)),
    ("cat", dict(learning_rate=0.025, random_seed=42, **CAT_BASE)),
    ("cat", dict(learning_rate=0.020, random_seed=7, **CAT_BASE)),
    ("cat", dict(learning_rate=0.030, random_seed=123, **CAT_BASE)),
]


def main() -> None:
    t0 = time.time()
    dev, sacred = cv.sacred_split(data.list_well_ids("train"))
    k9.fit_imputers(dev, TRAIN_DIR)
    dev_df = pd.read_parquet(CACHE / "dev_k9.parquet")
    sac_df = pd.read_parquet(CACHE / "sacred_k9.parquet")
    feats = [c for c in dev_df.columns if c not in {"well", "id", "target"}]
    X = dev_df[feats].iloc[::STRIDE].astype("float32")
    y = dev_df["target"].to_numpy(np.float32)[::STRIDE]
    g = dev_df["well"].to_numpy()[::STRIDE]
    Xs, ys = sac_df[feats].astype("float32"), sac_df["target"].to_numpy(np.float32)
    del dev_df, sac_df; gc.collect()
    sac_floor = rmse(np.zeros_like(ys), ys)

    # test features (small: 3 wells, ~20s) for the blended submission
    test_df = k9.build_dataset([data.horizontal_path(w, "test") for w in data.list_well_ids("test")],
                               is_train=False, label="test_v5")
    Xt = test_df[feats].astype("float32")
    anchor_t = test_df["last_known_tvt"].to_numpy(float)
    print(f"[{VER}] X {X.shape} (stride {STRIDE}) | leaves {_LEAVES}/depth {_DEPTH} | {len(feats)} feats | "
          f"sac_floor {sac_floor:.3f} | v5 (255/stride8) was 9.155")

    hyp = (f"FAST faithfulness probe: low-capacity profile (LGB leaves {_LEAVES} / cat depth {_DEPTH}) at "
           f"stride {STRIDE} vs v5_ensemble (255/depth7, stride-8 → sacred 9.155). Does the fast proxy track "
           "the quality config (rank-order/gap), and does 255 earn its cost or just overfit?") if FAST else \
          (f"S1a data lever: same 6-model recipe at stride {STRIDE} (v5 was stride-8 → sacred 9.155). More rows "
           "→ GBDTs generalize better and the kernel-gap (LB 9.644 vs kernel 9.251, same 3 wells) shrinks. "
           "Expect sacred to fall toward ~8.7.")
    exp = Experiment.start(
        version=VER, parent="v5_ensemble", hypothesis=hyp,
        predicted_delta=PRED_DELTA, confidence="low" if FAST else "medium",
        pipeline_changes=[f"LGB leaves={_LEAVES}, cat depth={_DEPTH}, stride {STRIDE}"], cloud_or_local="cloud")

    oof_d, sac_d, test_d = {}, {}, {}
    for i, (algo, params) in enumerate(MODELS):
        name = f"{VER}_{algo}{i}"
        res = train.train_variant(name, algo, X, y, g, params=params, save=True, fit_full=True, use_gpu="auto")
        full = joblib.load(train.PROBS / name / "model_full.pkl")
        oof_d[name], sac_d[name], test_d[name] = res.oof, full.predict(Xs), full.predict(Xt)
        print(f"  {name}: dev_oof {res.oof_rmse:.3f} | sacred {rmse(ys, sac_d[name]):.3f}")
        del full, res; gc.collect()

    # supervised blend: weights on dev OOF (leak-free), evaluated on sacred
    w, oof_blend, _ = blend.nm_optimize_oof(oof_d, y, allow_negative=False)
    sac_blend = blend.apply_blend(sac_d, w)
    sac_rmse = rmse(ys, sac_blend)
    simple = rmse(ys, np.mean(list(sac_d.values()), axis=0))
    print(f"\nblend weights: {dict((k, round(v, 3)) for k, v in w.items())}")
    print(f"blend dev-OOF {oof_blend:.3f} | SACRED blend {sac_rmse:.3f} | simple-avg {simple:.3f} "
          f"| floor {sac_floor:.3f} | v4 9.490")

    exp.record(oof_score_mean=sac_rmse, oof_score_per_fold=[rmse(ys, v) for v in sac_d.values()],
               holdout_score=sac_rmse, runtime_sec=time.time() - t0,
               extra={"sac_floor": sac_floor, "simple_avg": simple, "v4_sacred": 9.490,
                      "weights": {k: round(v, 3) for k, v in w.items()}})
    exp.note(f"{VER}: stride {STRIDE} 6-model blend sacred {sac_rmse:.3f} vs v5(stride8) 9.155 vs floor {sac_floor:.3f}")
    exp.commit()

    tvt = blend.apply_blend(test_d, w) + anchor_t
    ss = submission.build_submission(dict(zip(test_df["id"], tvt)))
    out = submission.save_submission(ss, VER)
    print(f"wrote {out}\n=== {VER} done in {time.time()-t0:.0f}s | SACRED {sac_rmse:.3f} "
          f"(v5 stride8 9.155, floor {sac_floor:.3f}) ===")


if __name__ == "__main__":
    main()
