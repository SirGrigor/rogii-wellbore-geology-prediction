"""S3c — tabular foundation model (TabPFN) as a decorrelated stack member. COLAB (GPU + internet).

The last evidence-backed swing: an in-context-learning estimator is a fundamentally different
function family from GBDT (kojimar's stack weighted the foundation-model branch heavily). TabPFN
sees only a ~10K-row context (its limit) vs the GBDT's 3M rows, so it WILL be weaker alone — the
whole bet is DECORRELATION. Go/no-go on sacred: does TabPFN⊕GBDT beat GBDT-alone?

Leak-free: by-well context/probe split inside dev (TabPFN's context wells ≠ probe wells); blend
weight fit on the held-out dev probe; verdict on a sacred subsample. Env: ROGII_TABPFN_CTX/EVAL.
(kojimar used 'TabICL'; we use TabPFN v2's regressor — same in-context foundation-model idea, but
regression-native. Decided on sacred; the 3-well LB is noise.)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import blend, dashboard as dash, train
from src.evaluate import rmse
from src.observer import Experiment

CACHE = Path(os.environ.get("DRIVE_ROOT") or "data") / "cache"
VER = "v10_tabpfn"
N_CTX = int(os.environ.get("ROGII_TABPFN_CTX") or 10000)
N_EVAL = int(os.environ.get("ROGII_TABPFN_EVAL") or 20000)
CAT = dict(depth=6, learning_rate=0.03, random_seed=42, l2_leaf_reg=2.0, min_data_in_leaf=15)


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(7)
    dev_df = pd.read_parquet(CACHE / "dev_k9.parquet")
    sac_df = pd.read_parquet(CACHE / "sacred_k9.parquet")
    feats = [c for c in dev_df.columns if c not in {"well", "id", "target"}]
    yd = dev_df["target"].to_numpy(np.float32); gd = dev_df["well"].to_numpy(); ys = sac_df["target"].to_numpy(np.float32)
    Xd = dev_df[feats].astype("float32").fillna(0.0)
    Xs = sac_df[feats].astype("float32").fillna(0.0)

    dash.goal_banner(VER, "TabPFN (in-context) ⊕ GBDT",
                     "orthogonal foundation model — last swing; the bet is decorrelation, not strength")
    exp = Experiment.start(
        version=VER, parent="v6s8_fast",
        hypothesis=("TabPFN (in-context tabular foundation model) is a different function family from GBDT → "
                    "decorrelated residuals → TabPFN⊕GBDT beats GBDT-alone on sacred, even though TabPFN-alone "
                    "(10K context vs 3M rows) is weaker. Last evidence-backed swing before harvest."),
        predicted_delta=0.10, confidence="low",
        pipeline_changes=["TabPFN stack member"], cloud_or_local="cloud")

    # GBDT baseline: honest full-dev OOF + bagged sacred
    r = train.train_variant(f"{VER}_cat", "cat", Xd, yd, gd, X_test=Xs, params=CAT, save=False, use_gpu="auto")
    gb_oof, gb_sac = r.oof, r.test_pred
    print(f"  [gbdt] dev_oof {r.oof_rmse:.3f} | sacred {rmse(ys, gb_sac):.3f}")

    # by-well context/probe split inside dev (leak-free) + subsamples
    wells = np.array(sorted(set(gd))); rng.shuffle(wells)
    cut = int(len(wells) * 0.8)
    ctx_w = set(wells[:cut].tolist())
    ctx_mask = np.array([w in ctx_w for w in gd])
    ctx_idx = np.where(ctx_mask)[0]; probe_idx = np.where(~ctx_mask)[0]
    ctx_sel = rng.choice(ctx_idx, min(N_CTX, len(ctx_idx)), replace=False)
    probe_sel = rng.choice(probe_idx, min(N_EVAL, len(probe_idx)), replace=False)
    sac_sel = rng.choice(len(ys), min(N_EVAL, len(ys)), replace=False)

    if not os.environ.get("TABPFN_TOKEN"):
        raise SystemExit(
            "TabPFN needs a license token (else it blocks on an interactive prompt in headless Colab).\n"
            "  1. Accept the license at https://ux.priorlabs.ai  (License tab)\n"
            "  2. Copy your key from https://ux.priorlabs.ai/account\n"
            "  3. In the Colab runner, BEFORE the bootstrap cell: os.environ['TABPFN_TOKEN']='<key>'\n"
            "  4. Re-run. (The token propagates to the training subprocess via os.environ.)")
    try:
        from tabpfn import TabPFNRegressor
    except Exception as e:
        raise SystemExit(f"TabPFN not installed ({e}). Add 'tabpfn' to bootstrap DEPS (internet ON downloads weights).")
    reg = TabPFNRegressor(device="cuda", ignore_pretraining_limits=True)
    print(f"[tabpfn] fit context {len(ctx_sel)} rows × {len(feats)} feats; predict probe {len(probe_sel)} + sacred {len(sac_sel)}…")
    reg.fit(Xd.iloc[ctx_sel].to_numpy(np.float32), yd[ctx_sel])
    tp_probe = np.asarray(reg.predict(Xd.iloc[probe_sel].to_numpy(np.float32)), float)
    tp_sac = np.asarray(reg.predict(Xs.iloc[sac_sel].to_numpy(np.float32)), float)

    gb_probe, y_probe = gb_oof[probe_sel], yd[probe_sel]
    gb_sac_e, y_sac_e = gb_sac[sac_sel], ys[sac_sel]
    tp_alone = rmse(y_sac_e, tp_sac); gb_alone = rmse(y_sac_e, gb_sac_e)
    corr = float(np.corrcoef(y_probe - tp_probe, y_probe - gb_probe)[0, 1])
    w, _, _ = blend.nm_optimize_oof({"tabpfn": tp_probe, "gbdt": gb_probe}, y_probe, allow_negative=False)
    ens = rmse(y_sac_e, blend.apply_blend({"tabpfn": tp_sac, "gbdt": gb_sac_e}, w))

    print(f"\n[{VER}] TabPFN-alone {tp_alone:.3f} | GBDT-alone {gb_alone:.3f} | TabPFN⊕GBDT {ens:.3f} "
          f"| tabpfn weight {w.get('tabpfn', 0):.3f} | resid corr {corr:.3f}  (sacred {len(sac_sel)}-row subsample)")

    exp.record(oof_score_mean=ens, oof_score_per_fold=[float(tp_alone), float(gb_alone), float(ens)],
               holdout_score=ens, runtime_sec=time.time() - t0,
               extra={"tabpfn_alone": tp_alone, "gbdt_alone": gb_alone, "ens": ens,
                      "tabpfn_weight": w.get("tabpfn", 0.0), "resid_corr": corr, "n_ctx": len(ctx_sel)})
    exp.note(f"TabPFN⊕GBDT {ens:.3f} vs GBDT-alone {gb_alone:.3f} (w={w.get('tabpfn',0):.3f}, corr={corr:.2f})")
    exp.commit()

    dash.verdict(VER, ens, time.time() - t0, simple_avg=gb_alone, parent=9.155)
    v = ("✅ GO — foundation model decorrelates + helps" if ens < gb_alone - 0.05
         else "⚠ marginal" if ens < gb_alone - 0.02 else "✗ NO-GO — no orthogonal lift → harvest")
    print(f"=== {VER}: {v} | ⊕ {ens:.3f} vs GBDT {gb_alone:.3f} | corr {corr:.2f} | {time.time()-t0:.0f}s ===")


if __name__ == "__main__":
    main()
