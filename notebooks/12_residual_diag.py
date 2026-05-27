"""Signal search, step 0 — DIAGNOSE where the 9.16 model fails, to direct the hunt. COLAB (GPU).

6 levers flat ⇒ the per-well-alignment paradigm (the 222 feats) is saturated. The missing signal is
in information the 222 DON'T encode. Rather than guess, let the residuals say *where*: slice sacred
error by regime, and rank features by correlation with |residual|. Whatever regime carries the error
is the axis to chase (likely cross-well/field structure the per-well alignment can't see).

Pure analysis (no new model claim) — fast. Reuses the Drive-cached 222.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src import dashboard as dash, train
from src.evaluate import rmse

CACHE = Path(os.environ.get("DRIVE_ROOT") or "data") / "cache"
VER = "v11_resid_diag"
CAT = dict(depth=6, learning_rate=0.03, random_seed=42, l2_leaf_reg=2.0, min_data_in_leaf=15)


def octile_rmse(name, var, resid, nbins=8):
    s = pd.Series(var)
    try:
        q = pd.qcut(s, nbins, duplicates="drop")
    except Exception:
        print(f"  [{name}] (constant/again unbinnable, skip)"); return
    g = pd.DataFrame({"bin": q, "r2": resid ** 2}).groupby("bin", observed=True)["r2"].agg(["mean", "count"])
    g["rmse"] = np.sqrt(g["mean"])
    lo, hi = g["rmse"].iloc[0], g["rmse"].iloc[-1]
    print(f"\n[{name}] residual RMSE by octile (low→high):  {'  '.join(f'{v:.2f}' for v in g['rmse'])}"
          f"   spread {hi-lo:+.2f}")


def main() -> None:
    dev_df = pd.read_parquet(CACHE / "dev_k9.parquet")
    sac_df = pd.read_parquet(CACHE / "sacred_k9.parquet")
    feats = [c for c in dev_df.columns if c not in {"well", "id", "target"}]
    yd = dev_df["target"].to_numpy(np.float32); gd = dev_df["well"].to_numpy(); ys = sac_df["target"].to_numpy(np.float32)

    dash.goal_banner(VER, "residual diagnosis (single cat)", "WHERE does 9.16 fail? → which axis carries the missing signal")
    r = train.train_variant(VER, "cat", dev_df[feats].astype("float32"), yd, gd,
                            X_test=sac_df[feats].astype("float32"), params=CAT, save=False, use_gpu="auto")
    pred = r.test_pred; resid = ys - pred; ar = np.abs(resid)
    print(f"\n[{VER}] sacred rmse {rmse(ys, pred):.3f} | mean|resid| {ar.mean():.2f} | p50 {np.median(ar):.2f} | p90 {np.quantile(ar,0.9):.2f}")

    # slice error by regime (columns the kernel already exposes)
    regimes = [("horizon (md_since)", "md_since"), ("frac-through-well", "frac"),
               ("align-uncertainty sig_std", "sig_std"), ("dtw_stoch_std", "dtw_stoch_std"),
               ("spatial knn_dist", "spatial_knn_dist"), ("formation std", "form_std_d"),
               ("dz (vertical)", "dz"), ("|true drift|", "__abs_target")]
    for name, col in regimes:
        if col == "__abs_target":
            octile_rmse(name, np.abs(ys), resid)
        elif col in sac_df.columns:
            octile_rmse(name, sac_df[col].to_numpy(), resid)
        else:
            print(f"  ({col} not in 222, skip)")

    # which features most correlate with |residual| → what regime the model is systematically off in
    X = sac_df[feats].fillna(0.0)
    cors = X.apply(lambda c: float(np.corrcoef(c, ar)[0, 1]) if c.std() > 1e-9 else 0.0)
    print("\n[features most correlated with |residual| — signal the model can't act on]:")
    print(cors.abs().sort_values(ascending=False).head(15).round(3).to_string())

    # spatial litmus: do wells with closer neighbors have lower error? (is cross-well signal latent?)
    if "spatial_knn_dist" in sac_df.columns:
        d = sac_df["spatial_knn_dist"].to_numpy(); m = np.median(d)
        near, far = ar[d < m].mean(), ar[d >= m].mean()
        verdict = "← spatial signal latent (chase cross-well)" if near < far - 0.15 else "← no clear spatial gradient"
        print(f"\n[spatial litmus] mean|resid|: near-neighbors {near:.2f} vs far {far:.2f}   {verdict}")
    print(f"\n=== {VER}: diagnosis done — the regime with the biggest RMSE spread is where the missing signal lives ===")


if __name__ == "__main__":
    main()
