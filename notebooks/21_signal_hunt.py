"""21 — Signal Hunt diagnostic on the 9.100 dev OOF. COLAB (CPU-fine; no GPU needed).

Tests whether the residuals of our current best model contain LURKING SIGNAL (avoidable bias the
model failed to extract) or are NOISE (irreducible). Headline = one number: R²_residual_boost.

  R² > 0.05  →  STRUCTURE EXISTS → SHAP names the features → FE recipe writes itself
  R² ≈ 0     →  noise ceiling PROVEN → alignment-paradigm is exhausted → pivot paradigm

Pipeline:
  1. Load dev_k9 cache (kernel 222 feats) + W1 (horizon geom)
  2. Fit lightweight 5-fold lgb GroupKFold-by-well to get OOF predictions
     (lighter than 17_max's deep stack — residual STRUCTURE is largely model-agnostic at the
     alignment-paradigm ceiling; a quick lgb-only OOF gives 90% of the truth in <10% the time)
  3. Run src.signal_hunt.hunt(p_oof, y, X, groups) → numbers + report.md + features.csv
  4. Plus rogii-specific slice plots: residual vs MD-from-PS / spatial (X,Y) / typewell distance

Decision drives the NEXT chunk:
  ceiling proven  →  soft-DTW dies; pivot to non-alignment vector (spatial-only, TabPFN per-well,
                     formation recovery via synth-decoder)
  structure found →  top SHAP features dictate the FE; build them; re-test
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src import cv, data, features_extra as fx, signal_hunt
from src.evaluate import rmse
from src.observer import Experiment

CACHE = Path(os.environ.get("DRIVE_ROOT") or "data") / "cache"
OUT = Path("docs/signal_hunt"); OUT.mkdir(parents=True, exist_ok=True)
VER = "v20_hunt"


def _oof_lgb(X: np.ndarray, y: np.ndarray, g: np.ndarray, *, n_folds: int = 5, seed: int = 42) -> np.ndarray:
    """Lightweight 5-fold lgb OOF (matches our prod feature set; ~10 min on dev_k9)."""
    oof = np.zeros(len(y), dtype=np.float32)
    for k, (tr, va) in enumerate(GroupKFold(n_folds).split(X, y, g)):
        m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.04, num_leaves=127,
                              min_child_samples=20, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=3.0, random_state=seed,
                              n_jobs=-1, verbose=-1)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = m.predict(X[va])
        print(f"   fold {k+1}/{n_folds}  RMSE {rmse(y[va], oof[va]):.3f}")
    return oof


def _slice_plots(residual: np.ndarray, dev: pd.DataFrame, out_path: Path) -> None:
    """Rogii-specific localization: where do the residuals live?"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    # residual vs MD-from-PS (need MD-like column; kernel cache has 'rel_md' or similar geometry feat — fall back to row index per well)
    for col in ("rel_md", "md_since_ps", "dz_since_ps"):
        if col in dev.columns:
            ax.scatter(dev[col].to_numpy()[:50000], residual[:50000], s=1, alpha=0.2)
            ax.set_xlabel(col); break
    else:
        ax.text(0.5, 0.5, "no MD-from-PS column found", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("residual (y − p_oof)"); ax.set_title("residual vs MD-from-PS")
    ax.axhline(0, color="k", linewidth=0.5)

    ax = axes[0, 1]                                    # spatial heatmap (residual MSE per (X,Y) tile)
    xcol = next((c for c in ("X", "x_med", "x_h_med") if c in dev.columns), None)
    ycol = next((c for c in ("Y", "y_med", "y_h_med") if c in dev.columns), None)
    if xcol and ycol:
        sc = ax.scatter(dev[xcol], dev[ycol], c=np.abs(residual), cmap="viridis", s=1, alpha=0.4,
                        vmax=np.quantile(np.abs(residual), 0.95))
        plt.colorbar(sc, ax=ax, label="|residual|")
        ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.set_title("|residual| spatial map")
    else:
        ax.text(0.5, 0.5, "no X/Y cols found", ha="center", va="center", transform=ax.transAxes)

    ax = axes[1, 0]                                    # per-well RMSE distribution
    per_well = pd.DataFrame({"well": dev["well"], "sq": residual ** 2}).groupby("well")["sq"].mean() ** 0.5
    ax.hist(per_well, bins=50)
    ax.axvline(rmse_all := float(np.sqrt((residual ** 2).mean())), color="r", label=f"overall {rmse_all:.2f}")
    ax.set_xlabel("per-well RMSE"); ax.set_ylabel("# wells"); ax.legend()
    ax.set_title(f"per-well RMSE dist  (median {per_well.median():.2f}  p90 {per_well.quantile(0.9):.2f})")

    ax = axes[1, 1]                                    # residual autocorrelation within wells (avg lag-1)
    autocorr = []
    for w, g in pd.DataFrame({"w": dev["well"], "r": residual}).groupby("w"):
        r = g["r"].to_numpy()
        if len(r) >= 5:
            autocorr.append(float(np.corrcoef(r[:-1], r[1:])[0, 1]))
    ax.hist(autocorr, bins=40); ax.set_xlabel("lag-1 autocorr of residual within well")
    ax.set_title(f"within-well autocorr  (median {np.median(autocorr):+.3f})")
    fig.suptitle(f"signal hunt — slice maps  (current RMSE {rmse_all:.3f})", y=1.0)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)
    print(f"   slice_map → {out_path}")


def main() -> None:
    t0 = time.time()
    if not (CACHE / "dev_k9.parquet").exists():
        raise SystemExit(f"missing {CACHE/'dev_k9.parquet'} — run notebooks/06_kernel_baseline.py to build cache first")

    print(f"=== {VER} (signal hunt diagnostic) | CACHE={CACHE} ===")
    exp = Experiment.start(version=VER, parent="v16_max",
                           hypothesis="Residuals of the current 9.1-RMSE stack contain LURKING SIGNAL "
                                      "extractable from the kernel+W1 feature set (R²_residual_boost > 0.05) "
                                      "→ if true, FE on top SHAP features lifts past 9.1.",
                           predicted_delta=0.0,                  # diagnostic, no direct delta
                           confidence="medium",
                           pipeline_changes=["residual-boost diagnostic (R² + adv-AUC + Bayes-floor)"],
                           cloud_or_local="cloud")

    # --- (1) load dev_k9 cache + W1 ---
    print("[1] loading dev_k9 + W1 features")
    dev_w, _ = cv.sacred_split(data.list_well_ids("train"))
    dev = pd.read_parquet(CACHE / "dev_k9.parquet")
    base = [c for c in dev.columns if c not in {"well", "id", "target"}]
    w1 = fx.build_extra(dev_w, "train")[fx.W1]
    dev = dev.join(w1, on="id"); dev[fx.W1] = dev[fx.W1].fillna(0)
    feats = base + fx.W1
    print(f"   dev rows {len(dev):,}  features {len(feats)}")

    # --- (2) lightweight OOF (lgb only, 5 folds) ---
    print("[2] generating OOF via 5-fold lgb (GroupKFold by well)")
    X = dev[feats].astype("float32").to_numpy()
    y = dev["target"].to_numpy(np.float32)
    g = dev["well"].to_numpy()
    t = time.time()
    p_oof = _oof_lgb(X, y, g)
    base_rmse = rmse(y, p_oof)
    print(f"   local lgb OOF RMSE = {base_rmse:.3f}  ({(time.time()-t)/60:.1f} min)")
    print(f"   (for reference: 17_max deep-stack sacred = 9.100; this lighter OOF is the residual-structure proxy)")

    # --- (3) SIGNAL HUNT ---
    print("[3] running 4-layer signal hunt")
    r = signal_hunt.hunt(p_oof=p_oof, y=y, X=dev[feats], groups=g,
                         feature_names=feats, n_top_features=20, out_dir=OUT)

    # --- (4) rogii slice plots ---
    print("[4] rendering rogii slice maps")
    _slice_plots(y - p_oof, dev, OUT / "slice_map.png")

    # --- (5) verdict ---
    print(f"\n=== {VER} VERDICT ===")
    print(f"  current RMSE (light OOF)   = {r.current_rmse:.3f}")
    print(f"  irreducible Bayes-floor    = {r.irreducible_rmse:.3f}  (reachable room {r.reachable_rmse_room:+.3f})")
    print(f"  R²_residual_boost          = {r.r2_residual_boost:+.4f}    (>0.05 = STRUCTURE)")
    print(f"  AUC_right_vs_wrong         = {r.auc_right_vs_wrong:.4f}    (>0.55 = STRUCTURE)")
    print(f"  ceiling proven             = {r.ceiling_proven}")
    print(f"\n  TOP-10 features by residual SHAP:")
    for i, (name, imp) in enumerate(r.top_features_residual[:10], 1):
        print(f"   {i:2d}.  {imp:.4f}   {name}")
    print(f"\n  >>> {r.verdict}")
    print(f"\n  full report: {OUT}/signal_hunt_report.md")
    print(f"  slice maps:  {OUT}/slice_map.png")

    exp.record(oof_score_mean=base_rmse, oof_score_per_fold=[base_rmse], holdout_score=base_rmse,
               runtime_sec=time.time() - t0, extra={
                   "r2_residual_boost": float(r.r2_residual_boost),
                   "auc_right_vs_wrong": float(r.auc_right_vs_wrong),
                   "irreducible_rmse": float(r.irreducible_rmse),
                   "reachable_rmse_room": float(r.reachable_rmse_room),
                   "ceiling_proven": bool(r.ceiling_proven),
                   "top5_residual_features": [n for n, _ in r.top_features_residual[:5]],
               })
    exp.note(r.verdict)
    exp.commit()
    print(f"=== {VER} done | {(time.time()-t0)/60:.1f} min total ===")


if __name__ == "__main__":
    main()
