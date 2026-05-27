"""Prediction post-processing (S3a) — per-well smoothing + clipping of the drift curve.

TVT along a wellbore is a smooth geological curve, but the GBDT predicts each row
independently → high-frequency jitter. A Savitzky-Golay filter on the predicted drift
*per well* (ordered by iloc) removes that jitter; clipping bounds rare outliers. The
top-3 public solution uses exactly this. The smoothing WINDOW is tuned on dev-OOF (never
on sacred) — then applied unchanged to sacred/test (leak-free, L48/L53).

Operates on predictions keyed by `id = '{well}_{iloc}'`, so it needs no raw data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .evaluate import rmse


def _well_iloc(ids):
    well = np.array([i.rsplit("_", 1)[0] for i in ids])
    iloc = np.array([int(i.rsplit("_", 1)[1]) for i in ids])
    return well, iloc


def smooth_per_well(ids, pred, window: int = 21, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay smooth `pred` within each well (ordered by iloc). Short wells / windows
    are handled (window clamped to an odd value ≤ well length; skipped if too short)."""
    pred = np.asarray(pred, float)
    out = pred.copy()
    well, iloc = _well_iloc(ids)
    order = np.arange(len(ids))
    df = pd.DataFrame({"well": well, "iloc": iloc, "order": order})
    for _, g in df.groupby("well", sort=False):
        g = g.sort_values("iloc")
        idx = g["order"].to_numpy()
        n = len(idx)
        wl = min(window, n)
        if wl % 2 == 0:
            wl -= 1
        if wl < poly + 2:                      # too short to smooth meaningfully
            continue
        out[idx] = savgol_filter(pred[idx], wl, poly)
    return out


def tune_window(ids, oof_pred, y, windows=(7, 11, 15, 21, 31, 41, 61), poly: int = 2):
    """Pick the smoothing window that minimises OOF RMSE (leak-free). Returns (best_w, rmse_table)."""
    base = rmse(y, oof_pred)
    table = {0: base}
    best_w, best_r = 0, base
    for w in windows:
        r = rmse(y, smooth_per_well(ids, oof_pred, w, poly))
        table[w] = r
        if r < best_r:
            best_r, best_w = r, w
    return best_w, table


def fit_clip(train_drift, q: float = 0.001) -> tuple[float, float]:
    """Clip bounds from the train drift's [q, 1-q] quantiles (rare-outlier guard)."""
    lo, hi = np.quantile(np.asarray(train_drift, float), [q, 1 - q])
    return float(lo), float(hi)
