"""Known-prefix self-correlation — the accessible core of the winning recipe (P1).

Brief slide 9: the horizontal well's GR has higher resolution than the type-well's, and
the pre-PS portion gives THIS well's own GR↔TVT relationship. So to place a post-PS
sample in TVT, match its local GR signature against the well's own known-prefix GR
signatures — no cross-well GR-scale mismatch, no global monotonic-path constraint.

Implementation (mirrors the public 9.251 / target-free kernels): for each window scale,
build z-normalized GR windows over the prefix, index them with a cKDTree, and for every
post-PS sample query its GR window → nearest prefix window → that window's TVT. Multi-scale
(8/15/25) + an ensemble + a confidence (NN distance) + a per-well trust (more known data =
more trust). Pure nearest-neighbor (no model fit). GATE on the sacred wells.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial import cKDTree

from . import data
from .align import fill_gr
from .config import TVT_INPUT_COL


def _zwindows(gr: np.ndarray, w: int) -> np.ndarray:
    """(n-w+1, w) z-normalized GR windows; row p covers gr[p:p+w]."""
    W = sliding_window_view(gr, w).astype(float)
    mu = W.mean(1, keepdims=True)
    sd = W.std(1, keepdims=True)
    return (W - mu) / np.where(sd > 1e-9, sd, 1.0)


def selfcorr_features(horiz_df: pd.DataFrame, windows: tuple[int, ...] = (8, 15, 25),
                      stride: int = 2, gr_col: str = "GR", band: float = 80.0,
                      smooth: int = 51) -> pd.DataFrame:
    """Per-row self-correlation TVT estimates for one well, aligned to horiz_df.index.

    Templates are restricted to prefix windows whose TVT is within `band` ft of the
    last-known TVT (excludes the vertical build section, which spans the whole column),
    and the ensemble is median-smoothed along MD (the post-PS TVT is smooth). Estimates
    are clipped to anchor±band. These are GBDT FEATURES, not standalone predictions.
    """
    n = len(horiz_df)
    ps = data.ps_index(horiz_df)
    gr = fill_gr(horiz_df[gr_col])
    tvt_known = horiz_df[TVT_INPUT_COL].to_numpy(dtype=float)   # true TVT on prefix, NaN after PS
    idx = horiz_df.index
    anchor = float(tvt_known[ps - 1]) if ps > 0 else float(np.nanmean(tvt_known))

    cols: dict[str, np.ndarray] = {}
    ests = []
    for w in windows:
        sc = np.full(n, np.nan)
        conf = np.full(n, np.nan)
        if ps - w >= 3 and n - w + 1 > 0:
            Wz = _zwindows(gr, w)                                # (n-w+1, w)
            centers = np.arange(len(Wz)) + w // 2               # sample index per window
            tvt_at_all = tvt_known[np.arange(len(Wz)) + w // 2]
            # prefix windows near the anchor TVT (drop the build section)
            in_pre = (np.arange(len(Wz)) + w // 2) < ps
            near = in_pre & (np.abs(tvt_at_all - anchor) <= band)
            pre_pos = np.where(near)[0][::stride]
            if len(pre_pos) >= 3:
                tree = cKDTree(Wz[pre_pos])
                tvt_at = tvt_at_all[pre_pos]
                dist, nn = tree.query(Wz, k=1)
                est = np.clip(tvt_at[nn], anchor - band, anchor + band)
                sc[centers] = est
                conf[centers] = dist
        cols[f"selfcorr_tvt_w{w}"] = sc
        cols[f"selfcorr_conf_w{w}"] = conf
        ests.append(sc)

    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows → NaN (handled below)
            ens = np.nanmean(np.column_stack(ests), axis=1) if ests else np.full(n, np.nan)
    # median-smooth along MD (post-PS TVT is smooth); fall back to anchor where undefined
    ens_s = pd.Series(ens).rolling(smooth, center=True, min_periods=1).median().to_numpy()
    cols["selfcorr_tvt_ens"] = ens
    cols["selfcorr_tvt_smooth"] = np.where(np.isnan(ens_s), anchor, ens_s)
    cols["selfcorr_trust"] = float(np.clip(ps / 200.0, 0.0, 0.6))
    return pd.DataFrame(cols, index=idx)
