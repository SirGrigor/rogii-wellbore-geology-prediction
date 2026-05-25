"""Type-well ↔ horizontal-well GR alignment (DTW) — the core geosteering signal.

A vertical *type-well* gives GR as a function of TVT (monotonic in TVT). The
horizontal well traverses that geological column; its GR signature, matched to the
type-well's, tells you the TVT at each measured-depth sample (brief slides 6–7).
Dynamic Time Warping over the GR curves yields, per horizontal sample, the matched
type-well TVT — a direct TVT estimate and a strong model feature.

This module is the production tooling: NaN-robust GR (GR is ~29% NaN), a fast
windowed DTW (dtaidistance C backend with a pure-numpy fallback), and well-level
helpers that emit per-row features. The *best* alignment recipe (handling the well
climbing back up, multi-resolution, prefix-calibration) is a modeling experiment;
this gives a sane, fast default. **GATE alignment features** (synth_decoder.gate)
before trusting them — does the matched TVT beat the carry-forward floor on the
rows it affects?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CURVE_COLS, TYPEWELL_DEPTH_COL


def fill_gr(gr: np.ndarray | pd.Series) -> np.ndarray:
    """Linear-interpolate interior GR NaNs; edge-fill the ends. Always returns finite."""
    s = pd.Series(np.asarray(gr, dtype=float))
    s = s.interpolate(method="linear", limit_direction="both")
    if s.isna().any():               # all-NaN well (degenerate) → zeros
        s = s.fillna(0.0)
    return s.to_numpy()


def _zscore(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    sd = a.std()
    return (a - a.mean()) / sd if sd > 0 else a - a.mean()


def warping_path(a: np.ndarray, b: np.ndarray, *, window: int | None = None,
                 normalize: bool = True) -> list[tuple[int, int]]:
    """DTW warping path between two 1-D curves (a → horizontal, b → type-well).

    Uses dtaidistance's C backend when available (fast); falls back to a compact
    pure-numpy DTW. `window` is a Sakoe-Chiba band (in samples) limiting warping.
    """
    a = _zscore(a) if normalize else np.asarray(a, float)
    b = _zscore(b) if normalize else np.asarray(b, float)
    try:
        from dtaidistance import dtw

        kw = {"use_c": True}
        if window:
            kw["window"] = int(window)
        return dtw.warping_path(a, b, **kw)
    except Exception:
        return _dtw_path_numpy(a, b, window=window)


def _dtw_path_numpy(a: np.ndarray, b: np.ndarray, window: int | None = None) -> list[tuple[int, int]]:
    n, m = len(a), len(b)
    w = max(window or max(n, m), abs(n - m))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        jlo, jhi = max(1, i - w), min(m, i + w)
        for j in range(jlo, jhi + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        i, j = (i - 1, j - 1) if step == 0 else ((i - 1, j) if step == 1 else (i, j - 1))
    return path[::-1]


def align_curves(h_gr: np.ndarray, tw_gr: np.ndarray, tw_tvt: np.ndarray,
                 *, window: int | None = None) -> dict[str, np.ndarray | float]:
    """Align one horizontal GR curve to a type-well (GR, TVT). Per-horizontal-row outputs.

    Returns:
        matched_tvt  : type-well TVT at the matched index (the TVT estimate)
        local_shift  : matched index minus a straight-line expectation (climb/fall signal)
        dtw_cost     : scalar mean |local_shift| (lower = cleaner match)
    """
    h = fill_gr(h_gr)
    tw = fill_gr(tw_gr)
    tw_tvt = np.asarray(tw_tvt, dtype=float)
    n = len(h)
    path = warping_path(h, tw, window=window)
    matched_idx = np.full(n, -1, dtype=float)
    for i, j in path:
        matched_idx[i] = j               # monotonic path → last write per i is fine
    # forward/back-fill any horizontal index the path skipped
    matched_idx = pd.Series(matched_idx).replace(-1, np.nan).interpolate(
        limit_direction="both").fillna(0).to_numpy()
    idx_int = np.clip(matched_idx.round().astype(int), 0, len(tw_tvt) - 1)
    matched_tvt = tw_tvt[idx_int]
    expected = np.linspace(0, len(tw) - 1, n)
    local_shift = matched_idx - expected
    return {
        "matched_tvt": matched_tvt,
        "local_shift": local_shift,
        "dtw_cost": float(np.mean(np.abs(local_shift))),
    }


def alignment_features(horiz_df: pd.DataFrame, typewell_df: pd.DataFrame,
                       *, window: int | None = None,
                       gr_col: str = CURVE_COLS[0],
                       tw_gr_col: str = CURVE_COLS[0],
                       tw_tvt_col: str = TYPEWELL_DEPTH_COL) -> pd.DataFrame:
    """Per-row alignment features for one well, aligned index to `horiz_df`.

    Columns: align_tvt, align_shift, align_cost (cost is constant per well).
    """
    out = align_curves(
        horiz_df[gr_col].to_numpy(),
        typewell_df[tw_gr_col].to_numpy(),
        typewell_df[tw_tvt_col].to_numpy(),
        window=window,
    )
    return pd.DataFrame({
        "align_tvt": out["matched_tvt"],
        "align_shift": out["local_shift"],
        "align_cost": out["dtw_cost"],
    }, index=horiz_df.index)


def predict_tvt(horiz_df: pd.DataFrame, typewell_df: pd.DataFrame, **kw) -> np.ndarray:
    """Convenience: the alignment's TVT estimate per horizontal row (= align_tvt)."""
    return alignment_features(horiz_df, typewell_df, **kw)["align_tvt"].to_numpy()
