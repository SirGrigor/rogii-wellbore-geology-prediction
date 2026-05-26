"""Geosteering decoder — recover post-PS TVT by tracking the GR signature (Viterbi).

The scored region is the lateral, where the well is ~horizontal: vertical geometry is
flat there and the drift is pure geological dip, recoverable only by matching the
horizontal GR to the type-well's GR(TVT). Naive global DTW fails because the path must
be monotonic — it can't represent the well climbing up *and* back down through dipping
layers. We instead model TVT as a hidden state tracked along measured depth:

    states     = the type-well TVT grid (the geological column)
    emission   = match between horizontal GR[i] and type-well GR at TVT=t (z-normalized)
    transition = smoothness — TVT changes gradually per MD step (bounded dip)
    pinning    = on the known pre-PS samples, force the state to the known TVT
                 (per-well calibration); post-PS is GR-driven, continuity carries across PS

Viterbi gives the most likely TVT path → the post-PS slice is the prediction, plus
match-confidence features for the residual GBDT. Pure dynamic programming (no model fit);
GATE its outputs on the sacred wells before trusting them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .align import fill_gr
from .config import CURVE_COLS, TYPEWELL_DEPTH_COL


def _smooth_z(a: np.ndarray, win: int) -> np.ndarray:
    s = pd.Series(np.asarray(a, dtype=float))
    if win > 1:
        s = s.rolling(win, center=True, min_periods=1).mean()
    sd = s.std()
    return ((s - s.mean()) / sd).to_numpy() if sd > 0 else (s - s.mean()).to_numpy()


def decode_tvt(
    h_gr: np.ndarray,
    known_tvt: np.ndarray,          # full-length; true TVT on the known prefix, NaN after PS
    tw_gr: np.ndarray,
    tw_tvt: np.ndarray,             # monotonic type-well TVT grid
    *,
    n_states: int = 400,
    band: int = 40,                 # max state move per MD step (bounds dip rate)
    smooth_win: int = 7,
    lam: float = 0.05,              # transition smoothness weight (per state-step^2)
    pin: float = 50.0,              # cost pulling known-prefix samples to their true TVT
) -> dict[str, np.ndarray]:
    """Viterbi TVT path for one well. Returns tvt (full-length path) + match features."""
    h = _smooth_z(fill_gr(h_gr), smooth_win)
    # subsample the type-well to n_states evenly across TVT
    idx = np.linspace(0, len(tw_tvt) - 1, min(n_states, len(tw_tvt))).round().astype(int)
    tvt_s = np.asarray(tw_tvt, float)[idx]
    g_s = _smooth_z(np.asarray(tw_gr, float)[idx], smooth_win)
    M, n = len(tvt_s), len(h)
    known = np.asarray(known_tvt, float)

    # emission e[i, j]: GR mismatch; on known samples, force the nearest state to the true TVT
    emit = (h[:, None] - g_s[None, :]) ** 2                      # (n, M)
    known_mask = ~np.isnan(known)
    if known_mask.any():
        ki = np.where(known_mask)[0]
        nearest = np.abs(known[ki, None] - tvt_s[None, :]).argmin(1)
        emit[ki] = pin * (np.arange(M)[None, :] != nearest[:, None])  # 0 at true state, pin elsewhere

    offsets = np.arange(-band, band + 1)
    pen = lam * offsets.astype(float) ** 2
    dp = emit[0].copy()
    back = np.zeros((n, M), dtype=np.int32)
    for i in range(1, n):
        stack = np.full((len(offsets), M), np.inf)
        for k, d in enumerate(offsets):                          # stack[k,j] = dp_prev[j-d] + pen[k]
            if d >= 0:
                stack[k, d:] = dp[: M - d] + pen[k]
            elif d < 0:
                stack[k, :M + d] = dp[-d:] + pen[k]
        best = stack.min(0)
        arg = stack.argmin(0)
        back[i] = np.arange(M) - offsets[arg]                    # chosen previous state j'
        dp = emit[i] + best

    # backtrace
    path = np.empty(n, dtype=np.int32)
    path[-1] = int(dp.argmin())
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]

    tvt_path = tvt_s[path]
    em_cost = emit[np.arange(n), path]                            # per-sample match cost (high = uncertain)
    return {"tvt": tvt_path, "match_cost": em_cost,
            "state_step": np.abs(np.diff(path, prepend=path[0])).astype(float)}


def decode_features(horiz_df: pd.DataFrame, typewell_df: pd.DataFrame, *,
                    gr_col: str = CURVE_COLS[0], tw_gr_col: str = CURVE_COLS[0],
                    tw_tvt_col: str = TYPEWELL_DEPTH_COL,
                    known_tvt_col: str = "TVT_input", **kw) -> pd.DataFrame:
    """Per-row geosteering features for one well, aligned to horiz_df.index.

    geo_tvt (decoded TVT — the prediction), geo_match_cost (GR-match uncertainty),
    geo_state_step (local TVT movement). Uses TVT_input as the known pre-PS prefix.
    """
    out = decode_tvt(horiz_df[gr_col].to_numpy(),
                     horiz_df[known_tvt_col].to_numpy(),
                     typewell_df[tw_gr_col].to_numpy(),
                     typewell_df[tw_tvt_col].to_numpy(), **kw)
    return pd.DataFrame({"geo_tvt": out["tvt"], "geo_match_cost": out["match_cost"],
                         "geo_state_step": out["state_step"]}, index=horiz_df.index)
