"""Type-well ↔ horizontal-well curve alignment (DTW).

This is the genuinely NEW skill for this competition (no prior art in the S6E5
lap-series toolkit). A vertical *type-well* gives the expected log-curve response
through the geological column; the horizontal well traverses that column at a
shallow angle. Aligning the horizontal curve to the type-well (Dynamic Time
Warping over depth) yields, per horizontal sample, an estimate of *where in the
type-well column it currently sits* — which is the structural signal TVT
prediction needs.

Skeleton: the alignment math is implemented; the exact curve to align on
(gamma-ray is the usual geosteering choice) is wired once Phase-0 confirms the
column names. Gate any alignment-derived feature with synth_decoder.gate before
trusting it (does it actually beat the model on the rows it affects?).
"""
from __future__ import annotations

import numpy as np


def _zscore(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    sd = a.std()
    return (a - a.mean()) / sd if sd > 0 else a - a.mean()


def warping_path(
    horizontal_curve: np.ndarray,
    typewell_curve: np.ndarray,
    *,
    normalize: bool = True,
    window: int | None = None,
):
    """DTW warping path between a horizontal curve and a type-well curve.

    Returns the list of (i, j) index pairs (i → horizontal sample, j → type-well
    sample). Uses dtaidistance if available; falls back to a compact pure-numpy
    DTW so the module imports even before the optional dep is installed.
    """
    a = _zscore(horizontal_curve) if normalize else np.asarray(horizontal_curve, float)
    b = _zscore(typewell_curve) if normalize else np.asarray(typewell_curve, float)
    try:
        from dtaidistance import dtw

        _, paths = dtw.warping_paths(a, b, window=window, use_c=False)
        return dtw.best_path(paths)
    except Exception:
        return _dtw_path_numpy(a, b)


def _dtw_path_numpy(a: np.ndarray, b: np.ndarray) -> list[tuple[int, int]]:
    """Minimal O(n*m) DTW with path backtrace (fallback only)."""
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return path[::-1]


def alignment_features(
    horizontal_curve: np.ndarray,
    typewell_curve: np.ndarray,
    typewell_depth: np.ndarray | None = None,
    **kw,
) -> dict[str, np.ndarray | float]:
    """Per-horizontal-sample alignment features derived from the DTW path.

    Returns:
        matched_tw_index : for each horizontal sample, the aligned type-well row.
        matched_tw_depth  : the type-well depth at that match (if depths provided)
                            — an estimate of structural position in the column.
        local_shift       : matched_tw_index minus a straight-line expectation
                            (how much the well is climbing/falling vs. type-well).
        dtw_cost          : scalar total alignment cost (lower = better match).

    Feed `matched_tw_depth` / `local_shift` into the model as features. GATE them.
    """
    path = warping_path(horizontal_curve, typewell_curve, **kw)
    n = len(horizontal_curve)
    matched = np.full(n, np.nan)
    for i, j in path:
        # last match wins per horizontal index (path is monotonic)
        matched[i] = j
    # straight-line expectation: type-well index proportional to position in well
    expected = np.linspace(0, len(typewell_curve) - 1, n)
    local_shift = matched - expected
    out: dict[str, np.ndarray | float] = {
        "matched_tw_index": matched,
        "local_shift": local_shift,
        "dtw_cost": float(np.nanmean(np.abs(local_shift))),
    }
    if typewell_depth is not None:
        td = np.asarray(typewell_depth, float)
        idx = np.clip(np.nan_to_num(matched, nan=0).astype(int), 0, len(td) - 1)
        out["matched_tw_depth"] = td[idx]
    return out
