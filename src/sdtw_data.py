"""Soft-DTW (C) — per-well episodes for the differentiable-DTW alignment model.

Each episode bundles what one well's soft-DTW forward needs: post-PS horizontal GR windows (queries),
a BAND of the typewell around the well's anchor (TVT-row nearest to last_known), and drift values for
that band. The band kills the GR-repeat ambiguity (drift bounded ±100 ft → ~few hundred typewell rows
suffice) and keeps the soft-DTW recursion tractable. Values are drift = TVT − last_known (frame-
independent, ensemble-compatible). torch-free so this unit-tests locally.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import data
from .config import DEPTH_COL, TARGET, TVT_INPUT_COL, TYPEWELL_DEPTH_COL
from .locator_data import _fill, _windows, _z   # reuse the GR-fill / z-score / sliding-window helpers


@dataclass
class SDTWEpisode:
    well: str
    q_gr: np.ndarray       # [N, L]   query GR windows (post-PS, per-well z-scored)
    q_y: np.ndarray        # [N]      true drift (TVT − last_known); np.nan for test
    q_ids: list[str]       # "{well}_{iloc}" — aligns to the kernel cache + submission
    tw_gr: np.ndarray      # [M, L]   typewell GR windows in the BAND around the anchor (z-scored)
    tw_drift: np.ndarray   # [M]      typewell TVT − last_known   (frame-independent)
    anchor_idx: int        # index within tw_gr/tw_drift of the anchor row (drift ≈ 0)


def build_episode(wid: str, split: str = "train", win: int = 32, band: int = 150) -> SDTWEpisode | None:
    h, tw = data.load_well(wid, split)
    h = h.sort_values(DEPTH_COL, kind="stable").reset_index(drop=True)
    ps = int(h[TVT_INPUT_COL].notna().sum())
    if ps < 2 or ps >= len(h):
        return None
    last_known = float(h[TVT_INPUT_COL].iloc[ps - 1])

    # horizontal post-PS query windows (per-well GR z-score)
    hg = _z(_fill(h["GR"].to_numpy(np.float32)))
    hw = _windows(hg, win)                                 # [n_h, L]
    q_gr = hw[ps:].astype(np.float32)
    q_y = (h[TARGET].to_numpy(np.float32)[ps:] - last_known) if TARGET in h.columns \
          else np.full(len(h) - ps, np.nan, np.float32)
    q_ids = [f"{wid}_{i}" for i in range(ps, len(h))]

    # typewell band around the anchor (TVT-row closest to last_known) — kills GR-repeat ambiguity
    tw = tw.sort_values(TYPEWELL_DEPTH_COL, kind="stable").reset_index(drop=True)
    tg = _z(_fill(tw["GR"].to_numpy(np.float32)))
    tww = _windows(tg, win)                                # [n_tw, L]
    tw_tvt = tw[TYPEWELL_DEPTH_COL].to_numpy(np.float64)
    anchor = int(np.abs(tw_tvt - last_known).argmin())
    lo, hi = max(0, anchor - band), min(len(tw_tvt), anchor + band + 1)
    tw_gr = tww[lo:hi].astype(np.float32)
    tw_drift = (tw_tvt[lo:hi] - last_known).astype(np.float32)

    return SDTWEpisode(well=wid, q_gr=q_gr, q_y=q_y, q_ids=q_ids,
                       tw_gr=tw_gr, tw_drift=tw_drift, anchor_idx=anchor - lo)


def build(well_ids: list[str], split: str = "train", win: int = 32, band: int = 150) -> list[SDTWEpisode]:
    eps = [build_episode(w, split, win, band) for w in well_ids]
    return [e for e in eps if e is not None]
