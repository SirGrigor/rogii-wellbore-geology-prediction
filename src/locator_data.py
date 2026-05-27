"""Phase-2 data pipeline for the LEARNED LOCATOR (supervised GR→typewell alignment).

Per well = one episode: a MEMORY of (GR-window → drift) slots from the typewell + the well's own pre-PS
section, and a set of post-PS QUERY GR-windows to locate. Values are drift = TVT − last_known_tvt
(frame-independent; typewell TVT shares the well's frame). GR is per-well z-scored (kills baseline
shifts → patterns comparable across query/memory). torch-free so it unit-tests locally; the torch model
(locator_model) consumes these episodes. See docs/locator_design.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data
from .config import DEPTH_COL, TARGET, TVT_INPUT_COL, TYPEWELL_DEPTH_COL


@dataclass
class Episode:
    well: str
    mem_gr: np.ndarray   # [n_mem, L] GR windows (typewell ∪ prefix), z-scored
    mem_val: np.ndarray  # [n_mem]   drift values  (TVT − last_known)
    q_gr: np.ndarray     # [n_q, L]  query GR windows (post-PS)
    q_y: np.ndarray      # [n_q]     target drift (np.nan for test)
    q_ids: list[str]     # "{well}_{iloc}"
    n_tw: int            # how many memory slots are typewell (rest are prefix)


def _fill(gr: np.ndarray) -> np.ndarray:
    s = pd.Series(gr).interpolate(method="linear", limit_direction="both")
    return s.fillna(0.0).to_numpy(np.float32)


def _z(x: np.ndarray) -> np.ndarray:
    m, s = float(np.nanmean(x)), float(np.nanstd(x))
    return ((x - m) / (s + 1e-6)).astype(np.float32)


def _windows(seq: np.ndarray, W: int) -> np.ndarray:
    """[n, 2W+1] edge-padded windows centred on each position (vectorised)."""
    L = 2 * W + 1
    if len(seq) == 0:
        return np.zeros((0, L), np.float32)
    padded = np.pad(seq, W, mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, L).astype(np.float32)


def build_episode(wid: str, split: str, W: int = 32) -> Episode | None:
    h, tw = data.load_well(wid, split)
    h = h.sort_values(DEPTH_COL, kind="stable").reset_index(drop=True)
    ps = int(h[TVT_INPUT_COL].notna().sum())
    if ps < 2 or ps >= len(h):
        return None
    last_known = float(h[TVT_INPUT_COL].iloc[ps - 1])

    hg = _z(_fill(h["GR"].to_numpy(np.float32)))            # per-well z-scored horizontal GR
    hw = _windows(hg, W)                                    # [n_h, L]

    tw = tw.sort_values(TYPEWELL_DEPTH_COL, kind="stable").reset_index(drop=True)
    tg = _z(_fill(tw["GR"].to_numpy(np.float32)))
    tw_w = _windows(tg, W)                                  # [n_tw, L]
    tw_v = (tw[TYPEWELL_DEPTH_COL].to_numpy(np.float32) - last_known)

    pre_w, pre_v = hw[:ps], (h[TVT_INPUT_COL].to_numpy(np.float32)[:ps] - last_known)   # prefix memory
    q_w = hw[ps:]                                            # query = post-PS windows
    q_y = (h[TARGET].to_numpy(np.float32)[ps:] - last_known) if TARGET in h.columns else np.full(len(h) - ps, np.nan, np.float32)
    q_ids = [f"{wid}_{i}" for i in range(ps, len(h))]

    return Episode(
        well=wid,
        mem_gr=np.concatenate([tw_w, pre_w], 0),
        mem_val=np.concatenate([tw_v, pre_v]).astype(np.float32),
        q_gr=q_w, q_y=q_y.astype(np.float32), q_ids=q_ids, n_tw=len(tw_w))


def build(well_ids: list[str], split: str = "train", W: int = 32) -> list[Episode]:
    eps = [build_episode(w, split, W) for w in well_ids]
    return [e for e in eps if e is not None]
