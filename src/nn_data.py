"""Phase-0 data pipeline for the sequence NN (S2).

Builds, per well, the raw GR signal + per-scored-point context, leaving the heavy
window extraction to the torch Dataset (slice-on-access) so we store each well's GR
**once** instead of materialising millions of overlapping windows.

Framing (see docs/nn_design.md): for each post-PS point we predict the **drift**
(TVT − last_known_tvt) — the same target the GBDT uses, so the OOF stacks cleanly.
GR is observed along the ENTIRE horizontal (only TVT is missing post-PS), so windows
can be centred with full context. GR is ~40% NaN → filled per well before use.

torch-free on purpose (pure numpy/pandas) so it unit-tests locally without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data
from .config import DEPTH_COL, TARGET, TVT_INPUT_COL

# context scalars per scored point (geometry + horizon — NOT the answer; all observed for test):
#   md_since_ps : steps drilled past the Prediction-Start anchor (drift grows with horizon)
#   dz_since_ps : Z change since the anchor (trajectory geometry, correlates with TVT drift)
#   incl_local  : local inclination proxy dZ/dMD
#   gr_at_point : GR right at the scored point
CTX_COLS = ["md_since_ps", "dz_since_ps", "incl_local", "gr_at_point"]


@dataclass
class WellSeq:
    """One well's filled GR track + the scalars needed to cut windows / build context."""
    gr: np.ndarray          # filled, GLOBALLY z-scored GR along MD (float32, full length)
    z: np.ndarray           # Z (depth) along MD
    md: np.ndarray          # MD along MD
    ps: int                 # Prediction-Start index
    last_known_tvt: float   # TVT_input[ps-1] — the drift anchor


@dataclass
class NNData:
    """Everything the torch Dataset needs for one split."""
    wells: dict[str, WellSeq]
    samples: list[tuple[str, int]]      # (well_id, iloc) for every scored point, in id order
    ids: list[str]                      # "{well}_{iloc}" — aligns to the GBDT cached frame
    y: np.ndarray                       # drift target (np.nan for test)
    ctx: np.ndarray                     # [N, len(CTX_COLS)] standardized context
    groups: np.ndarray                  # well id per sample (for GroupKFold)
    gr_stats: tuple[float, float]       # (mean, std) used to z-score GR — reuse across splits
    ctx_stats: tuple[np.ndarray, np.ndarray] = field(default=None)  # (mean, std) per ctx col


def _fill_gr(gr: np.ndarray) -> np.ndarray:
    """Linear-interpolate GR NaNs along MD, then edge-fill. (~40% NaN; GR is the signal.)"""
    s = pd.Series(gr).interpolate(method="linear", limit_direction="both")
    return s.fillna(0.0).to_numpy(np.float32)  # 0.0 only if a whole well is NaN (shouldn't happen)


def build(well_ids: list[str], split: str = "train", *,
          gr_stats: tuple[float, float] | None = None,
          ctx_stats: tuple[np.ndarray, np.ndarray] | None = None) -> NNData:
    """Build NN data for `well_ids`. Fit GR/ctx normalization on the first (dev) call and
    pass the returned stats to the sacred/test calls so all splits share one scale."""
    wells: dict[str, WellSeq] = {}
    samples: list[tuple[str, int]] = []
    ids: list[str] = []
    ys: list[float] = []
    ctx_rows: list[list[float]] = []
    groups: list[str] = []
    gr_accum: list[np.ndarray] = []  # to fit GR stats when not provided

    for wid in well_ids:
        h, _ = data.load_well(wid, split)
        h = h.sort_values(DEPTH_COL, kind="stable").reset_index(drop=True)
        ps = int(h[TVT_INPUT_COL].notna().sum())
        if ps < 1 or ps >= len(h):
            continue  # no anchor or no scored rows
        gr_filled = _fill_gr(h["GR"].to_numpy(np.float32))
        z = h["Z"].to_numpy(np.float32)
        md = h[DEPTH_COL].to_numpy(np.float32)
        last_known = float(h[TVT_INPUT_COL].iloc[ps - 1])
        gr_accum.append(gr_filled)
        wells[wid] = WellSeq(gr=gr_filled, z=z, md=md, ps=ps, last_known_tvt=last_known)

        tvt = h[TARGET].to_numpy(np.float32) if TARGET in h.columns else None  # train-only
        dmd_anchor = md - md[ps - 1]
        for i in range(ps, len(h)):
            samples.append((wid, i))
            ids.append(f"{wid}_{i}")
            groups.append(wid)
            ys.append(float(tvt[i] - last_known) if tvt is not None else np.nan)
            dz = float(z[i] - z[ps - 1])
            dmd = float(dmd_anchor[i]) or 1.0
            ctx_rows.append([float(i - ps), dz, dz / dmd, float(gr_filled[i])])

    ctx = np.asarray(ctx_rows, np.float32)
    # GR scale: fit on this split's GR if not given (dev), else reuse (sacred/test)
    if gr_stats is None:
        allgr = np.concatenate(gr_accum)
        gr_stats = (float(allgr.mean()), float(allgr.std() + 1e-6))
    gm, gs = gr_stats
    for w in wells.values():
        w.gr = (w.gr - gm) / gs
    # context scale
    if ctx_stats is None:
        ctx_stats = (ctx.mean(0), ctx.std(0) + 1e-6)
    cm, cs = ctx_stats
    ctx = (ctx - cm) / cs
    # gr_at_point lives in ctx too → re-scale it with GR stats for consistency (col index 3)
    return NNData(wells=wells, samples=samples, ids=ids, y=np.asarray(ys, np.float32),
                  ctx=ctx, groups=np.asarray(groups), gr_stats=gr_stats, ctx_stats=ctx_stats)
