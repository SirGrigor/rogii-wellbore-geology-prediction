"""Strategy-2 extra features for the GBDT — cheap, per-well, leakage-free.

Computed straight from the raw horizontal (fast: well loop, no 86-min kernel rebuild) and
merged onto the cached 222 by `id`. Wave-prefixed so the ablation in 09_features.py can add
them sequentially and measure each wave's marginal Δ on sacred (drop a wave if Δ ≥ 0).

  w1_* — horizon + geometry: distance/▲Z/▲MD since the PS anchor, inclination. Strong priors
         the alignment features don't encode (error grows with horizon; ▲Z is geometric drift).
  w2_* — GR texture: gradient/curvature/local roughness/level — GR *shape* beyond the DTW summaries.

All inputs are observed for test too (GR + trajectory along the whole well); no target leakage.
torch-free, local-testable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data
from .config import DEPTH_COL, TVT_INPUT_COL
from .nn_data import _fill_gr

W1 = ["w1_md_since_ps", "w1_dz_since_ps", "w1_dmd_since_ps", "w1_incl_cum", "w1_incl_local"]
W2 = ["w2_gr_at", "w2_gr_grad", "w2_gr_curv", "w2_gr_std15", "w2_gr_std51", "w2_gr_mean51"]


def build_extra(well_ids: list[str], split: str = "train") -> pd.DataFrame:
    """Per-scored-point extra features, indexed by `id` = '{well}_{iloc}' (matches the kernel cache)."""
    idx: list[str] = []
    recs: list[list[float]] = []
    for wid in well_ids:
        h, _ = data.load_well(wid, split)
        h = h.sort_values(DEPTH_COL, kind="stable").reset_index(drop=True)
        ps = int(h[TVT_INPUT_COL].notna().sum())
        if ps < 1 or ps >= len(h):
            continue
        gr = _fill_gr(h["GR"].to_numpy(np.float32))
        z = h["Z"].to_numpy(np.float64)
        md = h[DEPTH_COL].to_numpy(np.float64)
        grad = np.gradient(gr, md)                       # dGR/dMD
        curv = np.gradient(grad, md)                     # d²GR/dMD²
        s = pd.Series(gr)
        std15 = s.rolling(15, center=True, min_periods=1).std().to_numpy()
        std51 = s.rolling(51, center=True, min_periods=1).std().to_numpy()
        mean51 = s.rolling(51, center=True, min_periods=1).mean().to_numpy()
        z0, md0 = z[ps - 1], md[ps - 1]
        for i in range(ps, len(h)):
            dz = z[i] - z0
            dmd = (md[i] - md0) or 1.0
            j = max(ps, i - 10)                           # local window for local inclination
            dmd_loc = (md[i] - md[j]) or 1.0
            idx.append(f"{wid}_{i}")
            recs.append([
                float(i - ps), float(dz), float(md[i] - md0), float(dz / dmd), float((z[i] - z[j]) / dmd_loc),
                float(gr[i]), float(grad[i]), float(curv[i]), float(std15[i]), float(std51[i]), float(mean51[i]),
            ])
    return pd.DataFrame(recs, index=idx, columns=W1 + W2).astype(np.float32)
