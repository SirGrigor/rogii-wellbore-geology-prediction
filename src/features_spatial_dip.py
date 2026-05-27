"""Cross-well spatial dip field — the signal the residual diagnosis (v11) pointed to.

Diagnosis: sacred error is dominated by HORIZON (octile RMSE 3.7→12.0, spread +8.3) and LARGE DRIFT
(5.4→16.2, +10.8); and near-neighbour wells err far less (mean|resid| 5.54 vs 7.50). Reading: far from
PS the well's own GR↔typewell alignment degrades, but neighbour wells that physically drilled the same
(X,Y) carry the local geological DIP. The per-well 222 align each well to its OWN typewell — blind to
neighbours. This computes the cross-well dip.

FRAME-INDEPENDENT (dodges the falsified absolute-TVT M1, where typewell frames differ ~2000 ft):
fit a local weighted plane  TVT ≈ a·X + b·Y + c  to neighbour points and use ONLY the gradient (a,b)
— the dip — to extrapolate DRIFT from the well's own last-known point:
    dip_drift ≈ a·(X − X_ps) + b·(Y − Y_ps)
The frame offset c cancels. Leakage-safe: neighbour points come from the fit-set; the target's own
well is always excluded. Features are merged onto the cached 222 by id (no kernel rebuild).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import data
from .config import DEPTH_COL, TARGET, TVT_INPUT_COL

COLS = ["sd_dip_drift", "sd_dip_mag", "sd_nb_tvt_std", "sd_nb_dist"]


def build_cloud(fit_well_ids, split: str = "train", stride: int = 4):
    """Point cloud (x, y, tvt, well_index) from fit-set wells (need TVT → train split)."""
    xs, ys, tv, wi = [], [], [], []
    for k, w in enumerate(fit_well_ids):
        h, _ = data.load_well(w, split)
        h = h.sort_values(DEPTH_COL, kind="stable")
        if TARGET not in h.columns:
            continue
        x = h["X"].to_numpy(np.float64)[::stride]; y = h["Y"].to_numpy(np.float64)[::stride]
        t = h[TARGET].to_numpy(np.float64)[::stride]
        m = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
        xs.append(x[m]); ys.append(y[m]); tv.append(t[m]); wi.append(np.full(int(m.sum()), k))
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(tv), np.concatenate(wi),
            {w: k for k, w in enumerate(fit_well_ids)})


def dip_features(target_well_ids, split, cloud, k: int = 32) -> pd.DataFrame:
    """Per scored point: local weighted-plane dip from k nearest neighbour points (own well excluded
    via weight-masking) → frame-independent dip-extrapolated drift. Vectorised per well (batched 3×3
    solves) so it scales to millions of rows. Indexed by id '{well}_{iloc}'. S = coord normaliser."""
    cx, cy, ct, cw, widx = cloud
    tree = cKDTree(np.column_stack([cx, cy]))
    S = 1000.0
    idx, blocks = [], []
    for w in target_well_ids:
        h, _ = data.load_well(w, split)
        h = h.sort_values(DEPTH_COL, kind="stable").reset_index(drop=True)
        ps = int(h[TVT_INPUT_COL].notna().sum())
        if ps < 1 or ps >= len(h):
            continue
        X = h["X"].to_numpy(np.float64); Y = h["Y"].to_numpy(np.float64)
        x0, y0 = X[ps - 1], Y[ps - 1]
        sc = np.arange(ps, len(h)); qx, qy = X[sc], Y[sc]
        dist, nb = tree.query(np.column_stack([qx, qy]), k=k, workers=-1)          # (n,k)
        nbt = ct[nb]                                                                # neighbour TVT
        wgt = 1.0 / (dist + 1.0)
        wgt[cw[nb] == widx.get(w, -1)] = 0.0                                        # exclude own well
        ws = wgt.sum(1, keepdims=True); ws[ws < 1e-9] = 1.0; wgt = wgt / ws
        dx = (cx[nb] - qx[:, None]) / S; dy = (cy[nb] - qy[:, None]) / S            # centred + normalised
        A = np.stack([dx, dy, np.ones_like(dx)], 2) * wgt[:, :, None]               # (n,k,3)
        AtA = np.einsum("nki,nkj->nij", A, A) + 0.3 * np.eye(3)                     # ridge → shrink collinear dir
        Atb = np.einsum("nki,nk->ni", A, nbt * wgt)
        coef = np.linalg.solve(AtA, Atb[:, :, None])[:, :, 0]                       # (n,3) batched solve
        a = np.clip(coef[:, 0] / S, -0.08, 0.08); b = np.clip(coef[:, 1] / S, -0.08, 0.08)
        dd = np.clip(a * (X[sc] - x0) + b * (Y[sc] - y0), -150.0, 150.0)            # frame-indep (c cancels)
        blocks.append(np.column_stack([dd, np.hypot(a, b), nbt.std(1), dist.mean(1)]).astype(np.float32))
        idx.extend(f"{w}_{i}" for i in sc)
    return pd.DataFrame(np.vstack(blocks), index=idx, columns=COLS)
