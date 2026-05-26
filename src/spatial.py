"""Nearby-well spatial dip — the cross-well generalizing signal (M1).

Geology is spatially coherent: TVT (stratigraphic position) relates to the wellbore's
vertical coordinate Z through a geological datum surface that varies smoothly in map
view, **surf = Z - TVT**, and that surface is SHARED across wells (the field dips
coherently — brief slides 12-13). So a sample's TVT can be recovered from neighboring
wells' known geology: interpolate surf at its (X,Y) from nearby known samples, then
`TVT_pred = Z - surf(X,Y)`. Unlike per-well GR noise, this is cross-well structure that
should survive the sacred holdout.

Leakage rule: a well's own POST-PS TVT is the answer — never in the neighbor pool. The
pool = all OTHER wells' full TVT + the well's OWN pre-PS (known) samples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import data
from .config import TARGET, TVT_INPUT_COL


def build_surf_pool(well_ids: list[str], split: str = "train", subsample: int = 5):
    """Gather (X, Y, surf=Z-TVT, well) for KNOWN samples across wells (subsampled).

    Train: full TVT known. (For test pool we'd only have pre-PS; handled by the caller.)
    Returns dict with arrays xy (N,2), surf (N,), well (N,).
    """
    xs, ys, surf, wells = [], [], [], []
    for wid in well_ids:
        df = pd.read_csv(data.horizontal_path(wid, split))
        tvt = df[TARGET].to_numpy(float) if TARGET in df.columns else df[TVT_INPUT_COL].to_numpy(float)
        m = np.isfinite(tvt)
        x, y, z = df["X"].to_numpy(float), df["Y"].to_numpy(float), df["Z"].to_numpy(float)
        idx = np.where(m)[0][::subsample]
        xs.append(x[idx]); ys.append(y[idx]); surf.append(z[idx] - tvt[idx])
        wells.append(np.full(len(idx), wid))
    return {"xy": np.column_stack([np.concatenate(xs), np.concatenate(ys)]),
            "surf": np.concatenate(surf), "well": np.concatenate(wells)}


def spatial_tvt(horiz_df: pd.DataFrame, pool: dict, *, exclude_well: str | None = None,
                k: int = 24, eps: float = 1e-6) -> pd.DataFrame:
    """Per-row spatial TVT estimate for one well via inverse-distance surf interpolation.

    Excludes `exclude_well` from the neighbor pool (leakage-safe). Returns
    spatial_tvt (Z - interpolated surf), spatial_nn_dist (confidence), spatial_surf.
    """
    mask = pool["well"] != exclude_well if exclude_well is not None else np.ones(len(pool["surf"]), bool)
    xy, surf = pool["xy"][mask], pool["surf"][mask]
    tree = cKDTree(xy)
    q = np.column_stack([horiz_df["X"].to_numpy(float), horiz_df["Y"].to_numpy(float)])
    dist, nn = tree.query(q, k=k)
    w = 1.0 / (dist + eps)
    surf_hat = (surf[nn] * w).sum(1) / w.sum(1)            # inverse-distance weighted surf
    z = horiz_df["Z"].to_numpy(float)
    return pd.DataFrame({
        "spatial_tvt": z - surf_hat,
        "spatial_surf": surf_hat,
        "spatial_nn_dist": dist.mean(1),                    # mean neighbor distance (confidence)
        "spatial_nn_spread": surf[nn].std(1),               # neighbor surf disagreement
    }, index=horiz_df.index)
