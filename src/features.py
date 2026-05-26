"""Feature assembly for modeling.

Builds a per-row feature matrix from ONLY the columns present in both train and
test (`MD, X, Y, Z, GR, TVT_input` + the type-well) — never the train-only
geological markers or the `TVT` target (feature-availability leakage). Anchored on
the last-known TVT at the PS point; the model predicts TVT (or, in a residual
setup, the drift from the anchor) on the post-PS rows.

Feature groups:
  trajectory  — inclination, azimuth (sin/cos), vertical rate, dogleg  (from X,Y,Z)
  anchor      — last_known_tvt (per-well const), md_from_ps, z_from_ps
  position    — well_progress
  gr          — gr_filled + rolling/gradient/lag + deviation (features_sequence)
  alignment   — align_tvt, align_shift, align_cost (align.py)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import align, data, features_sequence as fs
from .config import DEPTH_COL, TARGET, TVT_INPUT_COL, WELL_ID

GR = "GR"


def _trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """Geometry of the well path from X,Y,Z vs MD. One well, index-aligned."""
    md = df[DEPTH_COL].to_numpy(dtype=float)
    x, y, z = (df[c].to_numpy(dtype=float) for c in ("X", "Y", "Z"))
    dmd = np.gradient(md)
    dmd[dmd == 0] = 1e-9
    dx, dy, dz = np.gradient(x), np.gradient(y), np.gradient(z)
    horiz = np.hypot(dx, dy)
    inclination = np.arctan2(horiz, np.abs(dz))          # 0 = vertical, π/2 = horizontal
    azimuth = np.arctan2(dy, dx)
    # dogleg: angular change of the unit step vector between consecutive samples
    v = np.column_stack([dx, dy, dz])
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm[norm == 0] = 1e-9
    u = v / norm
    cos_dl = np.clip(np.sum(u[1:] * u[:-1], axis=1), -1, 1)
    dogleg = np.concatenate([[0.0], np.arccos(cos_dl)])
    return pd.DataFrame({
        "traj_inclination": inclination,
        "traj_azimuth_sin": np.sin(azimuth),
        "traj_azimuth_cos": np.cos(azimuth),
        "traj_dz_dmd": dz / dmd,
        "traj_dogleg": dogleg,
    }, index=df.index)


def _anchor(df: pd.DataFrame) -> pd.DataFrame:
    """Last-known-TVT anchor + distance-from-PS features (train/test safe)."""
    ps = data.ps_index(df)
    md = df[DEPTH_COL].to_numpy(dtype=float)
    z = df["Z"].to_numpy(dtype=float)
    last_known_tvt = float(df[TVT_INPUT_COL].iloc[ps - 1]) if ps > 0 else float("nan")
    md_ps, z_ps = (md[ps - 1], z[ps - 1]) if ps > 0 else (md[0], z[0])
    return pd.DataFrame({
        "anchor_tvt": last_known_tvt,
        "md_from_ps": md - md_ps,
        "z_from_ps": z - z_ps,
    }, index=df.index)


def _poly_dip(df: pd.DataFrame, k: int = 100, deg: int = 1) -> pd.DataFrame:
    """Local geometric-dip signal from the known prefix (np.polyfit, analytical).

    Fits TVT-vs-MD on the last `k` known (pre-PS) samples and extrapolates the trend.
    Verified: this BEATS carry-forward only very near PS (first ~100 ft) and diverges
    far out — so it's a *feature*, not an anchor. The model combines `poly_drift` with
    `md_from_ps` to trust it near PS and discount it downhole.
      poly_slope : local dip (TVT per ft of MD), constant per well
      poly_drift : slope * (MD - MD_ps) — the dip-extrapolated drift from the anchor
    """
    ps = data.ps_index(df)
    md = df[DEPTH_COL].to_numpy(dtype=float)
    n = len(df)
    if ps >= 5:
        lo = max(0, ps - k)
        tvt_known = df[TVT_INPUT_COL].to_numpy(dtype=float)[lo:ps]   # = true TVT on prefix
        try:
            slope = float(np.polyfit(md[lo:ps], tvt_known, deg)[deg - 1] if deg == 1
                          else np.polyfit(md[lo:ps], tvt_known, 1)[0])
        except Exception:
            slope = 0.0
        md_ps = md[ps - 1]
    else:
        slope, md_ps = 0.0, md[0]
    return pd.DataFrame({"poly_slope": slope, "poly_drift": slope * (md - md_ps)}, index=df.index)


def build_well_features(horiz_df: pd.DataFrame, typewell_df: pd.DataFrame,
                        *, with_alignment: bool = True) -> pd.DataFrame:
    """Full per-row feature frame for ONE well, aligned to horiz_df.index."""
    df = horiz_df.copy()
    df[WELL_ID] = df.get(WELL_ID, "w")        # sequence helpers group by WELL_ID
    df[GR] = align.fill_gr(df[GR])            # GR ~29% NaN → fill before sequence FE

    parts = [_trajectory(horiz_df), _anchor(horiz_df), _poly_dip(horiz_df)]

    seq = fs.add_well_position(df)
    seq = fs.add_rolling(seq, [GR], windows=(5, 15, 31))
    seq = fs.add_gradient(seq, [GR])
    seq = fs.add_lags(seq, [GR], shifts=(1, 3, 5))
    seq = fs.add_deviation(seq, [GR])
    seq_cols = [c for c in seq.columns if c.startswith(("GR_", "well_progress"))]
    parts.append(seq.loc[horiz_df.index, seq_cols])
    parts.append(pd.DataFrame({"gr_filled": df[GR].to_numpy()}, index=horiz_df.index))

    if with_alignment:
        win = max(50, len(horiz_df) // 5)
        parts.append(align.alignment_features(horiz_df, typewell_df, window=win))

    return pd.concat(parts, axis=1)


def build_dataset(split: str = "train", well_ids: list[str] | None = None,
                  *, with_alignment: bool = True, post_ps_only: bool = True,
                  target: str = "residual"):
    """Assemble (X, y, groups, anchor) across wells.

    target:
      "residual" (default) — y = TVT - anchor_tvt (the DRIFT from the last-known TVT).
        Predicting absolute TVT fails: between-well level variance (9.2k–12.9k ft)
        dominates and trees can't extrapolate the level. Always model the drift, then
        de-residualize: TVT_pred = drift_pred + anchor. (strategy.md §3; verified.)
      "absolute" — y = TVT (only for diagnostics).

    Returns dict: X, y (np or None for test), groups (well-id/row), anchor (anchor_tvt/row),
    well_ids. `anchor` lets you recover TVT from a residual prediction:
        tvt_pred = model.predict(X) + anchor.
    When post_ps_only, only scored rows are kept (= the metric rows; for test = submission rows).
    """
    if target not in ("residual", "absolute"):
        raise ValueError("target must be 'residual' or 'absolute'")
    ids = well_ids if well_ids is not None else data.list_well_ids(split)
    X_parts, y_parts, grp_parts, anchor_parts = [], [], [], []
    for wid in ids:
        h = pd.read_csv(data.horizontal_path(wid, split))
        tw = pd.read_csv(data.typewell_path(wid, split))
        feats = build_well_features(h, tw, with_alignment=with_alignment)
        mask = data.post_ps_mask(h) if post_ps_only else np.ones(len(h), dtype=bool)
        X_parts.append(feats.loc[mask])
        grp_parts.append(np.full(int(mask.sum()), wid))
        anchor_parts.append(feats["anchor_tvt"].to_numpy()[mask])
        if split == "train":
            y_parts.append(h[TARGET].to_numpy()[mask])
    X = pd.concat(X_parts, ignore_index=True)
    groups = np.concatenate(grp_parts)
    anchor = np.concatenate(anchor_parts)
    if split == "train" and y_parts:
        tvt = np.concatenate(y_parts)
        y = (tvt - anchor) if target == "residual" else tvt
    else:
        y = None
    return {"X": X, "y": y, "groups": groups, "anchor": anchor, "well_ids": ids, "target": target}
