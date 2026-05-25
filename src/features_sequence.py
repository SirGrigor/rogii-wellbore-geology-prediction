"""Depth-sequence feature engineering, grouped per well.

The horizontal well is a depth-ordered series of log-curve readings. These are
the standard sequence features (rolling stats, gradients, lags/leads) — the depth
analogue of the lap-series FE from S6E5 — but computed strictly **within each
well** (grouped by WELL_ID, ordered by measured depth) so nothing leaks across
wells or across the fold boundary.

Curve columns are passed in (filled from config.CURVE_COLS once Phase-0 EDA
confirms the CSV headers), so this module is schema-agnostic and unit-testable.
"""
from __future__ import annotations

import pandas as pd

from .config import DEPTH_COL, WELL_ID


def _sorted_by_depth(df: pd.DataFrame, depth_col: str) -> pd.DataFrame:
    return df.sort_values([WELL_ID, depth_col], kind="stable")


def add_rolling(
    df: pd.DataFrame,
    curve_cols: list[str],
    windows: tuple[int, ...] = (5, 15, 31),
    depth_col: str = DEPTH_COL,
    stats: tuple[str, ...] = ("mean", "std"),
) -> pd.DataFrame:
    """Centered rolling stats per curve, per well. Adds `{col}_roll{w}_{stat}`."""
    out = _sorted_by_depth(df, depth_col).copy()
    g = out.groupby(WELL_ID, sort=False)
    for col in curve_cols:
        for w in windows:
            roll = g[col].rolling(window=w, center=True, min_periods=1)
            for stat in stats:
                out[f"{col}_roll{w}_{stat}"] = getattr(roll, stat)().reset_index(level=0, drop=True)
    return out


def add_gradient(
    df: pd.DataFrame,
    curve_cols: list[str],
    depth_col: str = DEPTH_COL,
) -> pd.DataFrame:
    """First difference of each curve w.r.t. measured depth, per well.

    Adds `{col}_grad` = Δcurve / Δdepth (rate of change along the well path).
    """
    out = _sorted_by_depth(df, depth_col).copy()
    g = out.groupby(WELL_ID, sort=False)
    d_depth = g[depth_col].diff()
    for col in curve_cols:
        d_curve = g[col].diff()
        out[f"{col}_grad"] = d_curve / d_depth.replace(0, pd.NA)
    return out


def add_lags(
    df: pd.DataFrame,
    curve_cols: list[str],
    shifts: tuple[int, ...] = (1, 3, 5),
    depth_col: str = DEPTH_COL,
    both_directions: bool = True,
) -> pd.DataFrame:
    """Lag (and optionally lead) features per curve, per well.

    Adds `{col}_lag{k}` and, if both_directions, `{col}_lead{k}`.
    """
    out = _sorted_by_depth(df, depth_col).copy()
    g = out.groupby(WELL_ID, sort=False)
    for col in curve_cols:
        for k in shifts:
            out[f"{col}_lag{k}"] = g[col].shift(k)
            if both_directions:
                out[f"{col}_lead{k}"] = g[col].shift(-k)
    return out


def add_well_position(df: pd.DataFrame, depth_col: str = DEPTH_COL) -> pd.DataFrame:
    """Normalized position along each well: 0.0 at the shallowest, 1.0 at the deepest.

    The depth analogue of S6E5's RaceProgress — often a strong contextual feature.
    """
    out = _sorted_by_depth(df, depth_col).copy()
    g = out.groupby(WELL_ID, sort=False)[depth_col]
    dmin, dmax = g.transform("min"), g.transform("max")
    span = (dmax - dmin).replace(0, pd.NA)
    out["well_progress"] = (out[depth_col] - dmin) / span
    return out
