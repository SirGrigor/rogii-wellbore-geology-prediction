"""Tier-0 EDA — per-well structure, PS detection, divergence, GR NaN, spatial, floor.

Run from repo root:  uv run python notebooks/01_eda.py
Prints a summary; numbers are transcribed into docs/eda.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import data
from src.config import TARGET, TVT_INPUT_COL


def ps_index(df: pd.DataFrame) -> int:
    """PS point = first post-PS row = number of known (non-NaN) TVT_input values.

    TVT_input is the true TVT on the known prefix and NaN after PS (verified).
    """
    return int(df[TVT_INPUT_COL].notna().sum())


def main() -> None:
    ids = data.list_well_ids("train")
    print(f"train wells: {len(ids)}")

    rows = []
    floor_sq_err = []   # squared error of carry-forward floor on post-PS rows (overall RMSE)
    n_scored = 0
    for wid in ids:
        df = pd.read_csv(data.horizontal_path(wid, "train"))
        ps = ps_index(df)
        n = len(df)
        gr_nan = int(df["GR"].isna().sum())
        post = df.iloc[ps:]
        n_post = len(post)
        if n_post and ps > 0:
            last_known_tvt = float(df[TARGET].iloc[ps - 1])  # = TVT_input[ps-1]
            err = (post[TARGET].to_numpy() - last_known_tvt)   # carry-forward floor
            floor_sq_err.append(err ** 2)
            n_scored += n_post
            max_div = float(np.abs(err).max())                 # how far TVT drifts post-PS
            well_floor_rmse = float(np.sqrt((err ** 2).mean()))
        else:
            max_div, well_floor_rmse = 0.0, 0.0
        rows.append(dict(
            well=wid, n=n, ps=ps, n_post=n_post,
            post_frac=round(n_post / n, 3) if n else 0,
            gr_nan=gr_nan, gr_nan_frac=round(gr_nan / n, 4) if n else 0,
            tvt_min=float(df[TARGET].min()), tvt_max=float(df[TARGET].max()),
            x=float(df["X"].mean()), y=float(df["Y"].mean()),
            max_div_post=round(max_div, 2), floor_rmse=round(well_floor_rmse, 3),
        ))

    s = pd.DataFrame(rows)
    overall_floor_rmse = float(np.sqrt(np.concatenate(floor_sq_err).mean()))

    print("\n=== per-well summary (describe) ===")
    print(s[["n", "ps", "n_post", "post_frac", "gr_nan_frac", "max_div_post", "floor_rmse"]].describe().round(3).to_string())
    print(f"\nwells with NO divergence (ps=-1): {(s.ps < 0).sum()}")
    print(f"total post-PS (scored-equivalent) train rows: {n_scored:,}")
    print(f"TVT range across all wells: [{s.tvt_min.min():.1f}, {s.tvt_max.max():.1f}] ft")
    print(f"GR NaN: wells with any NaN = {(s.gr_nan > 0).sum()}/{len(s)}; "
          f"mean NaN frac = {s.gr_nan_frac.mean():.4f}")
    print(f"\n*** FLOOR BASELINE (carry-forward last known TVT) — overall post-PS RMSE = {overall_floor_rmse:.4f} ft ***")
    print(f"    per-well floor RMSE: median {s.floor_rmse.median():.3f}, "
          f"mean {s.floor_rmse.mean():.3f}, p90 {s.floor_rmse.quantile(0.9):.3f}, max {s.floor_rmse.max():.3f}")
    print(f"    max post-PS divergence (|TVT-TVT_input|): median {s.max_div_post.median():.2f}, "
          f"max {s.max_div_post.max():.2f} ft")

    # spatial spread
    print(f"\nspatial: X [{s.x.min():.0f}, {s.x.max():.0f}], Y [{s.y.min():.0f}, {s.y.max():.0f}]")

    # test wells
    test_ids = data.list_well_ids("test")
    print(f"\ntest wells: {len(test_ids)} -> {test_ids}")
    ss = data.load_sample_submission()
    print(f"sample_submission rows: {len(ss):,}  (= post-PS rows to predict across test wells)")

    s.to_csv("reports/eda_per_well.csv", index=False)
    print("\nsaved reports/eda_per_well.csv")


if __name__ == "__main__":
    main()
