"""Data access for the per-well CSV-pair layout.

Confirmed file naming (from `kaggle competitions files`):
    {split}/{well_id}__horizontal_well.csv
    {split}/{well_id}__typewell.csv
    train/{well_id}.png                     (cross-section viz — reference, not target)

A "well_id" is the 8-hex prefix of the filename (e.g. ``000d7d20``). Column names
inside the CSVs are confirmed in Phase-0 EDA; this module is schema-agnostic and
just gives you tidy DataFrames keyed by well.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import numpy as np

from .config import TEST_DIR, TRAIN_DIR, TVT_INPUT_COL, WELL_ID

_HORIZ_SUFFIX = "__horizontal_well.csv"
_TYPEWELL_SUFFIX = "__typewell.csv"


def ps_index(df: pd.DataFrame) -> int:
    """Prediction-Start row = first scored (post-PS) row = number of known TVT_input values.

    TVT_input holds the true TVT on the known prefix and is NaN afterwards (verified EDA),
    and the known part is a contiguous prefix, so the count of non-NaN == the first NaN index.
    """
    return int(df[TVT_INPUT_COL].notna().sum())


def post_ps_mask(df: pd.DataFrame) -> np.ndarray:
    """Boolean mask of the scored (post-PS) rows — the rows the metric is computed on."""
    ps = ps_index(df)
    m = np.zeros(len(df), dtype=bool)
    m[ps:] = True
    return m


def _split_dir(split: str) -> Path:
    if split == "train":
        return TRAIN_DIR
    if split == "test":
        return TEST_DIR
    raise ValueError(f"split must be 'train' or 'test' (got {split!r})")


def list_well_ids(split: str = "train") -> list[str]:
    """All well ids in a split, sorted, derived from horizontal_well filenames."""
    d = _split_dir(split)
    ids = [p.name[: -len(_HORIZ_SUFFIX)] for p in d.glob(f"*{_HORIZ_SUFFIX}")]
    return sorted(ids)


def horizontal_path(well_id: str, split: str = "train") -> Path:
    return _split_dir(split) / f"{well_id}{_HORIZ_SUFFIX}"


def typewell_path(well_id: str, split: str = "train") -> Path:
    return _split_dir(split) / f"{well_id}{_TYPEWELL_SUFFIX}"


def cross_section_png(well_id: str) -> Path:
    """Train-only cross-section image (reference visualization)."""
    return TRAIN_DIR / f"{well_id}.png"


def load_well(well_id: str, split: str = "train") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (horizontal_df, typewell_df) for one well, each tagged with WELL_ID."""
    h = pd.read_csv(horizontal_path(well_id, split))
    t = pd.read_csv(typewell_path(well_id, split))
    h[WELL_ID] = well_id
    t[WELL_ID] = well_id
    return h, t


def load_horizontal(split: str = "train", well_ids: list[str] | None = None) -> pd.DataFrame:
    """Concatenate every well's horizontal_well CSV into one long DataFrame.

    Each row carries its WELL_ID so you can group for CV (never split a well across
    folds) and for per-well sequence features.
    """
    ids = well_ids if well_ids is not None else list_well_ids(split)
    frames = []
    for wid in ids:
        df = pd.read_csv(horizontal_path(wid, split))
        df[WELL_ID] = wid
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No horizontal_well CSVs found for split={split!r} under {_split_dir(split)}. "
            "Did you run `kaggle competitions download -c rogii-wellbore-geology-prediction -p data/raw` "
            "and unzip into data/raw/?"
        )
    return pd.concat(frames, ignore_index=True)


def load_typewells(split: str = "train", well_ids: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Map well_id -> its type-well DataFrame (one short vertical reference per well)."""
    ids = well_ids if well_ids is not None else list_well_ids(split)
    return {wid: pd.read_csv(typewell_path(wid, split)) for wid in ids}


def load_sample_submission() -> pd.DataFrame:
    from .config import SAMPLE_SUBMISSION
    return pd.read_csv(SAMPLE_SUBMISSION)
