"""Cross-validation: per-well grouping.

A single wellbore is a depth-ordered sequence; rows within one well are highly
autocorrelated. Splitting a well across folds leaks the answer. So all CV here is
**grouped by well_id** — every well lands entirely in one fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import CV_SEED, N_FOLDS, WELL_ID


def make_group_cv(n_folds: int = N_FOLDS):
    """A GroupKFold splitter. (GroupKFold is deterministic; CV_SEED is kept for
    parity with the rest of the config and any future shuffled variant.)"""
    _ = CV_SEED  # documented: GroupKFold itself is seedless
    return GroupKFold(n_splits=n_folds)


def well_groups(df: pd.DataFrame) -> np.ndarray:
    """The group vector (well_id per row) to pass as `groups=` to the splitter."""
    if WELL_ID not in df.columns:
        raise KeyError(
            f"{WELL_ID!r} column missing — load rows via data.load_horizontal(), "
            "which tags every row with its well id."
        )
    return df[WELL_ID].to_numpy()


def iter_folds(df: pd.DataFrame, n_folds: int = N_FOLDS):
    """Yield (fold_idx, train_idx, val_idx) with no well crossing the boundary."""
    cv = make_group_cv(n_folds)
    groups = well_groups(df)
    for i, (tr, va) in enumerate(cv.split(df, groups=groups)):
        yield i, tr, va
