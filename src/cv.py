"""Cross-validation: per-well grouping + a sacred well-holdout.

A single wellbore is a depth-ordered sequence; rows within one well are highly
autocorrelated. Splitting a well across folds leaks the answer. So all CV here is
**grouped by well_id** — every well lands entirely in one fold.

**Sacred holdout (validation discipline, handover §0/§7):** the 3-well test LB is
near-pure noise — NEVER tune or select to it (it's worse than S6E5's v51 trap). Carve
~150 of the 773 wells into a SACRED holdout, never used in CV/tuning, touched only for
a final honest check. Choose models on well-grouped OOF (dev wells) + the sacred set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import CV_SEED, HOLDOUT_SEED, N_FOLDS, WELL_ID

N_SACRED_WELLS = 150  # ~19% of 773 held out, never tuned to


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


def sacred_split(well_ids: list[str], n_sacred: int = N_SACRED_WELLS,
                 seed: int = HOLDOUT_SEED) -> tuple[list[str], list[str]]:
    """Split well ids into (dev_wells, sacred_wells). Deterministic by seed.

    dev_wells → all CV/tuning/model-selection. sacred_wells → touched ONLY for a final
    honest check, never tuned to (the 3-well LB is noise; this is the real holdout).
    """
    ids = sorted(well_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    sacred_idx = set(perm[:n_sacred].tolist())
    sacred = [ids[i] for i in range(len(ids)) if i in sacred_idx]
    dev = [ids[i] for i in range(len(ids)) if i not in sacred_idx]
    return dev, sacred
