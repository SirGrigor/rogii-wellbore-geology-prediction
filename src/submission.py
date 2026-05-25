"""Build + validate the competition submission.

Submission format (confirmed): one row per scored (post-PS) point,
    id  = "{well_id}_{iloc}"   (0-based row index within the well's horizontal CSV)
    tvt = predicted True Vertical Thickness (feet)

The set of scored ids is fixed by `sample_submission.csv`. These helpers map your
predictions onto it and assert every id is covered (0 unmapped) before writing —
the alignment bug that would silently tank the LB is caught here, locally.

NOTE: this competition is **kernels-only** (is_kernels_submissions_only=True). The
CSV produced here is submitted by a Kaggle *notebook* (see notebooks/kaggle_infer),
not via `kaggle competitions submit`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .config import SUBMISSION_ID, SUBMISSION_TARGET, SUBMISSIONS


def well_pred_ids(well_id: str, n_post: int, ps: int) -> list[str]:
    """The submission ids for a well's post-PS rows, in iloc order."""
    return [f"{well_id}_{ps + i}" for i in range(n_post)]


def build_submission(pred_by_id: dict[str, float], sample: pd.DataFrame | None = None) -> pd.DataFrame:
    """Map an {id: tvt} dict onto sample_submission; assert full, finite coverage."""
    ss = sample.copy() if sample is not None else data.load_sample_submission()
    ss[SUBMISSION_TARGET] = ss[SUBMISSION_ID].map(pred_by_id).astype(float)
    _validate(ss)
    return ss


def build_submission_from_wells(well_preds: dict[str, np.ndarray],
                                split: str = "test",
                                sample: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build from {well_id: predictions-for-post-PS-rows (iloc order)}.

    Reads each test well only to recover its PS index (so ids line up exactly with
    the horizontal CSV row indices the sample_submission uses).
    """
    pred_by_id: dict[str, float] = {}
    for wid, preds in well_preds.items():
        df = pd.read_csv(data.horizontal_path(wid, split))
        ps = data.ps_index(df)
        n_post = len(df) - ps
        preds = np.asarray(preds, dtype=float)
        if len(preds) != n_post:
            raise ValueError(
                f"well {wid}: got {len(preds)} preds but {n_post} post-PS rows (ps={ps}, n={len(df)})"
            )
        for i, p in enumerate(preds):
            pred_by_id[f"{wid}_{ps + i}"] = float(p)
    return build_submission(pred_by_id, sample=sample)


def _validate(ss: pd.DataFrame) -> None:
    n_missing = int(ss[SUBMISSION_TARGET].isna().sum())
    if n_missing:
        raise ValueError(f"{n_missing} submission ids have no prediction (unmapped) — id/row mismatch")
    if not np.isfinite(ss[SUBMISSION_TARGET].to_numpy()).all():
        raise ValueError("submission contains non-finite predictions (inf/NaN)")


def save_submission(ss: pd.DataFrame, name: str) -> Path:
    """Write submissions/<name>.csv (id,tvt). Returns the path."""
    _validate(ss)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    out = SUBMISSIONS / (name if name.endswith(".csv") else f"{name}.csv")
    ss[[SUBMISSION_ID, SUBMISSION_TARGET]].to_csv(out, index=False)
    return out
