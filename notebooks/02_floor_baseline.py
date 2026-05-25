"""Tier-0 floor baseline — carry-forward the last known TVT past the PS point.

TVT_input is the true TVT on the known prefix and NaN after the Prediction-Start
(PS) point. The naive floor for the scored (post-PS) rows is to carry forward the
last known TVT (geosteering "assume we keep going straight"). This is the anchor
every later model must beat. Establishes the diary's first entry + LB calibration.

Run from repo root:  uv run python notebooks/02_floor_baseline.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src import data
from src.config import SUBMISSION_TARGET, SUBMISSION_ID, SUBMISSIONS, TARGET, TVT_INPUT_COL
from src.evaluate import rmse
from src.observer import Experiment


def ps_index(df: pd.DataFrame) -> int:
    return int(df[TVT_INPUT_COL].notna().sum())


def well_floor(df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """(post-PS true TVT, carry-forward prediction value) for one well."""
    ps = ps_index(df)
    last_known = float(df[TARGET].iloc[ps - 1])
    y_true = df[TARGET].iloc[ps:].to_numpy()
    return y_true, last_known


def main() -> None:
    t0 = time.time()
    exp = Experiment.start(
        version="v0_floor",
        parent=None,
        hypothesis=("Carry-forward last-known TVT across the post-PS region is the naive "
                    "anchor. Expect ~16 ft RMSE (matches Deotte starter CV~15); this is the "
                    "floor that GR-alignment must beat toward the ~9.25 LB leaders."),
        predicted_delta=0.0,            # it's the anchor; no parent to improve on
        confidence="high",
        pipeline_changes=["carry-forward floor"],
        cloud_or_local="local",
    )

    ids = data.list_well_ids("train")
    # per-well floor error arrays, for overall RMSE + GroupKFold-by-well fold RMSE
    per_well_err: dict[str, np.ndarray] = {}
    for wid in ids:
        df = pd.read_csv(data.horizontal_path(wid, "train"))
        y_true, pred = well_floor(df)
        per_well_err[wid] = y_true - pred

    all_err = np.concatenate([per_well_err[w] for w in ids])
    overall = float(np.sqrt((all_err ** 2).mean()))

    # GroupKFold-by-well fold RMSE (5 folds over the 773 wells) — variance of the floor
    fold_rmses = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ids_arr = np.array(ids)
    for _, val_idx in kf.split(ids_arr):
        err = np.concatenate([per_well_err[w] for w in ids_arr[val_idx]])
        fold_rmses.append(float(np.sqrt((err ** 2).mean())))

    print(f"floor overall post-PS RMSE: {overall:.4f} ft")
    print(f"fold RMSEs (5-fold by well): {[round(x, 3) for x in fold_rmses]}")

    # ---- build test submission (carry-forward per test well) ----
    ss = data.load_sample_submission()
    preds: dict[str, float] = {}
    for wid in data.list_well_ids("test"):
        df = pd.read_csv(data.horizontal_path(wid, "test"))
        ps = ps_index(df)
        last_known = float(df[TVT_INPUT_COL].iloc[ps - 1])
        for idx in range(ps, len(df)):
            preds[f"{wid}_{idx}"] = last_known
    ss[SUBMISSION_TARGET] = ss[SUBMISSION_ID].map(preds)
    missing = int(ss[SUBMISSION_TARGET].isna().sum())
    print(f"submission rows: {len(ss)}, unmapped: {missing}")
    assert missing == 0, "some submission ids did not map to a prediction — check id/row alignment"

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    out = SUBMISSIONS / "v0_floor.csv"
    ss.to_csv(out, index=False)
    print(f"wrote {out}")

    exp.record(
        oof_score_mean=overall,
        oof_score_per_fold=fold_rmses,
        holdout_score=overall,           # deterministic floor: no train/holdout gap
        runtime_sec=time.time() - t0,
        extra={"n_train_wells": len(ids), "n_post_ps_rows": int(len(all_err)),
               "submission": str(out.name)},
    )
    exp.commit()
    print("diary: committed v0_floor")


if __name__ == "__main__":
    main()
