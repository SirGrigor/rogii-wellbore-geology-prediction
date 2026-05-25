"""Smoke tests for the comp-shaped modules. No competition data required."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import align, cv, evaluate, features_sequence
from src.config import WELL_ID


def _toy_wells(n_wells=3, n_per=20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for w in range(n_wells):
        md = np.sort(rng.uniform(1000, 2000, n_per))
        gr = np.sin(md / 50) + rng.normal(0, 0.1, n_per)
        for d, g in zip(md, gr):
            rows.append({WELL_ID: f"well{w}", "MD": d, "GR": g})
    return pd.DataFrame(rows)


def test_group_cv_keeps_wells_intact():
    df = _toy_wells()
    for _, tr, va in cv.iter_folds(df, n_folds=3):
        tr_wells = set(df.iloc[tr][WELL_ID])
        va_wells = set(df.iloc[va][WELL_ID])
        assert tr_wells.isdisjoint(va_wells), "a well leaked across the fold boundary"


def test_sequence_features_shapes():
    df = _toy_wells()
    out = features_sequence.add_rolling(df, ["GR"], windows=(3, 5))
    out = features_sequence.add_gradient(out, ["GR"])
    out = features_sequence.add_lags(out, ["GR"], shifts=(1, 2))
    out = features_sequence.add_well_position(out)
    assert len(out) == len(df)
    for col in ["GR_roll3_mean", "GR_roll5_std", "GR_grad", "GR_lag1", "GR_lead2", "well_progress"]:
        assert col in out.columns
    # progress is within [0, 1] per well
    assert out["well_progress"].between(0, 1).all()


def test_metric_dispatch():
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([1.0, 2.0, 3.0])
    assert evaluate.score(y, yhat) == 0.0
    assert evaluate.greater_is_better() is False  # rmse, lower better


def test_dtw_alignment_runs():
    a = np.sin(np.linspace(0, 6, 40))
    b = np.sin(np.linspace(0, 6, 25))
    feats = align.alignment_features(a, b, typewell_depth=np.linspace(0, 100, 25))
    assert feats["matched_tw_index"].shape == (40,)
    assert "matched_tw_depth" in feats
    assert np.isfinite(feats["dtw_cost"])
