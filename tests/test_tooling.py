"""Tests for the modeling tooling: align, features, train, blend, submission."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src import align, blend, features, features_sequence as fs, submission, train
from src.config import SUBMISSION_ID, SUBMISSION_TARGET, TARGET, TVT_INPUT_COL, WELL_ID


def _toy_well(n=120, ps=40, seed=0):
    """A toy horizontal well: known TVT_input prefix (len ps), NaN after; GR with some NaN."""
    rng = np.random.default_rng(seed)
    md = np.arange(1000, 1000 + n, dtype=float)
    tvt = 11000 + np.cumsum(rng.normal(0, 0.5, n))
    gr = 100 + 20 * np.sin(md / 7) + rng.normal(0, 2, n)
    gr[rng.choice(n, size=n // 10, replace=False)] = np.nan      # ~10% GR NaN
    tvt_input = tvt.copy()
    tvt_input[ps:] = np.nan                                       # NaN after PS
    return pd.DataFrame({"MD": md, "X": md * 0.1, "Y": md * 0.2,
                         "Z": -9000 - np.cumsum(np.abs(rng.normal(0.5, 0.1, n))),
                         "GR": gr, TARGET: tvt, TVT_INPUT_COL: tvt_input})


def _toy_typewell(m=80, seed=1):
    rng = np.random.default_rng(seed)
    tvt = np.linspace(10980, 11030, m)
    gr = 100 + 20 * np.sin(tvt / 7) + rng.normal(0, 2, m)
    return pd.DataFrame({"TVT": tvt, "GR": gr})


# ---- submission --------------------------------------------------------------
def test_submission_build_and_unmapped():
    sample = pd.DataFrame({SUBMISSION_ID: ["w_2", "w_3", "w_4"], SUBMISSION_TARGET: [0.0, 0, 0]})
    ss = submission.build_submission({"w_2": 11.0, "w_3": 12.0, "w_4": 13.0}, sample=sample)
    assert ss[SUBMISSION_TARGET].tolist() == [11.0, 12.0, 13.0]
    with pytest.raises(ValueError):                              # missing w_4 -> unmapped
        submission.build_submission({"w_2": 11.0, "w_3": 12.0}, sample=sample)


# ---- align -------------------------------------------------------------------
def test_align_fill_gr_and_features():
    gr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    filled = align.fill_gr(gr)
    assert np.isfinite(filled).all() and filled[1] == pytest.approx(2.0)
    h, tw = _toy_well(), _toy_typewell()
    af = align.alignment_features(h, tw, window=20)
    assert list(af.columns) == ["align_tvt", "align_shift", "align_cost"]
    assert len(af) == len(h) and np.isfinite(af["align_tvt"]).all()


# ---- sequence deviation ------------------------------------------------------
def test_sequence_deviation_no_lookahead():
    df = _toy_well().assign(**{WELL_ID: "w"})
    out = fs.add_deviation(df, ["GR"])
    for c in ["GR_dev_expmean", "GR_dev_first", "GR_accel"]:
        assert c in out.columns
    # expanding mean of first row == itself -> dev 0 at row 0 (no look-ahead)
    assert out["GR_dev_expmean"].iloc[0] == pytest.approx(0.0, abs=1e-9)


# ---- features parity ---------------------------------------------------------
def test_features_no_leakage_columns():
    h, tw = _toy_well(), _toy_typewell()
    feats = features.build_well_features(h, tw)
    leak = {TARGET, "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA", "Geology"}
    assert leak.isdisjoint(feats.columns), f"leakage cols present: {leak & set(feats.columns)}"
    assert len(feats) == len(h)
    assert {"traj_inclination", "anchor_tvt", "well_progress", "align_tvt"} <= set(feats.columns)


# ---- train factory -----------------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("ROGII_RUN_FIT_TESTS") != "1",
    reason="model fits run on Colab/CI only (no local training); set ROGII_RUN_FIT_TESTS=1 to enable",
)
def test_train_variant_lgb_smoke():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = X["a"] * 3 - X["b"] + rng.normal(0, 0.1, n)
    groups = np.repeat(np.arange(20), n // 20)                   # 20 wells
    res = train.train_variant("t_lgb", "lgb", X, y.to_numpy(), groups,
                              X_test=X.iloc[:10], params={"n_estimators": 50}, save=False)
    assert res.oof.shape == (n,) and np.isfinite(res.oof).all()
    assert len(res.fold_rmses) == 5 and res.oof_rmse < y.std()    # learns something


# ---- blend -------------------------------------------------------------------
def test_blend_nm_oof_improves_and_normalizes():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    good = y + rng.normal(0, 0.5, 500)
    bad = y + rng.normal(0, 2.0, 500)
    w, oof_rmse, _ = blend.nm_optimize_oof({"good": good, "bad": bad}, y)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert w["good"] > w["bad"]                                   # weights the better member up
    rep = blend.marginal_report({"good": good, "bad": bad}, y)
    assert rep[0]["member"] == "good"
