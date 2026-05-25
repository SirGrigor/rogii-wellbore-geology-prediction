"""Regression variant factory — GroupKFold-by-well training for TVT.

One entry point, `train_variant`, trains a gradient-boosting regressor across
GroupKFold-by-well folds (a well never crosses the fold boundary), returns OOF +
test predictions + per-fold RMSE, and persists artifacts to `probs/<version>/`
for retroactive blending. The X/y passed in are already restricted to the scored
(post-PS) rows by `features.build_dataset`, so the RMSE here is the metric RMSE.

Comp-shaped + regression-specific (the shared utils factory is classification +
Registry-coupled). Candidate to promote to kaggle-playground-utils once proven —
see TOOLKIT_PORT.md. Heavy/full-data runs go to Colab (cloud-first).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import MODEL_SEED, N_FOLDS, PROBS
from .evaluate import rmse


@dataclass
class TrainResult:
    version: str
    algo: str
    oof: np.ndarray
    test_pred: np.ndarray | None
    fold_rmses: list[float]
    oof_rmse: float
    runtime_sec: float
    params: dict[str, Any] = field(default_factory=dict)


def _lgb_defaults() -> dict:
    return dict(objective="regression", metric="rmse", n_estimators=2000,
                learning_rate=0.03, num_leaves=63, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, min_child_samples=50, random_state=MODEL_SEED,
                n_jobs=-1, verbosity=-1)


def _xgb_defaults() -> dict:
    return dict(objective="reg:squarederror", n_estimators=2000, learning_rate=0.03,
                max_depth=7, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                random_state=MODEL_SEED, n_jobs=-1, tree_method="hist")


def _cat_defaults() -> dict:
    return dict(loss_function="RMSE", iterations=2000, learning_rate=0.03, depth=8,
                random_seed=MODEL_SEED, verbose=False)


def _make_model(algo: str, params: dict):
    if algo == "lgb":
        import lightgbm as lgb
        return lgb.LGBMRegressor(**params)
    if algo == "xgb":
        import xgboost as xgb
        return xgb.XGBRegressor(**params)
    if algo == "cat":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**params)
    raise ValueError(f"unknown algo {algo!r} (lgb|xgb|cat)")


def _fit_predict(algo: str, params: dict, Xtr, ytr, Xva, Xte):
    """Train one fold; return (val_pred, test_pred_or_None)."""
    m = _make_model(algo, params)
    m.fit(Xtr, ytr) if algo != "xgb" else m.fit(Xtr, ytr, verbose=False)
    val = m.predict(Xva)
    test = m.predict(Xte) if Xte is not None else None
    return val, test


def train_variant(
    version: str,
    algo: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    X_test: pd.DataFrame | None = None,
    *,
    params: dict | None = None,
    n_folds: int = N_FOLDS,
    save: bool = True,
    fit_full: bool = False,
) -> TrainResult:
    """Train `algo` (lgb|xgb|cat) with GroupKFold-by-well. Returns OOF + test + per-fold RMSE.

    fit_full: after CV, refit one model on ALL rows and save it (joblib) to
    probs/<version>/model_full.pkl — the deployable model the Kaggle inference
    notebook loads (kernels-only submission). CV still gives the honest OOF RMSE.
    """
    t0 = time.time()
    defaults = {"lgb": _lgb_defaults, "xgb": _xgb_defaults, "cat": _cat_defaults}[algo]()
    defaults.update(params or {})

    X = X.reset_index(drop=True)
    y = np.asarray(y, dtype=float)
    oof = np.full(len(y), np.nan)
    test_acc = np.zeros(len(X_test)) if X_test is not None else None
    fold_rmses: list[float] = []

    cv = GroupKFold(n_splits=n_folds)
    for tr, va in cv.split(X, y, groups=groups):
        val, test = _fit_predict(algo, defaults, X.iloc[tr], y[tr], X.iloc[va], X_test)
        oof[va] = val
        fold_rmses.append(rmse(y[va], val))
        if test_acc is not None:
            test_acc += test / n_folds

    res = TrainResult(version=version, algo=algo, oof=oof, test_pred=test_acc,
                      fold_rmses=fold_rmses, oof_rmse=rmse(y, oof),
                      runtime_sec=time.time() - t0, params=defaults)
    if save:
        d = PROBS / version
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "oof.npy", oof)
        if test_acc is not None:
            np.save(d / "test.npy", test_acc)
    if fit_full:
        import joblib
        full = _make_model(algo, defaults)
        full.fit(X, y) if algo != "xgb" else full.fit(X, y, verbose=False)
        d = PROBS / version
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(full, d / "model_full.pkl")
    return res
