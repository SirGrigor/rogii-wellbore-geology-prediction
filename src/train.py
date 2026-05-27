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

import subprocess
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


N_EST_CAP = 4000            # high cap; early stopping decides the real count
EARLY_STOPPING_ROUNDS = 100


def _gpu_available() -> bool:
    """True if a CUDA GPU is present.

    Checks the NVIDIA driver node first (/proc/driver/nvidia/version) — reliable on
    Colab regardless of PATH — then torch.cuda, then nvidia-smi at PATH/known locations.
    (The old shutil.which('nvidia-smi') alone returned False on Colab → silent CPU.)
    """
    import os
    import shutil
    if os.path.exists("/proc/driver/nvidia/version"):
        return True
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    smi = shutil.which("nvidia-smi") or next(
        (p for p in ("/usr/bin/nvidia-smi", "/opt/bin/nvidia-smi") if os.path.exists(p)), None)
    if not smi:
        return False
    try:
        subprocess.run([smi], capture_output=True, check=True, timeout=10)
        return True
    except Exception:
        return False


def default_algo() -> str:
    """Pick the algo: a GPU-capable one (xgb) when a GPU is present, else lgb (CPU).

    LightGBM's Colab pip wheel has no GPU build, so on a T4 we default to **xgb**
    (device="cuda") to actually use the GPU — no manual env needed. Override with
    the ROGII_ALGO env var ("lgb" | "xgb" | "cat").
    """
    import os
    env = os.environ.get("ROGII_ALGO")
    if env:
        return env
    return "xgb" if _gpu_available() else "lgb"


# GPU note: xgb (device="cuda") + cat (task_type="GPU") are reliable on Colab T4.
# LightGBM GPU on Colab needs a special build and is unreliable, so lgb stays CPU —
# early stopping keeps it fast. For a GPU run prefer algo="xgb".
def _lgb_defaults(use_gpu: bool) -> dict:
    return dict(objective="regression", metric="rmse", n_estimators=N_EST_CAP,
                learning_rate=0.03, num_leaves=63, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, min_child_samples=50, random_state=MODEL_SEED,
                n_jobs=-1, verbosity=-1)


def _xgb_defaults(use_gpu: bool) -> dict:
    return dict(objective="reg:squarederror", n_estimators=N_EST_CAP, learning_rate=0.03,
                max_depth=7, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                random_state=MODEL_SEED, n_jobs=-1, tree_method="hist",
                device="cuda" if use_gpu else "cpu")


def _cat_defaults(use_gpu: bool) -> dict:
    p = dict(loss_function="RMSE", iterations=N_EST_CAP, learning_rate=0.03, depth=8,
             random_seed=MODEL_SEED, verbose=False)
    if use_gpu:
        p["task_type"] = "GPU"
    return p


def _n_est_key(algo: str) -> str:
    return "iterations" if algo == "cat" else "n_estimators"


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


def _fit_predict(algo: str, params: dict, Xtr, ytr, Xva, yva, Xte):
    """Train one fold WITH early stopping. Returns (val_pred, test_pred_or_None, best_iter)."""
    if algo == "lgb":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(**params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="rmse",
              callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                         lgb.log_evaluation(0)])
        best = m.best_iteration_ or params[_n_est_key(algo)]
    elif algo == "xgb":
        import xgboost as xgb
        m = xgb.XGBRegressor(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        best = (m.best_iteration or params[_n_est_key(algo)]) + 1
        if str(params.get("device", "")).startswith("cuda"):
            m.set_params(device="cpu")   # predict on CPU (inputs are CPU) → no device-mismatch warning/fallback
    else:  # cat
        m = _make_model(algo, params)
        m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
        best = m.get_best_iteration() or params[_n_est_key(algo)]
    val = m.predict(Xva)
    test = m.predict(Xte) if Xte is not None else None
    return val, test, int(best)


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
    use_gpu: bool | str = "auto",
) -> TrainResult:
    """Train `algo` (lgb|xgb|cat) with GroupKFold-by-well + early stopping. GPU-first.

    use_gpu: "auto" (default — use a CUDA GPU if present), True, or False. xgb (device=cuda)
        and cat (task_type=GPU) run on GPU when available; **lgb stays CPU** (its Colab GPU
        build is unreliable) but early-stops fast. If a GPU fit raises (driver/OOM), the run
        **automatically falls back to CPU** rather than crashing.
    fit_full: after CV, refit ONE model on ALL rows (no eval set → folds' mean best-iteration
        as n_estimators), saved to probs/<version>/model_full.pkl for the Kaggle inference
        notebook. xgb models are saved with device="cpu" so they predict cleanly on a CPU
        inference kernel (avoids the cuda↔cpu "mismatched devices" fallback).
    """
    gpu = _gpu_available() if use_gpu == "auto" else bool(use_gpu)
    if gpu and algo in ("xgb", "cat"):
        try:
            return _train_impl(version, algo, X, y, groups, X_test,
                               params, n_folds, save, fit_full, gpu=True)
        except Exception as e:
            print(f"[train] GPU run failed ({type(e).__name__}: {e}) — falling back to CPU")
    return _train_impl(version, algo, X, y, groups, X_test,
                       params, n_folds, save, fit_full, gpu=False)


def _train_impl(version, algo, X, y, groups, X_test, params, n_folds, save, fit_full, *, gpu):
    t0 = time.time()
    defaults = {"lgb": _lgb_defaults, "xgb": _xgb_defaults, "cat": _cat_defaults}[algo](gpu)
    defaults.update(params or {})
    dev = "GPU" if (gpu and algo in ("xgb", "cat")) else "CPU"
    # NB: 'on CPU' for lgb is BY DESIGN (LightGBM's Colab pip wheel has no GPU build), NOT a missing
    # GPU — cat/xgb still use the T4. The old 'gpu_available=' label wrongly implied no GPU; fixed.
    note = " (LightGBM: no Colab GPU build → CPU by design; cat/xgb use the GPU)" if algo == "lgb" else ""
    print(f"[train] {algo} on {dev}{note}; early stopping rounds={EARLY_STOPPING_ROUNDS}")

    X = X.reset_index(drop=True)
    y = np.asarray(y, dtype=float)
    oof = np.full(len(y), np.nan)
    test_acc = np.zeros(len(X_test)) if X_test is not None else None
    fold_rmses: list[float] = []
    best_iters: list[int] = []

    cv = GroupKFold(n_splits=n_folds)
    for tr, va in cv.split(X, y, groups=groups):
        val, test, best = _fit_predict(algo, defaults, X.iloc[tr], y[tr], X.iloc[va], y[va], X_test)
        oof[va] = val
        fold_rmses.append(rmse(y[va], val))
        best_iters.append(best)
        if test_acc is not None:
            test_acc += test / n_folds
    print(f"[train] fold best-iters {best_iters} (cap {N_EST_CAP})")

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
        # no eval set on the full fit → fix n_estimators at the folds' mean best-iter (+10%)
        full_params = dict(defaults)
        full_params[_n_est_key(algo)] = max(50, int(np.mean(best_iters) * 1.1))
        full = _make_model(algo, full_params)
        full.fit(X, y) if algo != "xgb" else full.fit(X, y, verbose=False)
        if algo == "xgb":
            full.set_params(device="cpu")   # portable: Kaggle inference kernel may be CPU
        d = PROBS / version
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(full, d / "model_full.pkl")
    return res
