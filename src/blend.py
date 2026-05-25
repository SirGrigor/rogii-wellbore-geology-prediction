"""Supervised blend — optimize AND select on the labeled GroupKFold OOF (never LB/holdout).

Carries the hard-won S6E5 blending discipline (docs/strategy.md §2):
  - L53: optimize blend weights on a LABELED CV signal, not by probing the LB. Here the
    signal is the GroupKFold OOF over the 773 train wells (we have full TVT labels).
  - L48: negative weights overfit when tuned on holdout — optimize them on OOF and SELECT the
    final blend by OOF RMSE. `allow_negative` is opt-in.
  - exact-metric: minimize RMSE directly (the competition metric).
  - decorrelation is necessary but NOT sufficient (L41/L50/L52): `marginal_report` shows whether
    each member actually lowers OOF RMSE; drop members that don't.

Self-contained (no Registry coupling). Predictions are TVT-scale floats.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

from .evaluate import rmse


def rank_normalize(a: np.ndarray) -> np.ndarray:
    """Rank-uniform to [0,1] — use when members live on different scales."""
    a = np.asarray(a, dtype=float)
    return (rankdata(a) - 1.0) / (len(a) - 1.0)


def _stack(oof: dict[str, np.ndarray], rank_norm: bool) -> tuple[list[str], np.ndarray]:
    names = list(oof)
    cols = [rank_normalize(oof[n]) if rank_norm else np.asarray(oof[n], float) for n in names]
    return names, np.column_stack(cols)


def nm_optimize_oof(
    oof: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    allow_negative: bool = False,
    rank_norm: bool = False,
    init: dict[str, float] | None = None,
) -> tuple[dict[str, float], float, dict]:
    """Nelder-Mead blend weights minimizing OOF RMSE. Weights normalized to sum 1.

    allow_negative lets a weight subtract a correlated member's error (powerful but
    overfit-prone — only honest because we optimize+select on OOF, per L48).
    """
    names, P = _stack(oof, rank_norm)
    y = np.asarray(y, dtype=float)
    w0 = (np.array([init.get(n, 1.0 / len(names)) for n in names]) if init
          else np.ones(len(names)) / len(names))

    def loss(w: np.ndarray) -> float:
        if not allow_negative:
            w = np.clip(w, 0.0, None)
        s = w.sum()
        if abs(s) < 1e-9:
            return 1e9
        return rmse(y, P @ (w / s))

    res = minimize(loss, w0, method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-7, "maxiter": 4000})
    w = res.x if allow_negative else np.clip(res.x, 0.0, None)
    w = w / w.sum()
    return dict(zip(names, w.tolist())), float(res.fun), {
        "init_rmse": float(rmse(y, P @ w0)), "niter": int(res.nit),
        "allow_negative": allow_negative, "rank_norm": rank_norm,
    }


def apply_blend(preds: dict[str, np.ndarray], weights: dict[str, float],
                rank_norm: bool = False) -> np.ndarray:
    """Weighted blend of (test) predictions using the OOF-fitted weights."""
    names = list(weights)
    cols = [rank_normalize(preds[n]) if rank_norm else np.asarray(preds[n], float) for n in names]
    w = np.array([weights[n] for n in names])
    return np.column_stack(cols) @ w


def pairwise_corr(oof: dict[str, np.ndarray]) -> "np.ndarray":
    """Pearson correlation matrix across members (for the ρ diversity view)."""
    names = list(oof)
    M = np.column_stack([oof[n] for n in names])
    return np.corrcoef(M, rowvar=False)


def marginal_report(oof: dict[str, np.ndarray], y: np.ndarray, **kw) -> list[dict]:
    """Per-member: solo OOF RMSE, and the blend RMSE WITHOUT it (leave-one-out).

    A member that doesn't raise leave-one-out RMSE (i.e. removing it doesn't hurt)
    is not contributing — drop it (decorrelation necessary-not-sufficient).
    """
    y = np.asarray(y, float)
    _, full_rmse, _ = nm_optimize_oof(oof, y, **kw)
    rows = []
    for n in oof:
        solo = rmse(y, oof[n])
        rest = {k: v for k, v in oof.items() if k != n}
        loo = nm_optimize_oof(rest, y, **kw)[1] if rest else float("inf")
        rows.append({"member": n, "solo_rmse": round(solo, 4),
                     "loo_blend_rmse": round(loo, 4),
                     "contribution": round(loo - full_rmse, 4)})  # >0 = helps
    return sorted(rows, key=lambda r: -r["contribution"])
