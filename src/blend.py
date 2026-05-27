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


def caruana_select(
    oof: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    score_fn=rmse,
    greater_is_better: bool = False,
    n_bag: int = 20,
    bag_frac: float = 0.5,
    n_init: int = 1,
    max_iters: int = 100,
    rank_norm: bool = False,
    seed: int = 42,
) -> tuple[dict[str, float], float, dict]:
    """Caruana greedy ensemble selection — with replacement, bagged, sorted-init.

    Caruana et al. 2004/2006 (ICML/ICDM). The overfit-RESISTANT alternative to
    `nm_optimize_oof`: greedily add the one member (with replacement) that most
    improves the OOF metric, starting from the best single model, repeated over
    bootstrap subsets of the model LIBRARY and averaged. Weights are selection
    counts → strictly positive, and no member can be driven to a degenerate weight
    the way Nelder-Mead can (that overfit burned us in S6E5 v51). This is the
    documented engine behind recent 1st-place tabular finishes (KG L56) and the
    blend method we previously lacked.

    Optimize+select on the LABELED OOF only (never LB/holdout — L53).
    `score_fn(y_true, y_pred) -> float`; `greater_is_better` flips RMSE↔AUC.
    Returns (weights summing to 1, OOF score of the bagged blend, info dict).
    """
    names, P = _stack(oof, rank_norm)              # P: (n_rows, n_members)
    y = np.asarray(y, dtype=float)
    n_members = len(names)
    if n_members == 1:
        return {names[0]: 1.0}, float(score_fn(y, P[:, 0])), {"n_members": 1}

    sign = 1.0 if greater_is_better else -1.0
    def gain(pred: np.ndarray) -> float:           # higher == better, always
        return sign * float(score_fn(y, pred))

    solo = np.array([gain(P[:, i]) for i in range(n_members)])
    rng = np.random.default_rng(seed)
    total = np.zeros(n_members)                    # accumulated per-bag weight votes

    for _ in range(n_bag):
        k = min(n_members, max(2, int(np.ceil(bag_frac * n_members))))
        lib = rng.choice(n_members, size=k, replace=False)
        order = lib[np.argsort(-solo[lib])]        # sorted init: best members first
        counts = np.zeros(n_members)
        init = order[: max(1, min(n_init, len(order)))]
        for i in init:
            counts[i] += 1.0
        cur_sum = P[:, init].sum(axis=1).astype(float)
        cur_n = float(counts.sum())
        best = gain(cur_sum / cur_n)
        for _ in range(max_iters):
            scores = [(gain((cur_sum + P[:, j]) / (cur_n + 1)), j) for j in lib]
            s, j = max(scores, key=lambda t: t[0])
            if s <= best + 1e-12:                  # no further improvement
                break
            counts[j] += 1.0
            cur_sum = cur_sum + P[:, j]
            cur_n += 1.0
            best = s
        total += counts / counts.sum()             # equal vote per bag

    w = total / total.sum()
    weights = dict(zip(names, w.tolist()))
    blend_score = float(score_fn(y, P @ w))
    bi = int(np.argmax(solo))
    info = {
        "n_members": n_members, "n_bag": n_bag, "bag_frac": bag_frac,
        "n_init": n_init, "rank_norm": rank_norm,
        "best_single": names[bi],
        "best_single_score": float(score_fn(y, P[:, bi])),
        "simple_mean_score": float(score_fn(y, P.mean(axis=1))),
        "blend_score": blend_score,
        "n_selected": int((w > 1e-4).sum()),
    }
    return weights, blend_score, info


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
