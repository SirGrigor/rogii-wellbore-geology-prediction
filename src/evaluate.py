"""Regression metrics, wired to the project MetricSpec.

`score()` returns the competition metric on its natural scale. `greater_is_better`
comes from config.METRIC so the diary autoflags and any blend optimizer agree on
direction. Update the dispatch once Phase-0 confirms the exact metric.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import METRIC


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


_REGISTRY = {
    "rmse": rmse,
    "mae": mae,
}


def score(y_true, y_pred) -> float:
    """The competition metric (natural scale). Lower or higher better per METRIC."""
    fn = _REGISTRY.get(METRIC.name)
    if fn is None:
        raise NotImplementedError(
            f"Metric {METRIC.name!r} not implemented in evaluate._REGISTRY. "
            "Add it once Phase-0 confirms the exact metric."
        )
    return fn(y_true, y_pred)


def greater_is_better() -> bool:
    return METRIC.greater_is_better
