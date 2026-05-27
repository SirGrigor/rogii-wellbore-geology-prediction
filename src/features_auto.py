"""Large-scale automated feature engineering — the #1 documented winner's edge (KG survey: FE-depth,
5/6 comps; NVIDIA Grandmasters Playbook #3 "generate more features, discover more patterns").

We've only ever hand-added a handful. This GENERATES many candidates — pairwise product/diff/ratio of the
most-important base features + per-well group aggregations — then IMPORTANCE-SELECTS the best. Names are
parseable so the exact selected set is rebuildable on sacred/test. Leakage-safe: selection uses a quick
GBDT importance (not target stats directly); group-aggs are within-well (no cross-row target leak).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _top_base(df, base, y, k):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=120, num_leaves=63, verbosity=-1, n_jobs=-1)
    m.fit(df[base].fillna(0), y)
    return pd.Series(m.feature_importances_, index=base).sort_values(ascending=False).head(k).index.tolist()


def _feat(df, name, group):
    p = name.split("|")
    if p[0] == "prod": return df[p[1]] * df[p[2]]
    if p[0] == "diff": return df[p[1]] - df[p[2]]
    if p[0] == "rat":  return (df[p[1]] / (df[p[2]].abs() + 1e-3)).clip(-1e4, 1e4)
    if p[0] == "gmean": return df.groupby(group)[p[1]].transform("mean")
    if p[0] == "gstd":  return df.groupby(group)[p[1]].transform("std").fillna(0.0)
    raise ValueError(name)


def candidate_names(top, n_grp=15):
    names = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            names += [f"prod|{top[i]}|{top[j]}", f"diff|{top[i]}|{top[j]}", f"rat|{top[i]}|{top[j]}"]
    for a in top[:n_grp]:
        names += [f"gmean|{a}", f"gstd|{a}"]
    return names


def build(df, names, group="well") -> pd.DataFrame:
    """Rebuild the named features on any frame (parseable names)."""
    return pd.DataFrame({n: _feat(df, n, group).astype(np.float32) for n in names}, index=df.index)


def select(df, base, y, k=30, keep=150, group="well"):
    """Generate ~k·(k-1)/2·3 + 2·15 candidates, train one GBDT, keep the `keep` most-important candidates
    with positive importance. Returns the selected feature NAMES (rebuild with build())."""
    import lightgbm as lgb
    top = _top_base(df, base, y, k)
    names = candidate_names(top)
    cand = build(df, names, group)
    X = pd.concat([df[base], cand], axis=1).fillna(0.0).replace([np.inf, -np.inf], 0.0)
    m = lgb.LGBMRegressor(n_estimators=200, num_leaves=63, verbosity=-1, n_jobs=-1)
    m.fit(X, y)
    imp = pd.Series(m.feature_importances_, index=X.columns)
    ci = imp[names].sort_values(ascending=False)
    sel = ci[ci > 0].head(keep).index.tolist()
    return sel, len(names)
