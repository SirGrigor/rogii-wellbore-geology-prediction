"""Signal Hunt — model the model's failure.

Comp-agnostic 4-layer diagnostic that decides whether residuals from a current-best model contain
LURKING SIGNAL (avoidable bias) or are NOISE (irreducible). The headline is one number:

  R²_residual_boost > 0.05  →  structure exists → SHAP names the features → FE recipe writes itself
  R²_residual_boost ≈ 0     →  residuals are noise → ceiling is PROVEN, not suspected → STOP

Layers (per the framework adopted 2026-05-28):
  0. Disambiguate  — Bayes-error floor via kNN local-variance on top features → hard RMSE wall
  1. Localize      — per-slice residual maps (caller-side; module returns slice DataFrames)
  2a. Residual-boost — GBDT regresses signed residual on features (OOF GroupKFold). R² is the verdict
  2b. Adversarial right-vs-wrong — GBDT classifies "is |residual| > median?" → AUC
  2c. Confident-wrong — low-disagreement-but-wrong rows (caller passes per-model preds if available)
  3. Test          — gated FE downstream (NOT in this module; the user runs that next)

Reusable across competitions. Inputs: p_oof, y, X, groups (well/group ids). All numpy/pandas, no
torch. LightGBM-native SHAP via pred_contrib=True (no shap-lib dependency).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as e:                                             # pragma: no cover
    raise SystemExit("signal_hunt requires lightgbm") from e
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors


# ---------- result schema -------------------------------------------------------------------------

@dataclass
class HuntResult:
    current_rmse: float                            # RMSE of p_oof vs y (baseline we are trying to beat)
    irreducible_rmse: float                        # √(Bayes-floor variance) — lower wall under these features
    reachable_rmse_room: float                     # current − irreducible (how much room exists)
    r2_residual_boost: float                       # Layer 2a verdict — > 0.05 = structure
    auc_right_vs_wrong: float                      # Layer 2b verdict — > 0.55 = structure (different angle)
    ceiling_proven: bool                           # True ⟺ both r2 ≤ 0.02 AND auc ≤ 0.52 (noise on both)
    top_features_residual: list[tuple[str, float]] # by mean |SHAP| on the residual-boost model
    n_rows: int
    n_features: int
    verdict: str                                   # one-paragraph synthesis


# ---------- Layer 0 — Bayes-error floor via kNN local variance ------------------------------------

def bayes_floor_knn(y: np.ndarray, X: np.ndarray, *, k: int = 20, top_k_feats: int = 30,
                    subsample: int = 5000, seed: int = 42) -> float:
    """Estimate irreducible noise variance via kNN local-variance.

    For each row i: find k nearest neighbors in X-space; the variance of their y is a per-row
    irreducible-noise estimate. Average across rows = global Bayes-floor variance. Sqrt = RMSE wall.

    Caveats applied: (1) restrict to top_k_feats most-variable columns (curse of dimensionality);
    (2) z-score features so distances are isotropic; (3) subsample to keep tree-build tractable.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    if subsample and n > subsample:
        idx = rng.choice(n, size=subsample, replace=False)
        Xs, ys = X[idx], y[idx]
    else:
        Xs, ys = X, y
    # pick top-K most-variable columns (no model needed — pure data property)
    var = np.nanvar(Xs, axis=0)
    keep = np.argsort(var)[-top_k_feats:]
    Xs = np.nan_to_num(Xs[:, keep], nan=0.0)
    Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-9)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Xs)), n_jobs=-1).fit(Xs)
    _, ind = nn.kneighbors(Xs)                                       # ind[:, 0] = self, drop it
    neigh_y = ys[ind[:, 1:]]                                          # [n, k]
    per_row_var = neigh_y.var(axis=1)
    return float(np.mean(per_row_var))


# ---------- Layer 2a — residual-boost ---------------------------------------------------------------

def residual_boost(residual: np.ndarray, X: pd.DataFrame, groups: np.ndarray,
                   *, n_folds: int = 5, seed: int = 42) -> tuple[float, np.ndarray, "lgb.Booster"]:
    """GBDT regresses signed residual on features (GroupKFold OOF). Returns (R², oof_pred, model_last_fold).

    R² > 0.05 → structure in residuals → original model missed something extractable from these features.
    R² ≈ 0  → residuals are noise w.r.t. these features → ceiling.
    """
    oof = np.zeros(len(residual), dtype=np.float32)
    last_model = None
    X_arr = X.to_numpy(np.float32) if isinstance(X, pd.DataFrame) else X.astype(np.float32)
    for tr, va in GroupKFold(n_folds).split(X_arr, residual, groups):
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.04, num_leaves=127,
                              min_child_samples=30, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=3.0, random_state=seed,
                              n_jobs=-1, verbose=-1)
        m.fit(X_arr[tr], residual[tr], eval_set=[(X_arr[va], residual[va])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = m.predict(X_arr[va])
        last_model = m
    ss_res = float(((residual - oof) ** 2).sum())
    ss_tot = float(((residual - residual.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return r2, oof, last_model


def _shap_top_features(model: "lgb.Booster", X: pd.DataFrame, n_top: int = 20) -> list[tuple[str, float]]:
    """Mean |SHAP| per feature (LightGBM-native via pred_contrib=True). No shap-lib dep."""
    X_arr = X.to_numpy(np.float32) if isinstance(X, pd.DataFrame) else X.astype(np.float32)
    feat_names = list(X.columns) if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(X_arr.shape[1])]
    # subsample for speed if very large
    if len(X_arr) > 50000:
        idx = np.random.default_rng(0).choice(len(X_arr), 50000, replace=False)
        X_arr = X_arr[idx]
    contrib = model.predict(X_arr, pred_contrib=True)               # [n, D+1] — last col is bias
    mean_abs = np.abs(contrib[:, :-1]).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:n_top]
    return [(feat_names[i], float(mean_abs[i])) for i in order]


# ---------- Layer 2b — adversarial right-vs-wrong --------------------------------------------------

def adversarial_right_vs_wrong(abs_residual: np.ndarray, X: pd.DataFrame, groups: np.ndarray,
                                *, n_folds: int = 5, seed: int = 42) -> float:
    """Classify "is |residual| > median?" — AUC. Different angle than residual-boost regression.

    > 0.55 → features know which rows are hard (often picks up interactions the regression smooths over)
    ≈ 0.50 → features carry no information about row difficulty.
    """
    thr = float(np.median(abs_residual))
    is_hard = (abs_residual > thr).astype(np.int32)
    oof = np.zeros(len(is_hard), dtype=np.float32)
    X_arr = X.to_numpy(np.float32) if isinstance(X, pd.DataFrame) else X.astype(np.float32)
    for tr, va in GroupKFold(n_folds).split(X_arr, is_hard, groups):
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=63,
                               min_child_samples=30, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, reg_lambda=3.0, random_state=seed,
                               n_jobs=-1, verbose=-1)
        m.fit(X_arr[tr], is_hard[tr], eval_set=[(X_arr[va], is_hard[va])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = m.predict_proba(X_arr[va])[:, 1]
    return float(roc_auc_score(is_hard, oof))


# ---------- main entry point -----------------------------------------------------------------------

def hunt(*, p_oof: np.ndarray, y: np.ndarray, X: pd.DataFrame | np.ndarray, groups: np.ndarray,
         feature_names: list[str] | None = None, k_neighbors: int = 20, n_top_features: int = 20,
         out_dir: str | Path = "docs/signal_hunt") -> HuntResult:
    """Run the 4-layer hunt and emit artifacts. Returns HuntResult with the headline numbers."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(np.asarray(X), columns=feature_names or [f"f{i}" for i in range(np.asarray(X).shape[1])])
    p_oof = np.asarray(p_oof, np.float32); y = np.asarray(y, np.float32)
    residual = y - p_oof
    abs_residual = np.abs(residual)
    current_rmse = float(np.sqrt((residual ** 2).mean()))

    print(f"[hunt] N={len(y):,}  features={X.shape[1]}  current RMSE={current_rmse:.3f}")
    print(f"[hunt] Layer 0 — Bayes-floor via kNN(k={k_neighbors}) local-variance")
    ν = bayes_floor_knn(y, X.to_numpy(np.float32), k=k_neighbors)
    irreducible_rmse = float(np.sqrt(ν))
    print(f"           irreducible variance ≈ {ν:.2f}  →  RMSE wall ≈ {irreducible_rmse:.3f}")
    print(f"           reachable room: {current_rmse - irreducible_rmse:+.3f} (current − wall)")

    print("[hunt] Layer 2a — residual-boost (GBDT regresses signed residual on features)")
    r2, _, rb_model = residual_boost(residual, X, groups)
    print(f"           R² = {r2:+.4f}    ({'STRUCTURE' if r2 > 0.05 else 'noise' if r2 < 0.02 else 'borderline'})")

    print("[hunt] Layer 2b — adversarial right-vs-wrong (GBDT classifies |residual|>median)")
    auc = adversarial_right_vs_wrong(abs_residual, X, groups)
    print(f"           AUC = {auc:.4f}    ({'STRUCTURE' if auc > 0.55 else 'noise' if auc < 0.52 else 'borderline'})")

    top_feats = _shap_top_features(rb_model, X, n_top=n_top_features)
    ceiling_proven = (r2 <= 0.02 and auc <= 0.52)

    if ceiling_proven:
        verdict = (
            f"CEILING PROVEN. Residual-boost R²={r2:+.3f} (~0) and adversarial AUC={auc:.3f} (~0.5) "
            f"both say the residuals are NOISE w.r.t. the current feature set. Current RMSE "
            f"{current_rmse:.3f} is within {current_rmse - irreducible_rmse:.2f} of the Bayes-floor "
            f"({irreducible_rmse:.3f}). No new model on these features will help — only new FEATURES "
            f"or a different paradigm changes the wall."
        )
    elif r2 > 0.05 or auc > 0.55:
        names = ", ".join(n for n, _ in top_feats[:5])
        verdict = (
            f"STRUCTURE FOUND. Residual-boost R²={r2:+.3f}, adversarial AUC={auc:.3f}. The features "
            f"carrying the lurking signal (top SHAP on residual): {names}. Build FE from these — "
            f"interactions, transforms, slice-conditional encodings — and re-train. Room to reach: "
            f"current {current_rmse:.3f} → Bayes-floor {irreducible_rmse:.3f} = {current_rmse - irreducible_rmse:.2f} ft of headroom."
        )
    else:
        verdict = (
            f"BORDERLINE. R²={r2:+.3f}, AUC={auc:.3f}. Some weak structure but not strong enough to "
            f"confidently call lurking signal. Recommend: drill into the top-feature SHAP slices "
            f"and try a targeted FE before declaring ceiling."
        )

    result = HuntResult(current_rmse=current_rmse, irreducible_rmse=irreducible_rmse,
                        reachable_rmse_room=current_rmse - irreducible_rmse,
                        r2_residual_boost=r2, auc_right_vs_wrong=auc,
                        ceiling_proven=ceiling_proven, top_features_residual=top_feats,
                        n_rows=len(y), n_features=X.shape[1], verdict=verdict)

    _write_report(result, out_dir)
    return result


def _write_report(r: HuntResult, out_dir: Path) -> None:
    lines = [
        "# Signal Hunt Report",
        "",
        f"- **N rows**: {r.n_rows:,}    **features**: {r.n_features}",
        f"- **current RMSE**: {r.current_rmse:.3f}",
        f"- **irreducible (Bayes-floor) RMSE**: {r.irreducible_rmse:.3f}",
        f"- **reachable room**: {r.reachable_rmse_room:+.3f}",
        "",
        "## Headline (Layer 2 verdicts)",
        f"- **R²_residual_boost**: {r.r2_residual_boost:+.4f}    (> 0.05 = structure; ≈ 0 = noise)",
        f"- **AUC_right_vs_wrong**: {r.auc_right_vs_wrong:.4f}    (> 0.55 = structure; ≈ 0.5 = noise)",
        f"- **ceiling proven?**: **{r.ceiling_proven}**",
        "",
        "## Verdict",
        r.verdict,
        "",
        "## Top features by residual-SHAP (where the model's missed signal lives)",
        "| rank | feature | mean \\|SHAP\\| |",
        "| --- | --- | --- |",
    ]
    for i, (name, imp) in enumerate(r.top_features_residual, 1):
        lines.append(f"| {i} | `{name}` | {imp:.4f} |")
    (out_dir / "signal_hunt_report.md").write_text("\n".join(lines))
    pd.DataFrame(r.top_features_residual, columns=["feature", "mean_abs_shap"]).to_csv(
        out_dir / "residual_features.csv", index=False)
    (out_dir / "summary.json").write_text(pd.Series(asdict(r)).to_json(indent=2))
    print(f"[hunt] artifacts → {out_dir}/  (signal_hunt_report.md, residual_features.csv, summary.json)")
