# Toolkit port backlog

The reusable competition tooling was built across S6E3/S6E4/S6E5. Rather than copy-paste
it again (the drift problem), we are **consolidating the generic core into shared packages**
and porting the model-class-specific pieces *on first use*, once Phase-0 confirms the metric.

## Done — promoted to `kaggle-playground-utils` v0.2.0 (2026-05-25)

| Tool | Status | Notes |
|---|---|---|
| `observer.py` (experiment diary) | ✅ generalized | Score-neutral fields + `MetricSpec` (direction + thresholds). Regression-direction autoflags verified. |
| `diary.py` (render/CLI) | ✅ generalized | Metric label + lower/higher-is-better aware. |
| `viz.py` (ρ-matrix, score-rho scatter, fold boxplot) | ✅ generalized | Family from model dict; scorer callable in loader. |
| `probes.py` (MI, adversarial) | ✅ already shared | Supports `mutual_info_regression` — use directly in recon. |
| `encoding.py` (leakage-free TE) | ✅ already shared | Use for categorical curve-type / well-meta features. |

## Reused as dependency

| Tool | Source | Applies to rogii? |
|---|---|---|
| `synth_decoder.gate` (the GATE) | synth-decoder | ✅ Discipline applies — gate any recovered structure before trusting it. |
| `synth_decoder.adversarial` | synth-decoder | ✅ Train/test shift diagnostic. |
| `synth_decoder.matching` / `fingerprint` | synth-decoder | ❌ Assume synthetic generator. rogii is REAL data — skip. |

## To port — at the start of the MODELING phase (after metric confirmed)

These hardcode binary classification (`objective="binary"`, `roc_auc_score`). Porting them
*before* the metric/direction is confirmed would be wasted work, so they wait for Phase-0.

| Tool | S6E5 source | Port spec |
|---|---|---|
| **Variant factory** | `playground-s6e5/src/train.py` | Parametrize objective + eval: LGB `regression`/`regression_l1`, XGB `reg:squarederror`, CatBoost `RMSE`. Replace `roc_auc_score` with the confirmed regression scorer. Add target transform hook (Yeo-Johnson — S5E9 toolkit). Promote to `kaggle-playground-utils.train` with a `Task` enum (clf/reg). |
| **Blend math** | `playground-s6e5/src/blend_math.py` | `nm_optimize_holdout` + `rank_normalize` are reusable; pass a `scorer` + `greater_is_better` instead of hardcoded AUC. Keep supervised-on-labeled-holdout discipline (feedback_supervised-blend-not-lb-probing); the quadratic-LB-fit helper is LB-probing — keep but de-emphasize. |
| **Paradigm radar** | `playground-s6e5/notebooks/_viz_paradigm_radar.py` | Coverage map over experiment axes. Generic once axes are redefined for a regression/sequence comp (FE-depth, alignment, model class, target transform, CV scheme, ...). |

## NEW for this competition (no prior art — build in `src/`)

| Need | Where | Note |
|---|---|---|
| Type-well ↔ horizontal **curve alignment** (DTW) | `src/align.py` | The genuinely new skill vs. S6E5's lap-series. Likely the main signal. |
| Depth-**sequence** FE (rolling/lag/gradient over measured depth) | `src/features_sequence.py` | Analogous to S6E5 lap-series FE but over depth, grouped per well. |
| Per-well **GroupKFold** CV | `src/cv.py` | Never split a single well across folds (leakage). |
