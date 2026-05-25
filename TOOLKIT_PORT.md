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

## BUILT in rogii/src (2026-05-25) — regression-shaped; candidates to promote to utils later

The shared utils `train.py`/`blend.py` are classification + Registry-coupled (S6E4 lineage);
retrofitting them risked destabilizing the shared toolkit, so we built regression versions in
rogii/src. Promote to `kaggle-playground-utils` (with a `Task` enum clf/reg) once proven on rogii.

| Tool | rogii/src | Notes |
|---|---|---|
| **Variant factory** | `src/train.py` `train_variant(algo, ...)` | LGB/XGB/CatBoost regression, GroupKFold-by-well, per-fold RMSE, artifacts to `probs/<v>/`, `fit_full` saves the deployable model for the Kaggle inference notebook. |
| **Supervised blend** | `src/blend.py` `nm_optimize_oof` | RMSE objective, `allow_negative`, **optimize + SELECT on GroupKFold OOF** (L48/L53). `rank_normalize` + `marginal_report` (decorrelation-necessary-not-sufficient). No Registry coupling. |

## Still to port — when needed

| Tool | S6E5 source | Port spec |
|---|---|---|
| **Paradigm radar** | `playground-s6e5/notebooks/_viz_paradigm_radar.py` | Coverage map over experiment axes. Redefine axes for this comp (FE-depth, alignment recipe, model class, target transform, CV scheme, spatial). Port when ≥5 models exist. |
| target transform | S5E9 (Yeo-Johnson `TransformedTargetRegressor`) | Add to `train_variant` as an optional hook if the drift residual is skewed. |

## NEW for this competition (no prior art — build in `src/`)

| Need | Where | Note |
|---|---|---|
| Type-well ↔ horizontal **curve alignment** (DTW) | `src/align.py` | The genuinely new skill vs. S6E5's lap-series. Likely the main signal. |
| Depth-**sequence** FE (rolling/lag/gradient over measured depth) | `src/features_sequence.py` | Analogous to S6E5 lap-series FE but over depth, grouped per well. |
| Per-well **GroupKFold** CV | `src/cv.py` | Never split a single well across folds (leakage). |
