# Strategy — carried forward from S6E5 (+ the 9-comp winning-technique survey)

The S6E5 competition produced a large body of hard-won strategy (lessons L23–L53, KG nodes
`2026-23_s6e5-paradigm-sprint-postmortem` + `2026-25_winning-technique-survey`). This file distills
the **transferable** parts for ROGII and flags what does NOT transfer. Read before each tier.

## 0. Compute: cloud-first, always

**ALL model fits run on Colab — including quick sanity fits. No `.fit()` / `train_variant` on the
local machine, ever** (reinforced 2026-05-25). Verify pipelines by shapes/dtypes/imports/feature-
parity, NOT by training. `notebooks/03_baseline_lgb.py` and later training run via
`colab_runner.ipynb`. Local is for: EDA, feature assembly, DTW alignment, blend math (scipy on
arrays, not a model), GATE checks, submission assembly, tests (the one fit-test is gated behind
`ROGII_RUN_FIT_TESTS=1`). The runner installs `kaggle-playground-utils` from GitHub; `synth-decoder`
GATE checks stay local. (feedback_cloud-first-compute.)

## 1. Experiment methodology

- **Diary discipline is mandatory** (feedback_experiment-diary-required). `Experiment.start(hypothesis,
  predicted_delta, confidence)` BEFORE any training; `record()` per-fold; `commit()` runs the 7
  autoflags. Per-fold RMSE is mandatory output — **mean RMSE hides fold collapse** (S6E5 Phase-12).
- **L47 — lead with FE-depth and blend-method, NOT model-architecture breadth.** S6E5's biggest
  mistake: we tried 7 architectures chasing a wall that FE-depth + blending owned. Don't conflate
  "tried many models" with "explored the winnable space." For ROGII the FE-depth IS the alignment.
- **Paradigm radar before choosing what to try** — map coverage across axes (FE-depth, alignment
  method, model class, target transform, CV scheme, spatial features) so you attack gaps, not
  whatever's nearest. (`playground-s6e5/notebooks/_viz_paradigm_radar.py` — port when ≥5 models exist.)
- **Adversarial validation EARLY** (`synth_decoder.adversarial`, or `kpu.probes`): train-vs-test
  classifier. AUC≈0.5 → i.i.d., trust GroupKFold; AUC>0.6 → distribution shift, a real lever.
  Settles "is my CV trustworthy" with ~20 lines before investing.

## 2. Blending discipline (the most valuable, most error-prone lever)

The S6E5 endgame was almost entirely about blending. The lessons, ranked by how much they cost us:

- **L53 / feedback_supervised-blend-not-lb-probing — optimize blends on a LABELED CV signal, never by
  probing the public LB.** For ROGII this is *easier* than S6E5: we have full TVT labels on all 64
  train wells, so the labeled signal is just the **GroupKFold OOF**. Index every candidate, score each
  on OOF RMSE, NM-optimize the blend on OOF, report marginal contribution + direction. Submit only the
  OOF-winner. (S6E5 had to leak-match test rows to get a labeled signal; we don't.)
- **L48 — negative weights overfit when tuned on holdout. Optimize negative-weight blends on OOF
  (leak-free), and SELECT the final submission by the OOF score, never by holdout.** Negative weights
  have more freedom to fit noise; holdout-opt → LB collapse (S6E5 v51: holdout 0.954 → LB 0.952).
  Port `blend_math.nm_optimize_holdout` as `nm_optimize_oof` with an RMSE objective + `allow_negative`.
- **Exact-metric optimization.** Optimize the blend on **RMSE directly** (not a proxy). 4/6 surveyed
  comps won with exact-metric hill-climbing.
- **L41 / L50 / L52 — decorrelation is necessary but NOT sufficient.** A blend lifts only with a
  member that is BOTH decorrelated (low ρ) AND ≥ the anchor's quality. A weaker decorrelated model
  drags the blend down (monotonically). The "winnable zone" = high-quality + low-ρ (`kpu.viz.
  score_rho_scatter` visualizes it). You cannot merge upward from a single best solution.
- **rank-normalize before blending** when members have different scales (`blend_math.rank_normalize`).
  rank_max peer-merge only helps with narrow-band support (ρ∈[0.9995,0.9997]) — rare; don't assume it.

## 3. Regression toolkit (from S5E9 — directly applies; ROGII is RMSE)

- **Anchor + residual (VERIFIED necessary).** `TVT_input` is NaN post-PS, so the anchor is the
  **last-known TVT** at PS. **Model the drift Δ = TVT − anchor, never absolute TVT** — a 40-well
  pipeline check put absolute-TVT LGB at ~113 ft RMSE (trees can't extrapolate the 9.2k–12.9k ft
  level across wells) vs the ~12 ft floor. `features.build_dataset(target="residual")` does this by
  default and returns `anchor`; de-residualize with `tvt_pred = drift_pred + anchor`. Floor
  (carry-forward, Δ=0) = 15.91 ft — the baseline to beat.
- **Target transform** (`TransformedTargetRegressor` + Yeo-Johnson/PowerTransformer) on Δ if its
  distribution is skewed; always invert for the metric.
- **BayesianRidge meta-learner** for stacking OOF predictions (S5E9 winner's meta).
- **CenteredIsotonicRegression / isotonic calibration** of predictions before/after blend.
- **"RMSE cage"** variance-expansion post-proc (tail-stretch around a pivot) — a cheap final lever.

## 4. Feature-engineering depth — what transfers vs not

**Transfers (real-data sequence FE):**
- **"Encode the data's TRUE structure as explicit features"** (the single biggest edge, 5/6 comps).
  For ROGII that structure is geological: the **DTW GR-alignment** (`src/align.py` → `matched_tw_depth`,
  `local_shift`), geological **dip** (slides 12–13), and **neighboring-well** context.
- **Deviation / 2nd-derivative sequence features (L46 caveat, Amex).** Basic lags + short rolling were
  redundant on S6E5 (L46) — but `last_minus_mean` (GR minus the well's expanding-mean GR = anomaly),
  `last − first`, and diff-of-diffs (GR-gradient acceleration) are DIFFERENT quantities and were never
  fairly tested. Compute over MD with an **expanding** (not full-well) window to avoid leak.
- **Null-importance feature selection** (Amex): target-shuffle null per feature, keep only features
  beating their null. CV-safe pruner once the feature set explodes.
- **GATE everything** (`synth_decoder.gate`): a recovered structure (e.g. an alignment-derived TVT)
  is worth using only if it beats the model on the rows it affects. Catches the "accuracy illusion."

**Does NOT transfer (synthetic-only):**
- Digit extraction / quantization fingerprints / generator-formula recovery — ROGII is **real data**,
  no synthetic generator to reverse. `synth_decoder.matching` / `fingerprint` do not apply.

## 5. The "don'ts" that cost S6E5 the most

1. Don't chase architecture diversity while FE-depth + blending are unexplored (L47).
2. Don't tune negative-weight blends on holdout, and don't select submissions by holdout (L48).
3. Don't trust mean-only metrics — log per-fold; auto-flag fold collapse (Phase-12).
4. Don't expect to merge upward from a single strong solution (L52).
5. Don't force real-data tricks onto synthetic comps — or vice-versa (L49 rejections list).

## Tier mapping (see `docs/reconnaissance.md` for the competition specifics)
- **Tier 0:** anchor floor (`TVT=TVT_input`) + pure DTW alignment; honest GroupKFold OOF RMSE; adversarial validation.
- **Tier 1:** residual GBDT on alignment + GR-sequence + trajectory features; GATE each; LGB/XGB.
- **Tier 2:** spatial/dip + neighbor features, multi-resolution alignment, model diversity (+ Colab NN),
  supervised OOF blend (§2), regression post-proc (§3). Silver band.
