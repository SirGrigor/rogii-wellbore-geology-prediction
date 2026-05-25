# Phase-0 Reconnaissance — ROGII Wellbore Geology Prediction

**Date:** 2026-05-25 · **Source:** competition brief (`AI_wellbore_geology_prediction_task_en.pptx`),
`sample_submission.csv`, one train + one test well pair, public-kernel scan. Rules accepted
(`userHasEntered: True`).

## The task (confirmed)

Predict **True Vertical Thickness (TVT)**, in **feet**, at each 1-ft measured-depth step along a
horizontal well, **beyond the Prediction Start (PS) point**. Before PS the TVT is known (it equals
`TVT_input`); after PS it must be inferred by matching the horizontal well's **gamma-ray (GR)**
signature against the assigned vertical **type-well's** GR-vs-TVT profile (geosteering).

The brief literally describes curve alignment (slides 6–7): *"GR signature matches Typewell GR →
TVT increasing/decreasing/constant."* Slide 9: horizontal GR has higher resolution than typewell GR,
so the pre-PS horizontal GR (where TVT is known) can self-correlate the lateral. Slides 12–13:
geology **dips**, and dip is similar in **neighboring wells** — so well location/azimuth carry signal.

## Metric (CONFIRMED — slide 14)

> dTVT = manualTVT − predictedTVT for each predicted point. Quality = **RMSE of all dTVT.**

- **RMSE, feet, lower-is-better.** Public-LB leaders ≈ **9.25**. Deotte starters ≈ CV 15 → the edge
  is *technique* (alignment), not compute.
- Wired in `config.METRIC` (`MetricSpec(name="rmse", greater_is_better=False, ...)`). Thresholds are
  provisional on the ~9-ft scale until the baseline gives empirical fold variance.

## Data shape (confirmed)

| File | Rows (example) | Columns | Notes |
|---|---|---|---|
| `train/{id}__horizontal_well.csv` | 5278 | `MD,X,Y,Z,ANCC,ASTNU,ASTNL,EGFDU,EGFDL,BUDA,TVT,GR,TVT_input` | `TVT` = target; markers train-only |
| `test/{id}__horizontal_well.csv` | 5278 | `MD,X,Y,Z,GR,TVT_input` | **no markers, no TVT** |
| `train/{id}__typewell.csv` | 1296 | `TVT,GR,Geology` | `Geology` train-only |
| `test/{id}__typewell.csv` | 1296 | `TVT,GR` | |
| `sample_submission.csv` | 14151 | `id,tvt` | `id = {well_id}_{rowindex}`, rows AFTER PS only |

- **64 train wells / 64 typewells / 64 cross-section PNGs · 3+ test wells** (full count pending —
  the file API page-capped at 200; the 14,151 scored rows are the source of truth on test size).
- **Feature-availability rule (critical):** usable horizontal features = columns in BOTH splits =
  `MD,X,Y,Z,GR,TVT_input`. The 6 marker columns + typewell `Geology` are **train-only** — using them
  as model features is leakage. (They're fair game as auxiliary targets / analysis.)
- `GR` may contain NaN (brief slide 3). `TVT_input == TVT` until PS (row 0 confirmed).

## Key modeling levers (ranked, from the brief + data)

1. **Anchor + residual.** Predict `TVT = TVT_input + Δ`; model Δ only on post-PS rows. The naive
   `predict TVT = TVT_input` is the floor baseline — measure it first.
2. **GR curve alignment** (`src/align.py`): align horizontal GR ↔ typewell (GR→TVT) via DTW/DWT;
   `matched_tw_depth` is a direct TVT estimate. This is the winning public paradigm (DWT 9.251,
   "Target-Free Alignment"). **GATE any alignment feature** (synth_decoder.gate) before trusting it.
3. **Per-well sequence FE over MD** (`src/features_sequence.py`): rolling GR stats, GR gradient,
   lags/leads, normalized well progress.
4. **Trajectory / spatial features:** azimuth + inclination from `X,Y,Z` diffs; dip direction;
   neighboring-well context (geology dips correlate across nearby wells — slides 12–13).
5. **Pre-PS self-correlation:** use the well's own known pre-PS GR↔TVT as a per-well calibration.

## CV strategy

- **GroupKFold by well** (`src/cv.py`) — never split a well across folds. Only 64 wells → fold
  variance will be high; expect to need 5–10 folds and to watch `fold_instability`.
- Mirror the public/private structure: scored region = post-PS lateral only. The local metric must
  be computed **only on post-PS rows**, matching the submission.

## Compute-parity audit (L15)

| Author | Kernel | Signal |
|---|---|---|
| **Chris Deotte** (GM) | XGB Starter CV 15 · NN Starter CV 15.5 · EDA Starter | competing, but starters are weak (CV 15) → alignment beats raw GBDT |
| parthenos | DWT-based — **LB 9.251** (446 votes) | the top public technique = wavelet curve alignment |
| Pilkwang Kim | Target-Free Alignment for TVT (fresh 2026-05-25) | alignment without using the target |
| Roman Tamrazov | "SUPER SOLUTION TOP 3" · "BETTER 9.956" | top-tier blends |
| Mahdi Ravaghi | Hill Climbing · LightGBM | GBDT + ensembling baselines |
| Karnakbayev Artur | Physics-Informed Baseline | physics/geometry prior |

**Read:** Featured medals are top 5–10% (≈ rank 82–164 of 1638), not top-5. GMs at the top don't
block a medal. The bottleneck is the alignment technique — which is squarely our `align.py` track.
Portfolio-wise this is a NEW case type (curve-alignment regression / geosteering) vs S6E5 tabular
blending — strong T-shape value.

## Tiered medal plan

- **Tier 0 — floor (local, this/next session):** EDA on all 64 wells (PS detection, GR NaN rate,
  TVT vs TVT_input divergence, per-well length, spatial map). Baseline = `TVT = TVT_input`; then a
  per-well DTW-alignment-only prediction. Establish honest GroupKFold RMSE + LB calibration.
- **Tier 1 — alignment core (Colab if needed):** productionize `align.py` (DTW + a DWT variant),
  GATE the alignment features, feed `matched_tw_depth` + `local_shift` + GR-sequence FE + trajectory
  features into LGB/XGB residual models on Δ = TVT − TVT_input. Target ≤ ~10–11 RMSE (bronze zone).
- **Tier 2 — spatial + multi-curve (medal push):** neighboring-well/dip features, pre-PS
  self-calibration, multi-resolution alignment, model diversity (LGB/XGB/Cat + a sequence NN on
  Colab). Supervised blend on labeled GroupKFold OOF (feedback_supervised-blend-not-lb-probing).
  Target the silver band (≈ rank 82, RMSE near the 9.2 leaders).
- **Discipline every step:** `Experiment.start(hypothesis, predicted_delta)` BEFORE training;
  per-fold RMSE mandatory; gate recovered structure.

## Open items for next session

- [ ] Confirm full **test well count** (paginate the file API or derive from the 14,151 ids).
- [ ] Confirm how the submission `id` index maps to horizontal-well rows (iloc? an index column?).
- [ ] Measure the **floor**: RMSE of `predict TVT = TVT_input` on post-PS rows (local + LB).
- [ ] Read Deotte's EDA Starter + the DWT-based kernel for PS-detection + alignment specifics.
- [ ] Tune `MetricSpec` thresholds once baseline fold variance is known.
