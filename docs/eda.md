# Tier-0 EDA findings (2026-05-25)

Source: `notebooks/01_eda.py` over the full downloaded data (`reports/eda_per_well.csv`).
**These numbers supersede the provisional counts in `reconnaissance.md`** (which were read off a
page-capped file listing).

## Scale (corrected)
- **773 train wells** (not 64 — the file API page-capped at 200). **3 test wells.**
- Train: ~6,588 rows/well avg (min 2,058, max 12,141) → ~5.1M total rows; **3.78M post-PS rows**.
- Test: 3 wells, **14,151 post-PS rows** = the sample_submission size.

## PS point & target structure (corrected — important)
- `TVT_input` is the true TVT on the **known prefix** and **NaN after the PS point** (NOT a
  carry-forward, as first assumed). PS = `df["TVT_input"].notna().sum()`.
- PS at ~row 1,692 avg; the **post-PS scored region is ~73% of each well** (min 20%, max 88%).
- Submission `id = {well}_{row_index}` uses the **0-based iloc**; first scored id (`..._1442`) =
  the PS row. Verified: the floor submission mapped **0 unmapped of 14,151**.
- Post-PS TVT drifts substantially from the last-known value: median max-drift 20.5 ft, max 103.8 ft
  — i.e. the well climbs/descends through geology, which is exactly what alignment must recover.
- TVT range across wells: **[9,245, 12,894] ft**.

## GR (the alignment signal) — heavily missing
- **Every well has GR NaNs; mean NaN fraction ≈ 29%** (range 0.6%–80%). NaN handling (interpolate /
  mask) is a first-class concern for any GR-alignment or GR-sequence feature, not an afterthought.

## Floor baseline (anchor) — `v0_floor`
- **Carry-forward last-known TVT across the post-PS region → overall RMSE 15.91 ft.**
  5-fold-by-well: [17.9, 14.7, 15.6, 17.1, 13.9] (mean ~15.8, std ~1.6 → the diary's
  `fold_instability`/`fold_collapse` flags fired; thresholds are provisional for a deterministic floor).
- This **matches Deotte's starter CV ~15** → confirms our understanding of the task and metric.
  The LB leaders are ~9.25, so **GR-alignment is the ~6.7 ft lever** to close.
- Submission `submissions/v0_floor.csv` built + diary-logged. (LB calibration pending — see below.)

## Spatial
- Wells span X [2.86M, 3.04M], Y [1.01M, 1.14M] (feet). Dense field → neighboring-well / dip
  features are viable (brief slides 12–13).

## Submission mechanism — OPEN (blocker for LB calibration)
- `kaggle competitions submit` (CLI v2.0.1) returned **400 on CreateSubmission**. Combined with all
  public "solutions" being Kaggle **notebooks**, this is most likely a **kernels-only Code
  competition** → submit by running a notebook on Kaggle that emits `submission.csv`, not CSV upload.
- **Workflow implication:** train on Colab (save model artifacts), run a thin **Kaggle inference
  notebook** that loads artifacts + emits the submission. Confirm the submission type before Tier 1.
  (Could also be the known CLI 2.0.1 bug — upgrading the CLI or submitting via the website would test.)

## Implications for modeling
1. **Anchor = last-known TVT (carry-forward), not `TVT_input`** (NaN post-PS). Model the **drift**
   from the last-known value, OR predict absolute TVT, gated against the 15.91 floor.
2. **GR NaN ≈ 29%** — robust NaN handling before alignment/FE.
3. **773 wells** → GroupKFold variance is fine (the earlier "only 64 wells" concern was wrong).
4. The pre-PS known GR↔TVT (the prefix) is per-well calibration data — use it to anchor the alignment.
