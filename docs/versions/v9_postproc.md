# v9_postproc — hypothesis check

- **Parent**: `v6s8_fast`
- **Created**: 2026-05-27T11:44:51+00:00
- **Completed**: 2026-05-27T11:51:17+00:00
- **Cloud or local**: cloud
- **Git SHA**: `92628a8`

## Hypothesis
> border_count=254 (finer splits) + per-well savgol smoothing of the drift curve (window tuned on dev-OOF) + clip lower sacred below the ~9.16 GBDT floor.

- **Predicted improvement**: `+0.15000`
- **Actual improvement**: `+0.00054`
- **Match**: ⚠ off
- **Confidence stated**: medium

## Metrics
- **OOF RMSE**: `9.16587`
- **Per-fold RMSE**: `9.17067`, `9.16587`, `9.16587`
- **Holdout RMSE**: `9.16587`
- **Gap holdout−oof**: `+0.00000`
- **Runtime**: `392.6s`

## Changes from parent
**Pipeline:**
  - border_count=254/depth7
  - per-well savgol
  - clip

## Flags
- ⚠ prediction_undershot(actual=+0.00054 vs pred=+0.15000, ratio=0.00)
- ⚠ multiple_changes(n=3) — attribution ambiguous, consider ablation

## Human notes
- [2026-05-27T11:51:17+00:00] border254 Δ+0.026 + savgol(w=61) Δ-0.005: sacred 9.171→9.166
