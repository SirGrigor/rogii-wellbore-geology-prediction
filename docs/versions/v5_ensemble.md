# v5_ensemble — hypothesis check

- **Parent**: `v4_kernel9251`
- **Created**: 2026-05-26T18:42:42+00:00
- **Completed**: 2026-05-26T20:49:59+00:00
- **Cloud or local**: cloud
- **Git SHA**: `6406f15`

## Hypothesis
> LGB×3 + CatBoost×3(GPU) + supervised dev-OOF blend → push v4's 9.49 toward/below 9.25.

- **Predicted improvement**: `+0.30000`
- **Actual improvement**: `+0.33505`
- **Match**: ✓ matched
- **Confidence stated**: medium

## Metrics
- **OOF RMSE**: `9.15465`
- **Per-fold RMSE**: `9.44256`, `9.38062`, `9.62330`, `9.13556`, `9.14099`, `9.26331`
- **Holdout RMSE**: `9.15465`
- **Gap holdout−oof**: `+0.00000`
- **Runtime**: `7691.3s`

## Changes from parent
**Pipeline:**
  - 6-model ensemble + nm_optimize_oof blend

## Flags
(none)

## Human notes
- [2026-05-26T20:49:59+00:00] 6-model blend sacred 9.155 vs v4 9.490 vs floor 14.079
