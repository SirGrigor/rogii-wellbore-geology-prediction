# v15_squeeze — hypothesis check

- **Parent**: `v6s8_fast`
- **Created**: 2026-05-27T16:51:37+00:00
- **Completed**: 2026-05-27T17:05:02+00:00
- **Cloud or local**: cloud
- **Git SHA**: `ad54a6b`

## Hypothesis
> Negative-weight blend extracts lift from correlated members (CNN/locator) that 0-positive-weight discards; + RMSE-cage variance expansion fixes under-dispersion. Cheap, grounded in our measured failures.

- **Predicted improvement**: `+0.10000`
- **Actual improvement**: `+0.01132`
- **Match**: ⚠ off
- **Confidence stated**: low

## Metrics
- **OOF RMSE**: `9.15510`
- **Per-fold RMSE**: `9.17067`, `9.15510`, `9.19098`
- **Holdout RMSE**: `9.15510`
- **Gap holdout−oof**: `+0.00000`
- **Runtime**: `809.6s`

## Changes from parent
**Pipeline:**
  - negative blend
  - variance cage
  - adversarial

## Flags
- ⚠ prediction_undershot(actual=+0.01132 vs pred=+0.10000, ratio=0.11)
- ⚠ multiple_changes(n=3) — attribution ambiguous, consider ablation

## Human notes
- [2026-05-27T17:05:02+00:00] neg-blend 9.155 pos 9.171 cage 9.191 vs kernel 9.171 | adv AUC 0.483
