# v13_boosters — hypothesis check

- **Parent**: `v6s8_fast`
- **Created**: 2026-05-27T14:49:32+00:00
- **Completed**: 2026-05-27T15:13:37+00:00
- **Cloud or local**: cloud
- **Git SHA**: `f6b1f21`

## Hypothesis
> A more diverse booster set (lgb/cat/xgb/HistGB) gives a decorrelated blend member that lowers sacred below the lgb+cat ~9.16 — marginal, within-paradigm.

- **Predicted improvement**: `+0.05000`
- **Actual improvement**: `-0.00923`
- **Match**: ✗ sign mismatch
- **Confidence stated**: low

## Metrics
- **OOF RMSE**: `9.17564`
- **Per-fold RMSE**: `9.36835`, `9.17921`, `9.30630`, `9.44157`
- **Holdout RMSE**: `9.17564`
- **Gap holdout−oof**: `+0.00000`
- **Runtime**: `1452.2s`

## Changes from parent
**Pipeline:**
  - booster survey + diverse blend

## Flags
- ⚠ prediction_sign_mismatch(actual=-0.00923 vs pred=+0.05000)

## Human notes
- [2026-05-27T15:13:37+00:00] boosters: best single cat 9.179; lgb+cat 9.191 → all-4 9.176
