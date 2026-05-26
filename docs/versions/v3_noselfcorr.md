# v3_noselfcorr — hypothesis check

- **Parent**: `v0_floor`
- **Created**: 2026-05-26T11:28:17+00:00
- **Completed**: 2026-05-26T11:31:11+00:00
- **Cloud or local**: cloud
- **Git SHA**: `7cdc28a`

## Hypothesis
> Geometry + naive-align, NO self-correlation — judged on sacred.

- **Predicted improvement**: `+0.30000`
- **Actual improvement**: `+0.73428`
- **Match**: ⚠ off
- **Confidence stated**: low

## Metrics
- **OOF RMSE**: `15.17557`
- **Per-fold RMSE**: `15.45479`, `14.89306`, `15.55268`, `17.21755`, `14.80624`
- **Holdout RMSE**: `15.17557`
- **Gap holdout−oof**: `+0.00000`
- **Runtime**: `173.4s`

## Changes from parent
**Features:**
  - - selfcorr_*

## Flags
- ⚠ fold_collapse(worst=17.21755, mean=15.17557)
- ⚠ prediction_overshot(actual=+0.73428 vs pred=+0.30000, ratio=2.45)

## Human notes
- [2026-05-26T11:31:11+00:00] sacred 15.176 vs floor 14.079
