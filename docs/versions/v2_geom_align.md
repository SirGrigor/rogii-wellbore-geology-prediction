# v2_geom_align — hypothesis check

- **Parent**: `v2_geom`
- **Created**: 2026-05-26T10:51:08+00:00
- **Completed**: 2026-05-26T10:53:41+00:00
- **Cloud or local**: cloud
- **Git SHA**: `407da73`

## Hypothesis
> Add naive-DTW alignment back on top of geometry — judged on the sacred holdout.

- **Predicted improvement**: `+0.30000`
- **Actual improvement**: `+0.14401`
- **Match**: ⚠ off
- **Confidence stated**: low

## Metrics
- **OOF RMSE**: `15.17557`
- **Per-fold RMSE**: `15.45479`, `14.89306`, `15.55268`, `17.21755`, `14.80624`
- **Holdout RMSE**: `15.17557`
- **Gap holdout−oof**: `+0.00000`
- **Runtime**: `153.3s`

## Changes from parent
**Features:**
  - + align_*

## Flags
- ⚠ fold_collapse(worst=17.21755, mean=15.17557)
- ⚠ prediction_undershot(actual=+0.14401 vs pred=+0.30000, ratio=0.48)

## Human notes
- [2026-05-26T10:53:41+00:00] sacred 15.176 vs floor 14.079 (TRAILS by 1.096)
