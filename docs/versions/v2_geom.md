# v2_geom — hypothesis check

- **Parent**: `v0_floor`
- **Created**: 2026-05-26T10:48:56+00:00
- **Completed**: 2026-05-26T10:51:08+00:00
- **Cloud or local**: cloud
- **Git SHA**: `407da73`

## Hypothesis
> Geometry-only residual model (no alignment) — judged on the sacred holdout.

- **Predicted improvement**: `+0.30000`
- **Actual improvement**: `+0.59026`
- **Match**: ✓ matched
- **Confidence stated**: low

## Metrics
- **OOF RMSE**: `15.31959`
- **Per-fold RMSE**: `15.24943`, `15.01064`, `15.47243`, `17.09912`, `14.87287`
- **Holdout RMSE**: `15.31959`
- **Gap holdout−oof**: `+0.00000`
- **Runtime**: `131.9s`

## Changes from parent
**Features:**
  - - align_*

## Flags
- ⚠ fold_collapse(worst=17.09912, mean=15.31959)

## Human notes
- [2026-05-26T10:51:08+00:00] sacred 15.320 vs floor 14.079 (TRAILS by 1.240)
