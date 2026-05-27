# v12_spatialdip — hypothesis check

- **Parent**: `v6s8_fast`
- **Created**: 2026-05-27T14:14:46+00:00
- **Completed**: 2026-05-27T14:21:24+00:00
- **Cloud or local**: cloud
- **Git SHA**: `2a766f1`

## Hypothesis
> Frame-independent cross-well dip (neighbours' TVT-plane gradient → drift extrapolation) adds signal the per-well 222 miss, esp. at long horizon where self-alignment degrades (v11: error 3.7→12 by horizon; near-neighbour wells 5.54 vs 7.50).

- **Predicted improvement**: `+0.10000`
- **Actual improvement**: `-0.00425`
- **Match**: ✗ sign mismatch
- **Confidence stated**: low

## Metrics
- **OOF RMSE**: `9.17067`
- **Per-fold RMSE**: `9.17067`, `9.17792`
- **Holdout RMSE**: `9.17067`
- **Gap holdout−oof**: `+0.00000`
- **Runtime**: `403.0s`

## Changes from parent
**Pipeline:**
  - cross-well dip field

## Flags
- ⚠ prediction_sign_mismatch(actual=-0.00425 vs pred=+0.10000)

## Human notes
- [2026-05-27T14:21:24+00:00] cross-well dip: sacred 9.171→9.178 (Δ+0.007), far Δ-0.005
