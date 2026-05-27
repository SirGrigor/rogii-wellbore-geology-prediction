# v8_feats — hypothesis check

- **Parent**: `v6s8_fast`
- **Created**: 2026-05-27T11:09:34+00:00
- **Completed**: 2026-05-27T11:22:14+00:00
- **Cloud or local**: cloud
- **Git SHA**: `9339c34`

## Hypothesis
> Horizon+geometry (▲Z since PS, md-since-PS, inclination) and GR-texture (gradient, curvature, roughness) carry signal the 222 alignment feats miss → sacred < the ~9.16 floor.

- **Predicted improvement**: `+0.10000`
- **Actual improvement**: `+0.00686`
- **Match**: ⚠ off
- **Confidence stated**: low

## Metrics
- **OOF RMSE**: `9.15956`
- **Per-fold RMSE**: `9.17067`, `9.15956`, `9.15956`, `9.17067`
- **Holdout RMSE**: `9.15956`
- **Gap holdout−oof**: `+0.00000`
- **Runtime**: `803.3s`

## Changes from parent
**Pipeline:**
  - +W1 horizon/geom
  - +W2 GR texture

## Flags
- ⚠ prediction_undershot(actual=+0.00686 vs pred=+0.10000, ratio=0.07)
- ⚠ multiple_changes(n=2) — attribution ambiguous, consider ablation

## Human notes
- [2026-05-27T11:22:14+00:00] feat ablation: base 9.171 | +W1 9.160 | +W1W2 9.160 | +W2only 9.171
