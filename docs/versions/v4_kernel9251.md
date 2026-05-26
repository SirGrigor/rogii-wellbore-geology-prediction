# v4_kernel9251 — hypothesis check

- **Parent**: `v0_floor`
- **Created**: 2026-05-26T17:55:43+00:00
- **Completed**: 2026-05-26T18:28:37+00:00
- **Cloud or local**: cloud
- **Git SHA**: `e590109`

## Hypothesis
> Ported 9.251 feature engine (PF/DTW/beam/NCC/affine/spatial-imputers) + single LGB. Expect to clear the floor decisively and approach ~10-11 (ensemble+blend reach 9.25).

- **Predicted improvement**: `+3.00000`
- **Actual improvement**: `+6.42015`
- **Match**: ⚠ off
- **Confidence stated**: medium

## Metrics
- **OOF RMSE**: `9.48970`
- **Per-fold RMSE**: `9.99188`, `12.49968`, `12.89365`, `10.59905`, `12.01088`
- **Holdout RMSE**: `9.48970`
- **Gap holdout−oof**: `+0.00000`
- **Runtime**: `2009.7s`

## Changes from parent
**Features:**
  - + full 9.251 build_well features
**Pipeline:**
  - port baseline (lgb)

## Flags
- ⚠ fold_collapse(worst=12.89365, mean=9.48970)
- ⚠ fold_instability(std=1.24897)
- ⚠ prediction_overshot(actual=+6.42015 vs pred=+3.00000, ratio=2.14)
- ⚠ multiple_changes(n=2) — attribution ambiguous, consider ablation

## Human notes
- [2026-05-26T18:28:37+00:00] ported 9.251 features; sacred 9.490 vs floor 14.079 (BEATS)
