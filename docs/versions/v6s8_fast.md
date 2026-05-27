# v6s8_fast — hypothesis check

- **Parent**: `v5_ensemble`
- **Created**: 2026-05-27T06:24:13+00:00
- **Completed**: 2026-05-27T07:16:24+00:00
- **Cloud or local**: cloud
- **Git SHA**: `b0e2126`

## Hypothesis
> FAST faithfulness probe: low-capacity profile (LGB leaves 63 / cat depth 6) at stride 8 vs v5_ensemble (255/depth7, stride-8 → sacred 9.155). Does the fast proxy track the quality config (rank-order/gap), and does 255 earn its cost or just overfit?

- **Predicted improvement**: `+0.00000`
- **Actual improvement**: `-0.01176`
- **Match**: ✗ sign mismatch
- **Confidence stated**: low

## Metrics
- **OOF RMSE**: `9.16642`
- **Per-fold RMSE**: `9.54192`, `9.42074`, `9.58188`, `9.10503`, `9.16836`, `9.21064`
- **Holdout RMSE**: `9.16642`
- **Gap holdout−oof**: `+0.00000`
- **Runtime**: `3184.8s`

## Changes from parent
**Pipeline:**
  - LGB leaves=63, cat depth=6, stride 8

## Flags
(none)

## Human notes
- [2026-05-27T07:16:24+00:00] v6s8_fast: stride 8 6-model blend sacred 9.166 vs v5(stride8) 9.155 vs floor 14.079
