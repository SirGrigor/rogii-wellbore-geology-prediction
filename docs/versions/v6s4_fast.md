# v6s4_fast — hypothesis check

- **Parent**: `v5_ensemble`
- **Created**: 2026-05-27T07:55:26+00:00
- **Completed**: 2026-05-27T08:09:52+00:00
- **Cloud or local**: cloud
- **Git SHA**: `0984948`

## Hypothesis
> FAST faithfulness probe: low-capacity profile (LGB leaves 63 / cat depth 6) at stride 4 vs v5_ensemble (255/depth7, stride-8 → sacred 9.155). Does the fast proxy track the quality config (rank-order/gap), and does 255 earn its cost or just overfit?

- **Predicted improvement**: `+0.00000`
- **Actual improvement**: `-0.03711`
- **Match**: ✗ sign mismatch
- **Confidence stated**: low

## Metrics
- **OOF RMSE**: `9.19176`
- **Per-fold RMSE**: `9.52307`, `9.11699`, `9.28730`, `9.19277`
- **Holdout RMSE**: `9.19176`
- **Gap holdout−oof**: `+0.00000`
- **Runtime**: `920.8s`

## Changes from parent
**Pipeline:**
  - LGB leaves=63, cat depth=6, stride 4

## Flags
(none)

## Human notes
- [2026-05-27T08:09:52+00:00] v6s4_fast: stride 4 6-model blend sacred 9.192 vs v5(stride8) 9.155 vs floor 14.079
