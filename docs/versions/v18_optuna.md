# v18_optuna — hypothesis check

- **Parent**: `v16_max`
- **Created**: 2026-05-27T20:14:59+00:00
- **Completed**: 2026-05-27T21:48:44+00:00
- **Cloud or local**: cloud
- **Git SHA**: `abae453`

## Hypothesis
> Optuna HPO on CatBoost (the audit's missing depth lever — we never tuned) pushes the deep stack past the un-tuned 9.100, on dev-OOF (never sacred).

- **Predicted improvement**: `+0.05000`
- **Actual improvement**: (no parent or no result yet)
- **Match**: —
- **Confidence stated**: medium

## Metrics
- **OOF RMSE**: `9.14219`
- **Per-fold RMSE**: `9.18722`, `9.23108`, `9.27427`, `9.32240`, `9.56438`
- **Holdout RMSE**: `9.14219`
- **Gap holdout−oof**: `+0.00000`
- **Runtime**: `5668.3s`

## Changes from parent
**Pipeline:**
  - Optuna CatBoost HPO 30 trials

## Flags
(none)

## Human notes
- [2026-05-27T21:48:44+00:00] Optuna cat → stack 9.142 vs max 9.100 (best params {'depth': 6, 'learning_rate': 0.02538727325623757, 'l2_leaf_reg': 2.0825545257757114, 'min_data_in_leaf': 16, 'border_count': 254, 'random_strength': 1.6962866591697565, 'bagging_temperature': 0.711591672037549})
