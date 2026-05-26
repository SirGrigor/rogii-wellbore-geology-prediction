# v3_selfcorr — hypothesis check

- **Parent**: `v3_noselfcorr`
- **Created**: 2026-05-26T11:31:11+00:00
- **Completed**: 2026-05-26T11:33:54+00:00
- **Cloud or local**: cloud
- **Git SHA**: `7cdc28a`

## Hypothesis
> Add known-prefix self-correlation features (P1) — judged on sacred.

- **Predicted improvement**: `+0.30000`
- **Actual improvement**: `-0.07738`
- **Match**: ✗ sign mismatch
- **Confidence stated**: low

## Metrics
- **OOF RMSE**: `15.25295`
- **Per-fold RMSE**: `15.88589`, `15.14899`, `15.49685`, `17.35612`, `14.86130`
- **Holdout RMSE**: `15.25295`
- **Gap holdout−oof**: `+0.00000`
- **Runtime**: `163.6s`

## Changes from parent
**Features:**
  - + selfcorr_*

## Flags
- ⚠ fold_collapse(worst=17.35612, mean=15.25295)
- ⚠ prediction_sign_mismatch(actual=-0.07738 vs pred=+0.30000)

## Human notes
- [2026-05-26T11:33:54+00:00] sacred 15.253 vs floor 14.079
