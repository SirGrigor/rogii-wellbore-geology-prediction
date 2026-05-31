# v22_trusted_baseline — hypothesis check

- **Parent**: `v5_ensemble`
- **Created**: 2026-05-31T20:47:03+00:00
- **Completed**: 2026-05-31T20:49:40+00:00
- **Cloud or local**: cloud
- **Git SHA**: `b41bbf1`

## Hypothesis
> Trusted residual-GBDT baseline re-established under the discovery-first lens after the LB reality-check (top-1 6.69 vs pack 9.25 => ceiling claim FALSE, 2.5ft signal extractable). Expect sacred ~14-15 (at floor); this is the honest anchor for discovery, NOT the medal attempt. Judge ONLY on sacred.

- **Predicted improvement**: `+0.00000`
- **Actual improvement**: `-6.09830`
- **Match**: ✗ sign mismatch
- **Confidence stated**: high

## Metrics
- **OOF RMSE**: `15.77358`
- **Per-fold RMSE**: `15.88589`, `15.14899`, `15.49685`, `17.35612`, `14.86130`
- **Holdout RMSE**: `15.25295`
- **Gap holdout−oof**: `-0.52063`
- **Runtime**: `474.4s`

## Changes from parent
**Pipeline:**
  - verified-API rebuild of the residual baseline; sacred-gated

## Flags
- ⚠ fold_collapse(worst=17.35612, mean=15.77358)
- ⚠ methodology_leak(|oof-holdout|=0.52063)
- ⚠ silent_regression(Δimprove=-6.09830 vs v5_ensemble)

## Human notes
_None yet. Add with:_  `python -m src.diary flag v22_trusted_baseline "..."`
