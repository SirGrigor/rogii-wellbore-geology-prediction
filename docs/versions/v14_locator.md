# v14_locator — hypothesis check

- **Parent**: `v6s8_fast`
- **Created**: 2026-05-27T15:35:33+00:00
- **Completed**: 2026-05-27T16:42:21+00:00
- **Cloud or local**: cloud
- **Git SHA**: `e7e191c`

## Hypothesis
> A SUPERVISED sequence aligner (BiGRU + locality cross-attention to typewell∪prefix, trained on TVT) is decorrelated from the kernel's unsupervised DTW; OOF-calibrated + blended it lowers sacred below ~9.16. Final swing — else harvest.

- **Predicted improvement**: `+0.10000`
- **Actual improvement**: `-0.00425`
- **Match**: ✗ sign mismatch
- **Confidence stated**: low

## Metrics
- **OOF RMSE**: `9.17067`
- **Per-fold RMSE**: `16.00829`, `13.54788`, `16.12324`, `15.60752`, `15.65275`
- **Holdout RMSE**: `9.17067`
- **Gap holdout−oof**: `+0.00000`
- **Runtime**: `4007.5s`

## Changes from parent
**Pipeline:**
  - learned sequence locator ⊕ kernel-GBDT

## Flags
- ⚠ fold_collapse(worst=16.12324, mean=9.17067)
- ⚠ fold_instability(std=1.05234)
- ⚠ prediction_sign_mismatch(actual=-0.00425 vs pred=+0.10000)

## Human notes
- [2026-05-27T16:42:21+00:00] loc⊕kernel 9.171 vs kernel 9.171 (w=0.000, corr=0.70, loc-alone 14.154)
