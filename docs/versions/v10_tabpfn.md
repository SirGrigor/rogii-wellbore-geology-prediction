# v10_tabpfn — hypothesis check

- **Parent**: `v6s8_fast`
- **Created**: 2026-05-27T13:34:23+00:00
- **Completed**: 2026-05-27T13:40:15+00:00
- **Cloud or local**: cloud
- **Git SHA**: `7898fe1`

## Hypothesis
> TabPFN (in-context tabular foundation model) is a different function family from GBDT → decorrelated residuals → TabPFN⊕GBDT beats GBDT-alone on sacred, even though TabPFN-alone (10K context vs 3M rows) is weaker. Last evidence-backed swing before harvest.

- **Predicted improvement**: `+0.10000`
- **Actual improvement**: `+0.08579`
- **Match**: ✓ matched
- **Confidence stated**: low

## Metrics
- **OOF RMSE**: `9.08063`
- **Per-fold RMSE**: `10.92601`, `9.08063`, `9.08063`
- **Holdout RMSE**: `9.08063`
- **Gap holdout−oof**: `+0.00000`
- **Runtime**: `360.9s`

## Changes from parent
**Pipeline:**
  - TabPFN stack member

## Flags
- ⚠ fold_collapse(worst=10.92601, mean=9.08063)
- ⚠ fold_instability(std=1.06543)

## Human notes
- [2026-05-27T13:40:15+00:00] TabPFN⊕GBDT 9.081 vs GBDT-alone 9.081 (w=0.000, corr=0.87)
