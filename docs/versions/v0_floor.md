# v0_floor — hypothesis check

- **Parent**: `—`
- **Created**: 2026-05-25T19:30:09+00:00
- **Completed**: 2026-05-25T19:30:12+00:00
- **Cloud or local**: local
- **Git SHA**: `b991c27`

## Hypothesis
> Carry-forward last-known TVT across the post-PS region is the naive anchor. Expect ~16 ft RMSE (matches Deotte starter CV~15); this is the floor that GR-alignment must beat toward the ~9.25 LB leaders.

- **Predicted improvement**: `+0.00000`
- **Actual improvement**: (no parent or no result yet)
- **Match**: —
- **Confidence stated**: high

## Metrics
- **OOF RMSE**: `15.90985`
- **Per-fold RMSE**: `17.87073`, `14.67460`, `15.64053`, `17.12786`, `13.85012`
- **Holdout RMSE**: `15.90985`
- **Gap holdout−oof**: `+0.00000`
- **Runtime**: `3.9s`

## Changes from parent
**Pipeline:**
  - carry-forward floor

## Flags
- ⚠ fold_collapse(worst=17.87073, mean=15.90985)
- ⚠ fold_instability(std=1.66881)

## Human notes
_None yet. Add with:_  `python -m src.diary flag v0_floor "..."`
