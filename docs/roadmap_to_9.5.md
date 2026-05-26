# Measurable approach vector → ~9.5 RMSE

A tracked descent from the floor to medal range, grounded in the data + the two top public
kernels (DWT-based **9.251**, target-free). Replaces ad-hoc "build-a-feature-and-hope" with
milestones, expected vs measured deltas, and decision gates. **The sacred 150-well holdout is
the only number we trust** (dev OOF overfits; the 3-well LB is noise).

## Anchors (measured)
| reference | sacred RMSE | note |
|---|---|---|
| carry-forward floor (Δ=0) | **14.08** | the bar to beat |
| Deotte GBDT starter | ~15 (CV) | ≈ our geometry+align models — naive-GBDT level |
| our best so far (v1 lgb) | 14.45 | still above floor (overfits) |
| **top public ensemble** | **9.25** | the full recipe below |
| **our target** | **~9.5** | medal band (silver ≈ rank 82 / 1638) |

Gap to close: **14.08 → 9.5 ≈ 4.6 ft**, distributed across the recipe's components (no single
one does it — proven: naive DTW 113, Viterbi 47, self-corr 20, all > floor).

## Measurement protocol (what makes this "measurable")
1. **Fixed harness:** sacred_split (150 wells, seed-locked) + one **regularized LGB** model
   (lgb generalizes ~0.7 ft better than xgb here — generalization > GPU speed). Same model across
   experiments so component deltas are comparable.
2. **Metric per experiment:** sacred RMSE, Δ-vs-floor, **Δ-vs-previous** (the marginal contribution).
3. **Diary = scoreboard:** every component logged via the observer with a *predicted* delta first;
   `experiments.jsonl` (auto-pushed from Colab) is the descent record. A component that doesn't move
   sacred RMSE is dropped (decorrelation-necessary-not-sufficient).
4. **Reproduction check:** when the full recipe is in, we should land near **9.25** — if we don't,
   we have a fidelity bug, not a new idea. That's the ultimate measurable validation.

## The descent — stages, the lever, expected vs measured
Expected deltas are HYPOTHESES (the kernels give no ablation) — the point is to MEASURE them.

| stage | component(s) | why it should generalize | expected sacred | measured |
|---|---|---|---|---|
| now | geometry + naive-align + self-corr (xgb) | — | 15.2 (lgb 14.45) | ✔ 15.18 / 14.45 |
| **M1** | **nearby-well spatial dip** (cKDTree → weighted dip plane from neighbors' full TVT) | geology is spatially coherent across the field (slides 12-13); cross-well, not per-well noise | **≤ 13.5 (break floor)** | ✗ v1 surf=Z−TVT interp **547ft** — falsified: TVT is typewell-frame (baseline differs ~2000ft well-to-well), not a global datum |

> **M1 course-correction (2026-05-26):** the surf-datum hypothesis is falsified — TVT isn't cross-well
> comparable (per-well typewell frames). Wells ARE densely clustered (~1366 ft spacing), so spatial
> structure exists, but only **frame-independent** quantities (dip *rate* dTVT/d-displacement) are
> shared — naive absolute-TVT interpolation can't work. This is the 3rd from-scratch component to miss
> on a subtle detail the public kernel already handles (typewell frames / momentum / uncertainties).
> **DECISION FORK:** (A) faithfully PORT the 9.251 kernel's `build_well` + numba helpers (PF-momentum,
> multi-scale/stochastic DTW, beam, typewell-aware spatial, formations) → reproduce ~9.25 on sacred
> (a measurable target), then innovate on top; vs (B) keep reinventing each component (slower, repeatedly
> missing kernel details). Recommend **(A)** — it's the more measurable + adequate path to 9.5.
| **M1** | **PF-with-momentum backbone** (anchored GR tracker, tracks TVT velocity) | smooth anchored path; the GR lever done right | with spatial: **~12.5** | — |
| **M2** | multi-scale DTW (radii 20/50/100/200) + stochastic-DTW **uncertainties** (std/cv) | diverse alignment estimates + confidence the GBDT can weight | **~11.5** | — |
| **M2** | beam search (multi-config) + multi-scale NCC + GR affine calibration | more decorrelated estimates; fixes GR scale mismatch | **~11.0** | — |
| **M3** | formation/boundary context (6 markers + typewell Geology) | structural priors | **~10.3** | — |
| **M3** | GBDT **ensemble** (LGB×3 + CatBoost×3 seeds) + hill-climb blend on sacred-OOF | variance reduction + supervised blend (L48/L53) | **~9.5** | — |

## Decision gates
- **G1 (after M1):** if spatial + PF do NOT break the floor on sacred → the generalizable signal is
  weaker than the kernels suggest; re-examine (GR-scale, neighbor density for test wells) before more build.
- **G2 (after M2):** if we're not ≤ ~11.5 → an alignment component is buggy; ablate, don't pile on.
- **G3 (after M3):** if the full stack ≠ ~9.25 → fidelity gap vs the kernel; diff against it.
- Drop any component with non-positive marginal Δ on sacred. Stop adding when sacred RMSE plateaus.

## Why this is "adequate"
It (a) targets a measured number (9.5) against measured anchors, (b) attributes the gap to concrete
recipe components with expected deltas, (c) measures each on the only trustworthy signal (sacred),
(d) has explicit kill/continue gates, and (e) ends with a reproduction check against the known 9.25.
The biggest, most-generalizable lever (nearby-well spatial dip) is sequenced first.
