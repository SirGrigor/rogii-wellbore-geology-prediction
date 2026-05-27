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
| ~~our old target~~ | ~~9.5~~ | ✅ beaten (sacred 9.155) — superseded |
| **v5_ensemble (submitted)** | sacred **9.155** | **LB 9.644** (3-well, noisy). Same 3 wells: kernel **9.251** vs us **9.644** → **~0.4 ft fidelity+data gap** (we train on 1/8 rows; kernel uses all) |
| **NEW target** | **~8.2** | winning-pool moved here (below old public best 9.25 → needs new signal, not just the port) |

Gap to close (reset 2026-05-27): **9.155 → 8.2 ≈ 0.95 ft sacred**. The old 9.25 public best is no
longer enough — the pool moved to 8.2, so part of this gap requires *orthogonal* signal beyond the
ported kernel, not only refinement. See "Descent to 8.2" below.

Gap to close (historical): **14.08 → 9.5 ≈ 4.6 ft**, distributed across the recipe's components (no single
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
| **M1 (A)** | **ported 9.251 engine (222 feats), single LGB, stride-8 (380K rows)** | PF-momentum + spatial imputers + DTW/beam + slope, stacked under a GBDT | ≤13.5 | **✅ 9.490 (Δfloor −4.59) — broke floor, at target, near public-best 9.251** |
| **M3** | **LGB×3 + CatBoost×3(GPU) + supervised OOF blend (stride 8)** | model diversity + variance reduction; cat is the strongest family here | ≤9.25 | **✅ 9.155 — beat target AND public-best-9.251 (on our 150-well holdout). cat-GPU solo 9.14–9.26 (best); blend≈simple-avg; lgb1/lgb2 zeroed.** |

**M3 DONE (2026-05-26).** Sacred 9.155 on stride-8 (1/8 data), no tuning. Headroom remains: lower stride
(more data) + target transform. Caveat: 9.155 is our *sacred* (150 train wells); the 3-well LB is a
separate noisy check — submit once to confirm scale, trust sacred for decisions.

---

## Descent to 8.2 (reset 2026-05-27, after first LB)
First LB confirmed scale (v5 LB 9.644). Sequenced cheap→expensive, each MEASURED on sacred with a
written hypothesis+predicted_delta FIRST (diary rule). Stop adding when sacred plateaus; periodically
confirm on LB but **decide on sacred**.

| stage | lever | why it should generalize | predicted sacred | measured |
|---|---|---|---|---|
| **S1a** | **full data: stride 8 → 1** (LGB binned is memory-safe; cat-GPU on full) | the kernel trains on all rows; GBDTs scale with data. Closes the *data* half of the kernel gap | 9.155 → **~8.7** | — |
| **S1b** | **per-well prediction smoothing** along MD (Savitzky-Golay / robust spline on the drift curve) | TVT along a wellbore is a smooth geological surface; GBDT predicts rows independently → jitter. Physical prior, ~free | −0.1…−0.3 | — |
| **S1c** | **fidelity diff vs the 9.251 kernel** (features identical? its GBDT params? any post-proc/clipping?) | on the SAME 3 LB wells kernel=9.251 vs us=9.644 → a concrete gap to close against a known number (G3) | toward 9.25 LB | — |
| **S2** | **decorrelated NN: 1D-CNN/GRU on (GR seq + trajectory) → drift**, blended/stacked with the GBDT | the port is target-FREE DTW; a *supervised* sequence model learns GR→TVT directly → orthogonal signal. This is the lever that beats the moved pool (8.2 < old public 9.25). Compute-parity: Deotte ⇒ out-GBDT-ing him is hard; our edge is the NN | **~8.4** | — |
| **S3** | refine: target transform (Huber/per-step Δ), feature selection, stacking>blend | squeeze variance once new signal is in | **~8.2** | — |

**FAST proxy validated (2026-05-27, v6s8_fast):** 63-leaf/depth-6 at stride-8 → sacred **9.166** vs v5's
255/depth-7 **9.155** (Δ+0.011 = noise) in **53 min** (~1.7× faster; ceiling is cat-GPU rounds, not LGB
leaves). ⇒ **iterate the ladder in `ROGII_FAST=1`**; 255 adds no measured quality, kept only for a final
re-check at the winning stride (more data *might* let its capacity pay off — faithfulness was at 1/8 data).
Also: the blend zeroed **lgb1/lgb2** in both v5 and v6s8_fast → drop them (free, zero-weight) → lean
4-model recipe (lgb0 + cat3/4/5).

**Decision gate G4 (after S1):** if full-data + smoothing + fidelity-fix doesn't get sacred ≲ 8.6 / LB
≲ 9.25, the gap is structural → go to S2 (new model family), don't keep tuning GBDT. **Lead with new
signal (data, NN, features), not HPO** (HPO refines existing signal; it won't move a 1-ft wall).
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
>
> **CHOSE (A) — done (2026-05-26):** ported the 9.251 `build_well` → `src/kernel9251.py` (PF-momentum,
> multi-scale/stochastic DTW, beam, NCC, affine cal, formation/ANCC spatial imputers; credit
> nihilisticneuralnet). 222 features, target=drift, validated locally on 1 well (1.5s/well).
> `notebooks/06_kernel_baseline.py` (v4_kernel9251): DEV-fit imputers → build_dataset (cached) → single
> LGB → sacred. **Queued for Colab.** Single-LGB target ≈ 10-11 (clear floor); the LGB×3+Cat×3 ensemble +
> hill-climb blend (M3) → ~9.25. Then innovate on top.
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
