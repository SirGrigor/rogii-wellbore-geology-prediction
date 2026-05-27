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
| **S1a** | **full data: stride 8 → 1** (LGB binned is memory-safe; cat-GPU on full) | the kernel trains on all rows; GBDTs scale with data. Closes the *data* half of the kernel gap | 9.155 → **~8.7** | ✗ **FALSIFIED (2026-05-27):** stride-8 9.166 → stride-4 (2× data) **9.192** — flat/worse. Data lever exhausted; not data-limited. **Ladder stopped** (stride-2/1 would also ~9.16). Capacity also flat (63≈255) ⇒ both GBDT levers dead → **G4: pivot to S2 (new signal)**. Note: simple-avg 9.166 < OOF-blend 9.192 (blend overfits dev w/ 4 corr. models) |
| **S1b** | **per-well prediction smoothing** along MD (Savitzky-Golay / robust spline on the drift curve) | TVT along a wellbore is a smooth geological surface; GBDT predicts rows independently → jitter. Physical prior, ~free | −0.1…−0.3 | — |
| **S1c** | **fidelity diff vs the 9.251 kernel** (features identical? its GBDT params? any post-proc/clipping?) | on the SAME 3 LB wells kernel=9.251 vs us=9.644 → a concrete gap to close against a known number (G3) | toward 9.25 LB | — |
| **S2** | **decorrelated NN: 1D-CNN/GRU on (GR seq + trajectory) → drift**, blended/stacked with the GBDT | the port is target-FREE DTW; a *supervised* sequence model learns GR→TVT directly → orthogonal signal. This is the lever that beats the moved pool (8.2 < old public 9.25). Compute-parity: Deotte ⇒ out-GBDT-ing him is hard; our edge is the NN | **~8.4** | **Phase-1 NO-GO (2026-05-27, v7_nn_p1):** 1D-CNN(GR±128+ctx) on full 3.05M rows → NN-alone sacred **13.87** (barely < floor), blend weight **0.000**, NN⊕GBDT **9.172** = GBDT-only **9.172** (no lift), **resid corr 0.621**. The NN's errors substantially overlap the GBDT's — the 222 feats already encode GR↔typewell alignment, and Phase-2 attention would *raise* corr (same signal). Decorrelation didn't materialize ⇒ don't build Phase 2. Pivot to S-feat. |
| **S3** | refine: target transform (Huber/per-step Δ), feature selection, stacking>blend | squeeze variance once new signal is in | **~8.2** | — |

**LB IS NOISE — confirmed (2026-05-27):** v6s4_fast (sacred 9.166, ~tied with v5's 9.155) scored **LB
10.085** vs v5's **9.644** — a 0.44 swing on 3 wells between sacred-equivalent models. ⇒ the 3-well LB
cannot rank our models; **decide on sacred, always.** v5 stays the locked submission. Makes S1c
(chase the 9.64-vs-9.25 LB gap) a clear noise-chase — deprioritized.

**FAST proxy validated (2026-05-27, v6s8_fast):** 63-leaf/depth-6 at stride-8 → sacred **9.166** vs v5's
255/depth-7 **9.155** (Δ+0.011 = noise) in **53 min** (~1.7× faster; ceiling is cat-GPU rounds, not LGB
leaves). ⇒ **iterate the ladder in `ROGII_FAST=1`**; 255 adds no measured quality, kept only for a final
re-check at the winning stride (more data *might* let its capacity pay off — faithfulness was at 1/8 data).
Also: the blend zeroed **lgb1/lgb2** in both v5 and v6s8_fast → drop them (free, zero-weight) → lean
4-model recipe (lgb0 + cat3/4/5).

**Decision gate G4 (after S1):** if full-data + smoothing + fidelity-fix doesn't get sacred ≲ 8.6 / LB
≲ 9.25, the gap is structural → go to S2 (new model family), don't keep tuning GBDT. **Lead with new
signal (data, NN, features), not HPO** (HPO refines existing signal; it won't move a 1-ft wall).

## ⛔ WALL at sacred ~9.16 (2026-05-27)
Four independent attempts to break below 9.16 all flat — this is a SIGNAL ceiling, not a tuning gap:
1. **data** (stride 8→4, 2× rows): 9.166 → 9.192 (flat/worse).
2. **capacity** (63→255 leaves): 9.166 ≈ 9.155 (flat).
3. **orthogonal NN** (v7, 1D-CNN raw GR): NN-alone 13.87, blend weight 0.000, resid corr 0.62 (not decorrelated).
4. **new features** (v8, W1 horizon/geom + W2 GR-texture): ΔW1 −0.011 (noise), ΔW2 0.000 (CatBoost didn't split on them — identical best-iters). **Keep W1, drop W2.**

Target 8.2 is ~1 ft below the ceiling. Remaining genuinely-orthogonal bets (untried): **cross-well
spatial drift-rate** (Wave-3, neighbours' frame-independent dip — leakage-careful, OOF-fold-only
neighbour stats), and a **reconnaissance revisit** (what do the 8.2 leaders do that the public 9.25
kernel doesn't?). If those miss too, ~9.16 is our honest result (broke floor 14.08, matched public-best
on sacred) and the medal likely isn't reachable this round — harvest the lessons + reusable toolkit.

## 🔓 Recon revisit — the wall has KNOWN exits (2026-05-27)
LB top 7.97 → 8.24 → 8.46…; **Deotte (NVIDIA GM) only #7 at 8.797** ⇒ the gap to 8.2 is METHOD, not
compute. Three public solutions (romantamrazov TOP-3 ~8.0; kojimar TabICL stack; mitchgansemer 9.40)
reveal what we're missing — and our last two bets used the **wrong instance** of the right idea:
- **TabICL** (tabular foundation / in-context model) is the orthogonal estimator that works — kojimar's
  stack is *dominated by the TabICL branch*. (Our raw-GR CNN was the wrong orthogonal model; corr 0.62.)
  Public `rogii-tabicl-mirror` dataset runs it offline.
- **Uncertainty/divergence features** are the right features: inter-signal std (master uncertainty),
  formation-consensus std/range, per-formation known-zone RMSE, estimator divergence. (Our W1/W2
  horizon/GR-texture were the wrong features → flat.) These tell the GBDT *where to trust the drift*.
- **savgol smoothing + clip post-processing** (the deferred Strategy-3) — the TOP-3 solution uses it.
- **ridge/NNLS stack** (positive weights), 8000 iters + heavier reg, DWT-GR features.

**Revised descent (evidence-backed):** S3a savgol+clip post-proc → S3b uncertainty/divergence feats →
S3c TabICL stack member → S3d ridge/NNLS stack. Medal looks reachable (method, not compute).

## ⚠ Reframe — public ceiling ≈ where we are; the 8.2 LB is NOISE (2026-05-27, after S3a)
**S3a flat (5th flat result):** savgol Δ−0.005 (noise), border_count=254 Δ+0.026 (hurt), uncertainty
feats already in the 222. Re-reading recon corrects an earlier over-claim:
- **Every PUBLIC kernel is our tier (~9.0):** romantamrazov "TOP-3" is *stale* (LB moved) and targets <9.0
  not 8.0; mitch 8.9; kojimar's TabICL stack isn't in the top-20 either. The 7.97–8.6 leaders are **private**
  — we do NOT actually know what reaches the medal pool.
- **The 8.2 'pool' is on the 3-well PUBLIC LB = NOISE** (we proved 0.44 swings for sacred-tied models, v6).
  Chasing it violates our own decide-on-sacred rule. The medal is the **private** LB (full test), where our
  robustly-validated **sacred 9.16** is the honest estimate — and may rank far better than the noisy public 8.2.
- **We're not failing — we matched the public ceiling with disciplined CV.** Remaining genuine-sacred bets:
  **TabICL** (decorrelated family; uncertain — public TabICL isn't top-tier) and a cheap more-iterations test.
  If those are flat too, harvest a robust 9.16 + the reusable toolkit (portfolio-over-medals).

## 🔬 NOT harvesting — evidence-directed signal search (2026-05-27, ~10wk to deadline)
TabPFN also NO-GO (corr 0.87, weight 0 — same lesson as the CNN). But 6 flats ≠ ceiling with 10 weeks left;
they prove the **per-well-alignment paradigm is saturated** — the missing signal is in info the 222 DON'T
encode. **Stop guessing; diagnose.** `12_residual_diag.py`: slice sacred error by horizon / align-uncertainty
/ spatial-density / drift-magnitude + rank features by corr with |resid| + a spatial litmus (do
close-neighbour wells err less?). The regime carrying the error = the axis to chase. Leading hypothesis:
**cross-well / field-level structure** (the per-well paradigm aligns each well to its OWN typewell
independently — it structurally can't see what neighbours reveal about the shared geology). Candidate signals
queued: cross-well dip field (neighbours' prefix dTVT/d-displacement → local dip → drift; frame-independent,
so it dodges the falsified absolute-TVT M1; leakage-free via known prefixes), per-step increment target,
joint field/graph model. (docs/postmortem.md = the fallback if the search dead-ends, not the current plan.)

**Cross-well dip — FLAT (v12_spatialdip, 7th flat).** Frame-independent neighbour-dip features (local
validation corr 0.32, FAR 0.37>NEAR 0.28 — signal real) added ZERO marginal Δ on sacred (+0.007; far −0.005;
CatBoost barely split on them). ⇒ the kernel's formation-plane-KNN ALREADY captures the cross-well dip; the
222-feature *representation* is saturated. The latent spatial advantage (near 5.54 vs far 7.50) is the kernel
exploiting it, not headroom. **Untouched lever:** the TARGET/LOSS — all 7 kept MSE-on-drift, but the biggest
diagnostic finding (large-drift under-prediction 5.4→16.2 = mean-regression) is a loss symptom. Next directed
test: sample-weighting / de-regression / Huber. If flat too → representation saturated; only a paradigm change
(GNN/seq2seq over the well-field) or accepting ~9.16 near-ceiling remains.

## 💡 STRUCTURAL FINDING — TVT is a geometric identity (2026-05-27, re-examination)
Stepped out of the kernel's GR-alignment worldview; read the .pptx brief + raw EDA. Found:
- **`TVT = −Z + g(formations) + const_well`**, exactly. Per-well fit: **Z coef = −1.000, std 0.0000** (all
  297 wells); pooled `drift = −dz + Δg(formations)` → **RMSE 0.007**. TVT is *defined* by the bit depth Z
  (known for test) relative to the 6 near-parallel dipping formation surfaces. **GR is only an indirect proxy
  to locate those surfaces.** (Shared-typewell transfer is dead — test wells have unique typewells.)
- ⇒ **The entire ~9.16 error is FORMATION-IMPUTATION error.** Test lacks formations; the kernel imputes them
  only as side-features for a GR-GBDT. **New paradigm (evidence-grounded): impute the 6 formation surfaces
  accurately → reconstruct TVT via the identity.** Genuinely different from GR-alignment, AND imputation
  accuracy is **directly measurable on train** (true formations known) — a tight loop the GR paradigm lacks.
- **Next experiment (S-formation):** hold out train wells → spatially impute their 6 formation surfaces from
  neighbours (validate vs known, ft-level) → reconstruct drift = −dz + Δg → sacred RMSE. Blend with the GBDT
  (decorrelated path). If imputation beats the kernel's incidental one → breaks the wall. Caveat: isolated
  wells (sparse neighbours) impute worst — and the 3 test wells may be isolated (limits LB, not sacred).

**S-formation — DEAD END, but it EXPLAINS the ceiling (2026-05-27, local test).** Identity confirmed (g-fit
RMSE 0.007 with TRUE formations). But IDW spatial imputation gives formation MAE ~28 ft, and reconstruction
`drift = −dz + g·Δ(imputed formations)` → **RMSE 55** (≫ 9.16). Reason: TVT/drift (±35 ft) is a *small
difference of large* formation depths (~−9400); differencing two ~28-ft-MAE imputations amplifies noise
beyond the signal. **This is WHY GR-alignment is necessary** — it resolves TVT *directly* to ~9 ft without
needing precise absolute formations; precise formations would themselves require GR-alignment (circular).
⇒ **~9.16 is near the GR-alignment ceiling, and GR-alignment is provably the right tool** (geometric
reconstruction is 6× worse). The structure is real but not an independent lever. Remaining: a booster
overview (within-paradigm, marginal-blend odds) per user ask; else the understood-ceiling harvest.
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
