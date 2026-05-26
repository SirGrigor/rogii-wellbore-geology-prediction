# Signal-revelation report — drift target (2026-05-26)

`notebooks/02b_revelation.py` (synth-decoder `signal_revelation`, fit-free MI+polyfit) on 120 dev
wells, target = drift Δ = TVT − anchor. Measures what carries signal + its shape, to guide FE.

## Dependency (MI with drift)
| feature | MI | read |
|---|---|---|
| anchor_tvt | 1.17 | level — proxy for drift magnitude/regime |
| **traj_azimuth_cos / sin** | **0.92 / 0.85** | **azimuth → geology dip direction** (brief slide 12) |
| GR_grad | 0.57 | GR rate-of-change carries signal |
| GR_roll31_mean / roll15 | 0.51 / 0.38 | smoothed GR level |
| GR_accel | ~0 | **dead** (2nd-derivative bust — consistent with S6E5 L46) |

## Shape (→ matched FE)
- **`z_from_ps` threshold, mono −0.97** and **`md_from_ps` mono +0.85** — strong monotonic
  geometric drift: the well's vertical/along-hole displacement largely *is* the drift (flat-geology
  component). Trees handle monotonic well; these are the backbone features.
- **`poly_drift` / `poly_slope` mono +0.86** — validates the polyfit dip feature carries real signal.
- `align_tvt` u-shaped, `align_shift` inverted-u — non-linear; the alignment recipe is still naive.

## Interactions (positive joint MI → build crosses)
`poly_slope×align_cost` (+0.25), `poly_drift×align_cost`, `align_tvt×align_cost`,
`align_tvt×z_from_ps`, `z_from_ps×align_cost`, `align_tvt×poly_slope`. → geometry × alignment crosses
carry joint signal (Tier-1 FE).

## The reframe
Pure geometry/trajectory (`z_from_ps`, `azimuth`, `md_from_ps`, `poly_drift`) is a **strong,
test-available** drift signal — physical: drift ≈ vertical displacement + an azimuth-dependent dip
correction. So a residual GBDT should beat the 15.91 ft floor **even before the alignment recipe is
fixed**; alignment is one lever among several. The Colab baseline's feature-gain ranking quantifies
how much alignment adds vs geometry — that decides whether the alignment recipe is the priority.

## What this means for the plan
1. The baseline (`03_baseline_lgb.py`) should already beat the floor via geometry — confirm on Colab.
2. Don't over-index on alignment yet; first see its gain rank. Drop `GR_accel` (dead).
3. Tier-1 FE candidates (gate each with/without fit): azimuth-dip interactions, `z_from_ps`/`md_from_ps`
   threshold/monotonic shape, alignment×geometry crosses, a better alignment recipe.
4. **GATE caveat (handover §5):** fit-free residual proxies only flag bias; the definitive feature
   gate is with/without-fit OOF on dev wells (+ a final sacred-well check).
