# ROGII Wellbore Geology Prediction — Post-mortem (2026-05-27)

**Result: sacred RMSE ~9.16** (150-well holdout). Broke the carry-forward floor (14.08) and matched
the best *public* tier (~9.0–9.25) with disciplined CV. Locked submission: **v5_ensemble, public LB
9.644** (the 3-well public LB is noise — see below). Deadline 2026-08-05.

## What worked (floor 14.08 → 9.16)
- **Faithful port of the public 9.251 DWT kernel** (`src/kernel9251.py`: PF-momentum, multi-scale/
  stochastic DTW, beam search, multi-scale NCC, GR affine calibration, formation/ANCC spatial imputers
  — 222 features, drift target). The "go for A (port), don't be the smartest in the room" call: matched
  public-best fast instead of re-deriving each component (we'd already missed 3 components from scratch).
- **LGB + CatBoost ensemble** on those features, GroupKFold-by-well OOF + supervised blend → sacred 9.155.
- **Sacred-holdout discipline** (150 wells, seed-locked) as the only trusted number.

## The wall: 6 independent levers, all flat at ~9.16
| lever | result |
|---|---|
| more data (stride 8→4, 2× rows) | flat (9.166 → 9.192) |
| more capacity (63→255 leaves) | flat (9.166 ≈ 9.155) |
| orthogonal NN (1D-CNN on raw GR) | no lift; blend weight 0; **resid corr 0.62** |
| new features (horizon/geom/GR-texture) | flat (Δ −0.011 / 0.000) |
| post-proc (savgol) + CatBoost border_count=254 | flat (Δ −0.005; border *hurt* +0.026) |
| orthogonal foundation model (TabPFN) | no lift; blend weight 0; **resid corr 0.867** |

Six flat results = a **signal ceiling**, not a tuning gap. The GBDT already extracts the signal these
inputs carry.

## Lessons (transferable)
1. **A different model class ≠ a different signal.** Both "orthogonal model" bets (CNN, TabPFN) got zero
   blend weight because they relearned the GBDT's function from the same/related inputs (corr 0.62, 0.87).
   Decorrelation requires a genuinely different *signal source*, not just a different estimator.
2. **A 3-well public LB is pure noise.** v6 (sacred-tied with v5) swung 0.44 ft on the LB (9.644 → 10.085).
   The "8.2 winning pool" sits on that noisy LB → partly luck. **Decide on sacred, always**; never chase a
   noisy public LB. The medal is the private (full-test) LB, where our robust 9.16 is the honest estimate.
3. **Recognize the ceiling early.** After ~3 flat levers, stop refining and either find new *signal* or
   harvest. We spent the budget confirming, but the rule held: HPO/features won't move a signal wall.
4. **Recon: separate the published tier from the unpublished leaders.** Every public kernel (romantamrazov,
   mitch, kojimar/TabICL) is ~9.0 — *our tier*, which we matched. The 7.97–8.6 leaders never published, so
   the path below ~9.0 is genuinely unknown (private domain insight). Compute wasn't the gap — Deotte (NVIDIA
   GM) sat at #7 (8.797) — but the method to beat him isn't public.
5. **Faithful porting beats reinvention** for matching a strong public baseline efficiently.

## Reusable toolkit (the portfolio value)
- `kernel9251.py` — verbatim port pattern for adopting a strong public kernel as a feature engine.
- `colab/bootstrap.py` — idempotent Colab workflow (fresh clone → install → GPU-verify → run → Drive sync
  → diary-to-git), env-driven (stride/fast/window/token).
- `src/dashboard.py` — rich training dashboard (goal banner + live scoreboard + verdict), Colab-safe.
- Experiment-diary discipline (hypothesis + predicted_delta first; observer/MetricSpec).
- **OOF-stacking-by-id** across model families (GBDT + NN + TabPFN aligned on `{well}_{iloc}`).
- Feature-ablation harness (`09_features`), post-proc module (`postproc.py`, per-well savgol tuned on OOF),
  NN pipeline (`nn_data`/`nn_model`), TabPFN integration (`11_tabfm`).

## Status
Harvested 2026-05-27. Active modeling push concluded (evidence-backed levers exhausted). v5 stays the
final submission. Revisit only if the field publishes a sub-9 technique before the Aug-5 deadline.
