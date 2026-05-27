# S2 — the sequence NN (orthogonal signal toward 8.2)

GBDT-on-the-222-features is capped at sacred **~9.16** (data + capacity levers both flat,
2026-05-27). Target 8.2 needs *orthogonal* signal, not more GBDT. This is that bet.

## The framing: few-shot alignment
The standard view is "align horizontal GR to the typewell." Stronger: the **known prefix**
gives each well its own (GR→TVT) calibration. So per scored point:

> given a **support set** — this well's prefix `(GR,TVT)` pairs **+** the typewell `(GR,TVT)`
> reference — and a **query** GR context, retrieve/interpolate the matching TVT.

A few-shot regression. DTW does a hard, unsupervised version; a NN does a learned, supervised,
well-calibrated one. GR is observed along the **entire** horizontal (only TVT is missing
post-PS), so windows centre with full context. **Target = drift (TVT − last_known_tvt)** — same
as the GBDT, so OOF stacks cleanly.

## Phases
- **Phase 1 — 1D-CNN (cheapest test).** CNN over the local GR window + geometry context
  (`md_since_ps`, `dz_since_ps`, `incl_local`, `gr_at_point`) → drift. *No attention/typewell yet.*
  Answers: does raw GR carry signal **orthogonal** to the 222 features?
  - `src/nn_data.py` (torch-free, local-testable): per-well filled GR + scored samples + ctx.
  - `src/nn_model.py`: vectorised window materialise + CNN + by-well CV → OOF/sacred/test.
  - `notebooks/08_nn_phase1.py`: NN ⊕ lean GBDT, **blend WITH vs WITHOUT the NN on sacred**.
- **Phase 2 — neural alignment (the differentiator).** Cross-attention from the post-PS GR
  context (query) to the prefix∪typewell `(GR→TVT)` examples (keys=GR, values=TVT) — a learned
  soft-DTW, calibrated per-well by the prefix. Only if Phase 1 passes.
- **Phase 3 — finalise.** Tune + ensemble + submission (decide on sacred).

## Decision gates
- **G-NN1 (after Phase 1):** if the NN's blend weight ≈ 0 **and** NN⊕GBDT doesn't beat GBDT-only
  by ≥ ~0.1 on sacred → raw GR is saturated by the 222 feats → **don't build Phase 2**; fall back
  to Strategy 2 (new GBDT features). If it clearly helps (≥0.1) → build the attention model.
- **Overfit watch:** 773 wells is small for DL. NN dev-OOF ≪ sacred ⇒ overfit → more
  regularization/augmentation. Likeliest failure mode.

## Validation & integrity
- By-well `GroupKFold` (honest OOF; a well never crosses folds) + the seed-locked 150-well
  sacred holdout = the only number we trust. The **3-well LB is noise** (proven: v6 LB 10.085 vs
  v5 9.644 for sacred-tied models) — never decide on it.
- NN and GBDT use independent by-well folds; both are honest OOF, aligned by `id` for stacking.
- Per-point samples (millions) not seq2seq (773) — far better for DL; by-well CV handles
  within-well correlation.

## Compute
PyTorch on Colab T4 (high-RAM). Windows pre-materialised per-well (vectorised) → fast GPU
training. `torch` added to bootstrap DEPS. Env: `ROGII_NN_WINDOW` (128), `ROGII_NN_EPOCHS` (40).
