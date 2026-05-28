# C — Soft-DTW signal lever (the real reach)

The audit + frontier agent both endorsed proper **differentiable soft-DTW** as the rogii signal play.
My earlier locator was the strawman (point-wise attention without DTW's monotonic-sequence backbone).
This builds it right: **Cuturi soft-DTW between the post-PS horizontal GR and the typewell, with a
*learned* GR cost, trained supervised on TVT** — and read out via the **soft alignment matrix** as a
weighted-average TVT predictor.

## What it does (one frame)
```
horizontal GR (post-PS): h[0..N]   ─encode─►   H[0..N, d]
typewell (band ±W around anchor): t[a-W..a+W], TVT[j]   ─encode─►   T[0..M, d]
                                                  (shared SIAMESE GR-window encoder)

cost  D[i,j]  =  1 − cos(H[i], T[j])                ← learned similarity (not raw L2)
soft-DTW γ (band-restricted recursion):
       R[i,j] = D[i,j] + softmin_γ(R[i-1,j], R[i,j-1], R[i-1,j-1])
soft alignment α[i,j] = ∂R_final/∂D[i,j]  (Cuturi)   ← differentiable, monotonic by construction

readout       TVT_pred[i] = Σ_j  norm(α[i,:])_j  · TVT[j]
loss          MSE(TVT_pred − last_known , true drift)        end-to-end on TVT
```

## Why this is the right tool (and not point-wise attention)
- **Monotonic by construction** — DTW's edge that point/seq attention lacks. Soft-DTW relaxes "hard
  monotonic" into a *Gibbs distribution over monotonic paths* at temperature γ. Continuity, but soft.
- **Learned cost** — D is computed from a trained encoder, not raw GR L2 (which the hard-match showed is
  below floor). Encoder + cost are *supervised on TVT*, optimizing the real objective DTW can't.
- **The anchor + band kill the GR-repeat ambiguity** — restrict typewell candidates to ±W rows around the
  anchor's TVT (drift physically bounded ±100 ft → ~few hundred typewell rows). Same physics as the kernel's
  PF, but learned and differentiable.

## Tensors (first config)
GR-window encoder: same siamese 1D-CNN as the locator (L=65, d=64). γ=1.0 (tunable). Band W=150 typewell
rows (≈ generous drift envelope). Predict **drift** (anchored to last_known_tvt) — frame-independent,
ensemble-compatible with the kernel GBDT.

## Training & evaluation
- Episodic by well; **GroupKFold(5) by well** → honest OOF on dev; predict sacred + test (bagged folds).
- Loss = MSE on post-PS drift. AdamW, ~25 epochs, early-stop on by-well val.
- Augmentation (773 wells = small for DL): GR noise + window jitter + typewell sub-band shifts.
- **Blend with the kernel** via `caruana_select` (the now-wired engine) on dev-OOF → sacred verdict.

## The big practical issue + how I handle it
Soft-DTW is **sequential** (R[i,j] needs R[i-1,*]) → can't parallelize across i. Naive PyTorch with
autograd through the recursion is slow. **Mitigations:** (a) **band-restricted** (only W=150 j-cells per
i → ~600 ops/i × 4000 i = ~2.4M ops/well, fast on GPU); (b) anti-diagonal vectorization (compute all
cells on an anti-diagonal in parallel — cuts wall-clock ~3×); (c) if too slow, drop to `pytorch-softdtw-cuda`.

## Decision gate
- ✅ **soft-DTW⊕kernel < 9.100 by ≥ 0.05 on sacred** → real new signal, push to medal range.
- ✗ flat → DTW's monotonic envelope is genuinely the ceiling here too; 9.100 is the honest end.

## Build order
1. `src/sdtw_data.py` (torch-free, per-well episodes — query + typewell-band + anchor + drift target).
   Local-testable.
2. `src/sdtw_model.py` (siamese encoder + band-restricted soft-DTW recursion in torch + readout +
   episodic train_cv with augmentation). Local CPU smoke test on a tiny case.
3. `notebooks/20_sdtw.py` (Colab: train → OOF/sacred/test → Caruana blend with kernel → verdict).
