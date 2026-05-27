# Learned Locator — supervised GR→typewell alignment (the real Phase-2)

The public kernel locates the bit with **hand-coded, target-free** matchers (DTW/PF/beam/NCC) that
minimise *GR-match distance*. We build a **learned locator**: cross-attention from the horizontal GR
onto the typewell barcode, trained **end-to-end on TVT** — it optimises the real objective and can use
the 773 labelled wells the hand-coded matchers throw away. This is the lever we strawman-killed before
(the earlier CNN had *no typewell*); here the typewell is *inside* the model as attention memory.

## Core idea (soft, learned DTW)
For each scored point, a **query** GR pattern attends over a **memory** of (GR-pattern → TVT) slots; the
attention-weighted TVT is the prediction. Memory = the typewell **plus** the well's own pre-PS section
(slide 9: the horizontal's prefix GR is higher-res than the typewell). Same TVT frame throughout
(typewell TVT *is* the well's frame), so values are stored as **drift = TVT − last_known_tvt** →
frame-independent + ensemble-compatible.

## Architecture
```
shared SIAMESE GR-window encoder  E:  GR window [L] → embedding [d]   (1D-CNN, the same weights for q & k)

memory (per well, encoded ONCE):
   typewell pos j:  key_j = E(typewell GR window @ j)     val_j = tw_TVT_j − last_known
   prefix  pos p:  key_p = E(horiz GR window @ p)         val_p = TVT_p  − last_known
query (each post-PS point i):
   q_i = E(horiz GR window @ i)
   attn_i = softmax( q_i · K^T / √d  + locality_bias )  over memory     ← the LEARNED locator
   drift_i = attn_i · V            (+ small MLP head / residual, optional)
loss = MSE(drift_i, TVT_i − last_known)        # train end-to-end on the true target
```
- **Window, not point:** keys/queries encode a GR *window* (±W) — a single GR value is ambiguous; the
  *pattern* is what locates (the DTW analog). Encoder shared between query & memory → comparable space.
- **Memory encoded once per well**, reused across all its query points (cheap); attention is a matmul.
- **`locality_bias`** (optional, add if it helps): a soft prior that consecutive query points attend to
  monotonically-advancing typewell depths (the bit moves continuously) — keeps it from teleporting.

## Tensors (first config)
L = 65 (W=32) · d = 64 · encoder = 3×{Conv1d k5 → BN → ReLU → MaxPool2} → AdaptiveAvgPool → Linear(d) ·
single-head attention (add multi-head if it helps) · head = Linear(d→1) residual on the attended drift.

## Training & validation
- **Episodic by well:** each well = one episode (its memory + its query points); MSE on post-PS drift.
- **By-well GroupKFold(5)** → honest OOF on dev; predict sacred + test (bagged over folds). Blend with
  the GBDT (`nm_optimize_oof` on dev-OOF) → judge on **sacred** vs 9.16. The bet: a *decorrelated, strong*
  member (unlike the strawman CNN at corr 0.62 / weight 0).
- Predict **drift**; anchor to `last_known_tvt`.

## Small-data armour (773 wells — the real risk)
dropout (encoder + attention) · weight decay · **augmentation**: GR additive noise, window jitter,
random memory subsample (drop typewell/prefix slots), per-episode prefix length jitter · small model ·
early-stop on by-well val · gradient accumulation over wells for a stable step.

## Files / plan
`src/locator_data.py` (torch-free: per-well memory + query windows, global GR z-score) → `src/locator_model.py`
(siamese encoder + cross-attention + by-well CV → OOF/sacred/test) → `notebooks/15_locator.py` (orchestrate,
blend w/ GBDT, sacred verdict). Honest stance: most-promising untried lever; may overfit on 773 wells; judged
on sacred, built strong, not pre-killed.
