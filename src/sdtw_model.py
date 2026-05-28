"""Soft-DTW (C) — siamese GR encoder + band-restricted soft-DTW + α-readout, end-to-end on drift.

Cuturi 2017 soft-DTW between post-PS horizontal GR (queries) and the typewell band, with a *learned*
cosine cost from a siamese 1D-CNN GR-window encoder. The soft alignment α = ∂R_final/∂D is obtained
via autograd (create_graph=True), row-softmax-normalized, and used to predict drift as Σ_j α'[i,j]·
tw_drift[j]. End-to-end MSE on drift; episodic by well, GroupKFold(5).

Soft-DTW recursion: pure pytorch, **antidiagonal-vectorized** — each iteration is one GPU op on a
vector of ≤ min(N,M) cells, so ~N+M sequential steps instead of N*M scalar ops. Single well per batch
(no padding hassle for first build); stride-8 query subsample during training (keeps the graph at
~N=500 deep), full stride at inference under no_grad.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

from .sdtw_data import SDTWEpisode


# ---------- 1. encoder ----------------------------------------------------------------------------

class GREncoder(nn.Module):
    """Siamese 1D-CNN: GR window [B, L] → L2-normalized embedding [B, d] (for cosine cost)."""

    def __init__(self, l_in: int = 65, d: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, d, 5, padding=2), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: [B, L]
        return F.normalize(self.net(x.unsqueeze(1)), dim=-1)


# ---------- 2. antidiagonal-vectorized soft-DTW ----------------------------------------------------

def _softmin(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float) -> torch.Tensor:
    """softmin_γ(a,b,c) = −γ·log(Σ exp(−x/γ)). Numerically stable via min-subtraction."""
    m = torch.minimum(torch.minimum(a, b), c)
    return m - gamma * torch.log(
        torch.exp(-(a - m) / gamma) + torch.exp(-(b - m) / gamma) + torch.exp(-(c - m) / gamma))


def soft_dtw_final(D: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """D: [N, M] non-negative cost. Returns R[N,M] (scalar, differentiable).

    Skewed-coords recursion: R_skew[i, k] = R[i, k−i] (k = i+j). Column k depends only on columns
    k−1, k−2 → all cells of column k computed in ONE vectorized torch op. We store the column list
    (not the full skewed matrix) — each col is a fresh tensor built via .scatter (autograd-clean).
    """
    INF = 1e10
    N, M = D.shape
    K = N + M + 1
    # column 0: R[0,0]=0; rest = inf. Fresh tensor → no autograd hazard.
    col0 = D.new_full((N + 1,), INF); col0[0] = 0.0
    cols = [col0]
    for k in range(1, K):
        i_lo, i_hi = max(1, k - M), min(N, k - 1)
        if i_lo > i_hi:                               # no interior cells on this antidiagonal
            cols.append(D.new_full((N + 1,), INF))
            continue
        idx = torch.arange(i_lo, i_hi + 1, device=D.device)
        diag = cols[k - 2][idx - 1] if k >= 2 else D.new_full((idx.numel(),), INF)  # R[i-1, j-1]
        up   = cols[k - 1][idx - 1]                                                  # R[i-1, j  ]
        left = cols[k - 1][idx]                                                      # R[i  , j-1]
        sm   = _softmin(diag, up, left, gamma)
        cost = D[idx - 1, k - idx - 1]               # D[i-1, j-1] = D[i-1, k-i-1]
        vals = cost + sm                              # [len(idx)]
        # build col k functionally via scatter onto an INF base
        base = D.new_full((N + 1,), INF)
        col_k = base.scatter(0, idx, vals)
        cols.append(col_k)
    return cols[K - 1][N]                             # R[N, M] = R_skew[N, N+M]


def soft_alignment(D: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """α[i,j] = ∂R_final/∂D[i,j]. [N, M], autograd-friendly (create_graph for backprop)."""
    D_req = D.detach().clone().requires_grad_(True) if not D.requires_grad else D
    R = soft_dtw_final(D_req, gamma)
    (alpha,) = torch.autograd.grad(R, D_req, create_graph=True)
    return alpha


# ---------- 3. readout: α → drift ------------------------------------------------------------------

def predict_drift(H: torch.Tensor, T: torch.Tensor, tw_drift: torch.Tensor,
                  gamma: float = 1.0, tau: float = 0.5) -> torch.Tensor:
    """H: [N, d] query embeddings, T: [M, d] typewell embeddings, tw_drift: [M]. Returns drift [N]."""
    D = 1.0 - H @ T.t()                               # cosine cost (H, T already L2-normalized)
    alpha = soft_alignment(D, gamma)                   # [N, M]
    w = F.softmax(alpha / tau, dim=1)                  # row-normalize alignment → per-i distribution
    return (w * tw_drift.unsqueeze(0)).sum(dim=1)      # [N]


# ---------- 4. episodic training (GroupKFold by well) ---------------------------------------------

@dataclass
class SDTWResult:
    oof: dict[str, float]                              # id → predicted drift on dev
    sac: dict[str, float] | None                       # id → predicted drift on sacred (None if absent)
    test: dict[str, float] | None                      # id → predicted drift on test
    fold_rmse: list[float]
    epochs_run: int


def _augment(q_gr: np.ndarray, tw_gr: np.ndarray, noise: float = 0.05, jitter: int = 2,
             rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng()
    # GR noise (both sides) + small typewell band shift (jitter)
    q = q_gr + rng.normal(0, noise, q_gr.shape).astype(np.float32)
    t = tw_gr + rng.normal(0, noise, tw_gr.shape).astype(np.float32)
    s = int(rng.integers(-jitter, jitter + 1))
    if s != 0:
        t = np.roll(t, s, axis=0)
    return q, t


def _train_one_fold(train_eps: list[SDTWEpisode], val_eps: list[SDTWEpisode],
                    *, device: str = "cpu", d: int = 64, gamma: float = 1.0, tau: float = 0.5,
                    epochs: int = 25, lr: float = 1e-3, query_stride: int = 8,
                    seed: int = 42) -> tuple[nn.Module, list[float]]:
    """Train one fold; returns trained encoder + per-epoch val-RMSE history."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    enc = GREncoder(l_in=train_eps[0].q_gr.shape[1], d=d).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=1e-4)
    val_hist: list[float] = []
    for ep in range(epochs):
        enc.train(); order = rng.permutation(len(train_eps))
        for k in order:
            e = train_eps[k]
            q_aug, t_aug = _augment(e.q_gr[::query_stride], e.tw_gr, rng=rng)
            y_aug = e.q_y[::query_stride]
            valid = ~np.isnan(y_aug)
            if valid.sum() < 4:
                continue
            q = torch.from_numpy(q_aug[valid]).to(device)
            t = torch.from_numpy(t_aug).to(device)
            y = torch.from_numpy(y_aug[valid]).to(device)
            tw_drift = torch.from_numpy(e.tw_drift).to(device)
            H = enc(q); T = enc(t)
            pred = predict_drift(H, T, tw_drift, gamma=gamma, tau=tau)
            loss = F.mse_loss(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
        # val RMSE (no_grad, full stride)
        enc.eval(); sq, n = 0.0, 0
        with torch.no_grad():
            for e in val_eps:
                valid = ~np.isnan(e.q_y)
                if valid.sum() == 0: continue
                q = torch.from_numpy(e.q_gr[valid]).to(device)
                t = torch.from_numpy(e.tw_gr).to(device)
                tw_drift = torch.from_numpy(e.tw_drift).to(device)
                H = enc(q); T = enc(t)
                # inference still needs alignment → enable grad locally for the autograd.grad call
                with torch.enable_grad():
                    pred = predict_drift(H.detach().requires_grad_(False), T.detach().requires_grad_(False),
                                         tw_drift, gamma=gamma, tau=tau)
                y = e.q_y[valid]
                sq += float(((pred.cpu().numpy() - y) ** 2).sum()); n += valid.sum()
        rmse = (sq / max(n, 1)) ** 0.5
        val_hist.append(rmse)
        print(f"  fold ep{ep+1:02d}  val_rmse={rmse:.3f}")
    return enc, val_hist


def _predict_eps(enc: nn.Module, eps: list[SDTWEpisode], *, device: str = "cpu",
                 gamma: float = 1.0, tau: float = 0.5) -> dict[str, float]:
    out: dict[str, float] = {}
    enc.eval()
    for e in eps:
        q = torch.from_numpy(e.q_gr).to(device)
        t = torch.from_numpy(e.tw_gr).to(device)
        tw_drift = torch.from_numpy(e.tw_drift).to(device)
        with torch.enable_grad():
            H = enc(q); T = enc(t)
            pred = predict_drift(H, T, tw_drift, gamma=gamma, tau=tau).detach().cpu().numpy()
        for qid, p in zip(e.q_ids, pred):
            out[qid] = float(p)
    return out


def train_cv(train_eps: list[SDTWEpisode], *, sacred_eps: list[SDTWEpisode] | None = None,
             test_eps: list[SDTWEpisode] | None = None, n_folds: int = 5, device: str = "cpu",
             **kw) -> SDTWResult:
    """GroupKFold(by well) → OOF on train + averaged predictions on sacred/test (bag of folds)."""
    wells = np.array([e.well for e in train_eps])
    gkf = GroupKFold(n_splits=n_folds)
    oof: dict[str, float] = {}
    sac_sum: dict[str, list[float]] = {}
    test_sum: dict[str, list[float]] = {}
    fold_rmse: list[float] = []
    for fold, (tr, va) in enumerate(gkf.split(np.zeros(len(wells)), groups=wells)):
        print(f"[fold {fold+1}/{n_folds}] train={len(tr)} val={len(va)}")
        enc, hist = _train_one_fold([train_eps[i] for i in tr], [train_eps[i] for i in va],
                                    device=device, seed=42 + fold, **kw)
        fold_rmse.append(hist[-1])
        # val OOF
        for qid, p in _predict_eps(enc, [train_eps[i] for i in va], device=device,
                                   gamma=kw.get("gamma", 1.0), tau=kw.get("tau", 0.5)).items():
            oof[qid] = p
        # sacred / test bagged averaging
        if sacred_eps:
            for qid, p in _predict_eps(enc, sacred_eps, device=device,
                                       gamma=kw.get("gamma", 1.0), tau=kw.get("tau", 0.5)).items():
                sac_sum.setdefault(qid, []).append(p)
        if test_eps:
            for qid, p in _predict_eps(enc, test_eps, device=device,
                                       gamma=kw.get("gamma", 1.0), tau=kw.get("tau", 0.5)).items():
                test_sum.setdefault(qid, []).append(p)
    sac = {q: float(np.mean(ps)) for q, ps in sac_sum.items()} or None
    test = {q: float(np.mean(ps)) for q, ps in test_sum.items()} or None
    return SDTWResult(oof=oof, sac=sac, test=test, fold_rmse=fold_rmse, epochs_run=kw.get("epochs", 25))
