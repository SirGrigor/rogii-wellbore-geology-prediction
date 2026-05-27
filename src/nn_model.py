"""Phase-1 sequence NN (S2): a 1D-CNN over the local GR window + context → drift.

The cheapest test of the S2 thesis (docs/nn_design.md): does a model learning from the
RAW GR signal carry signal that's *orthogonal* to the GBDT's 222 alignment features? We
judge it by whether NN⊕GBDT beats GBDT-alone on the sacred holdout — not the NN alone.

Windows are materialised per-well with a vectorised sliding view (iterate 773 wells, not
millions of rows). torch-only here; the data prep (nn_data) stays torch-free + local-testable.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

from .config import N_FOLDS
from .nn_data import NNData


def materialize(d: NNData, window: int) -> np.ndarray:
    """[N, 2*window+1] GR windows centred on each scored point (edge-padded), in d.samples order.
    Built per-well via sliding_window_view so it's vectorised (well loop, not sample loop)."""
    L = 2 * window + 1
    out = np.empty((len(d.samples), L), np.float32)
    pos = 0
    # d.samples is grouped well-by-well in iloc order (ps..n), matching build()
    for wid, grp in _runs(d.samples):
        w = d.wells[wid]
        padded = np.pad(w.gr, window, mode="edge")                       # len = n + 2W
        wins = np.lib.stride_tricks.sliding_window_view(padded, L)       # [n, L], row i ↔ index i
        ilocs = [i for (_, i) in grp]
        out[pos:pos + len(ilocs)] = wins[ilocs]
        pos += len(ilocs)
    assert pos == len(d.samples)
    return out


def _runs(samples):
    """Yield (well_id, [(well,iloc)...]) for each contiguous same-well run."""
    cur, buf = None, []
    for s in samples:
        if s[0] != cur:
            if buf:
                yield cur, buf
            cur, buf = s[0], [s]
        else:
            buf.append(s)
    if buf:
        yield cur, buf


class GRWindowCNN(nn.Module):
    def __init__(self, n_ctx: int, ch=(32, 64, 64), drop=0.2):
        super().__init__()
        layers, c = [], 1
        for o in ch:
            layers += [nn.Conv1d(c, o, 5, padding=2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2)]
            c = o
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(c + n_ctx, 128), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, c):                       # x:[B,1,L]  c:[B,n_ctx]
        h = self.pool(self.cnn(x)).squeeze(-1)     # [B, ch]
        return self.head(torch.cat([h, c], 1)).squeeze(-1)


def _infer(model, X, C, device, bs=8192):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).unsqueeze(1).to(device)
            cb = torch.from_numpy(C[i:i + bs]).to(device)
            out.append(model(xb, cb).cpu().numpy())
    return np.concatenate(out)


def train_cv(dev: NNData, sac: NNData, test: NNData, *, window: int = 128,
             n_folds: int = N_FOLDS, epochs: int = 40, lr: float = 1e-3, bs: int = 2048,
             patience: int = 5, device: str | None = None, seed: int = 7):
    """By-well CV CNN → (oof[dev], sacred_pred, test_pred). Honest by-well OOF for stacking."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    Xd, Cd, yd = materialize(dev, window), dev.ctx, dev.y
    Xs, Cs = materialize(sac, window), sac.ctx
    Xt, Ct = materialize(test, window), test.ctx
    n_ctx = Cd.shape[1]
    oof = np.zeros(len(dev.samples), np.float32)
    sac_pred = np.zeros(len(sac.samples), np.float32)
    test_pred = np.zeros(len(test.samples), np.float32)
    fold_rmse = []
    for f, (tr, va) in enumerate(GroupKFold(n_folds).split(np.arange(len(yd)), groups=dev.groups)):
        model = GRWindowCNN(n_ctx).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        Xtr = torch.from_numpy(Xd[tr]).unsqueeze(1); Ctr = torch.from_numpy(Cd[tr]); ytr = torch.from_numpy(yd[tr])
        dl = DataLoader(TensorDataset(Xtr, Ctr, ytr), batch_size=bs, shuffle=True)
        best, best_state, bad = 1e9, None, 0
        for ep in range(epochs):
            model.train()
            for xb, cb, yb in dl:
                opt.zero_grad()
                loss = nn.functional.mse_loss(model(xb.to(device), cb.to(device)), yb.to(device))
                loss.backward(); opt.step()
            vp = _infer(model, Xd[va], Cd[va], device)
            vr = float(np.sqrt(((vp - yd[va]) ** 2).mean()))
            if vr < best - 1e-4:
                best, bad = vr, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if bad >= patience:
                break
        model.load_state_dict(best_state)
        oof[va] = _infer(model, Xd[va], Cd[va], device)
        sac_pred += _infer(model, Xs, Cs, device) / n_folds
        test_pred += _infer(model, Xt, Ct, device) / n_folds
        fold_rmse.append(best)
        print(f"  [nn] fold {f}: val_rmse {best:.3f}")
    return oof, sac_pred, test_pred, fold_rmse
