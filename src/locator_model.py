"""Learned locator (Phase-2): siamese GR-window encoder + locality-constrained soft cross-attention,
trained end-to-end on TVT drift. See docs/locator_design.md.

Local tests showed: (1) unconstrained matching is garbage (GR repeats across the column → ambiguity),
(2) even locality-capped RAW matching is below floor. So the model must EARN signal via (a) a learned
encoder (better-than-L2 similarity), (b) SOFT attention (variance reduction vs hard pick), (c) a soft
LOCALITY bias anchoring matches near the last-known TVT. Episodic by well; trained on drift.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .locator_data import Episode


class WinEncoder(nn.Module):
    """Shared siamese encoder: GR window [B,1,L] → embedding [B,d]."""
    def __init__(self, d=64, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1))
        self.proj = nn.Sequential(nn.Flatten(), nn.Dropout(drop), nn.Linear(64, d))

    def forward(self, w):                       # w: [B, L]
        return self.proj(self.net(w.unsqueeze(1)))   # [B, d]


class Locator(nn.Module):
    """drift_i = softmax_j( q_i·k_j/√d − λ·val_j² ) · val_j  (+ tiny residual head on the attended drift)."""
    def __init__(self, d=64, drop=0.1, loc_lambda=3e-4):
        super().__init__()
        self.enc = WinEncoder(d, drop)
        self.scale = d ** 0.5
        self.log_lambda = nn.Parameter(torch.tensor(float(np.log(loc_lambda))))  # learnable locality strength
        self.head = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))  # refine (attn_drift, attn_entropy)

    def forward(self, q_gr, mem_gr, mem_val, chunk=4096):
        k = self.enc(mem_gr)                                   # [M, d]
        v = mem_val                                            # [M]
        lam = torch.exp(self.log_lambda)
        loc = -lam * v.pow(2)                                  # [M] locality bias (penalise far-from-anchor)
        outs = []
        for s in range(0, len(q_gr), chunk):                   # chunk queries to bound memory
            q = self.enc(q_gr[s:s + chunk])                    # [b, d]
            logit = q @ k.t() / self.scale + loc[None, :]      # [b, M]
            a = torch.softmax(logit, 1)
            attn_drift = a @ v                                 # [b]
            ent = -(a * (a + 1e-9).log()).sum(1)               # [b] attention entropy (confidence)
            ref = self.head(torch.stack([attn_drift, ent], 1)).squeeze(1)
            outs.append(attn_drift + ref)                      # residual refinement
        return torch.cat(outs)


def _aug(gr, train, noise=0.05, jitter=2):
    if not train:
        return gr
    g = gr + torch.randn_like(gr) * noise
    if jitter:
        g = torch.roll(g, int(torch.randint(-jitter, jitter + 1, (1,))), dims=-1)
    return g


def train_cv(dev_eps, sac_eps, test_eps, n_folds=5, epochs=25, lr=1e-3, d=64,
             patience=5, device=None, seed=7, mem_sub=4000):
    """By-well CV over episodes → (oof dict{id:pred}, sac dict, test dict, fold_rmse).
    mem_sub: cap memory slots per episode (subsample) for speed/regularisation."""
    import gc
    from sklearn.model_selection import GroupKFold
    from .evaluate import rmse
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    wells = np.array([e.well for e in dev_eps])

    def subset(e):
        if e.mem_gr.shape[0] <= mem_sub:
            return e.mem_gr, e.mem_val
        idx = rng.choice(e.mem_gr.shape[0], mem_sub, replace=False)
        return e.mem_gr[idx], e.mem_val[idx]

    def infer(model, eps, train=False):
        out = {}
        model.train(train)
        for e in eps:
            mg, mv = subset(e)
            mg_t = _aug(torch.from_numpy(mg).to(device), train)
            qg_t = _aug(torch.from_numpy(e.q_gr).to(device), train)
            p = model(qg_t, mg_t, torch.from_numpy(mv).to(device))
            out[e.well] = p
        return out

    oof, sac, test, fr = {}, {}, {}, []
    for f, (tr, va) in enumerate(GroupKFold(n_folds).split(wells, groups=wells)):
        tr_eps = [dev_eps[i] for i in tr]; va_eps = [dev_eps[i] for i in va]
        model = Locator(d=d).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        best, best_state, bad = 1e9, None, 0
        for ep in range(epochs):
            model.train(); rng.shuffle(tr_eps); opt.zero_grad(); acc = 0
            for k, e in enumerate(tr_eps):
                mg, mv = subset(e)
                y = torch.from_numpy(e.q_y).to(device); m = torch.isfinite(y)
                if m.sum() == 0:
                    continue
                pred = model(_aug(torch.from_numpy(e.q_gr).to(device), True),
                             _aug(torch.from_numpy(mg).to(device), True),
                             torch.from_numpy(mv).to(device))
                loss = nn.functional.mse_loss(pred[m], y[m]) / 4
                loss.backward(); acc += 1
                if acc % 4 == 0:                              # grad-accum over wells → stable step
                    opt.step(); opt.zero_grad()
            opt.step(); opt.zero_grad()
            # val
            model.eval(); vy, vp = [], []
            with torch.no_grad():
                for e in va_eps:
                    mg, mv = subset(e)
                    p = model(torch.from_numpy(e.q_gr).to(device), torch.from_numpy(mg).to(device),
                              torch.from_numpy(mv).to(device)).cpu().numpy()
                    m = np.isfinite(e.q_y); vy.append(e.q_y[m]); vp.append(p[m])
            vr = rmse(np.concatenate(vy), np.concatenate(vp))
            if vr < best - 1e-3:
                best, bad = vr, 0; best_state = {kk: vv.cpu().clone() for kk, vv in model.state_dict().items()}
            else:
                bad += 1
            if bad >= patience:
                break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            for e in va_eps:
                mg, mv = subset(e)
                p = model(torch.from_numpy(e.q_gr).to(device), torch.from_numpy(mg).to(device),
                          torch.from_numpy(mv).to(device)).cpu().numpy()
                oof.update(dict(zip(e.q_ids, p)))
            for e in sac_eps:
                mg, mv = subset(e)
                p = model(torch.from_numpy(e.q_gr).to(device), torch.from_numpy(mg).to(device),
                          torch.from_numpy(mv).to(device)).cpu().numpy()
                for i, pid in enumerate(e.q_ids):
                    sac[pid] = sac.get(pid, 0.0) + p[i] / n_folds
            for e in test_eps:
                mg, mv = subset(e)
                p = model(torch.from_numpy(e.q_gr).to(device), torch.from_numpy(mg).to(device),
                          torch.from_numpy(mv).to(device)).cpu().numpy()
                for i, pid in enumerate(e.q_ids):
                    test[pid] = test.get(pid, 0.0) + p[i] / n_folds
        fr.append(best); print(f"  [loc] fold {f}: val_rmse {best:.3f}")
        del model; gc.collect()
    return oof, sac, test, fr
