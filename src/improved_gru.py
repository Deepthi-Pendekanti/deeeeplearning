"""
Phase 5 — Improved GRU (configurable) + robust trainer.

The improved model is a flexible GRU that supports the knobs we search over:
  * number of layers, hidden size, dropout, recurrent depth
  * optional input LayerNorm (robust to feature scale/shift)
  * optional residual "delta" head: predict pH CHANGE (pH_t - pH_{t-1}) instead
    of absolute pH, then add back the last observed pH. This directly targets
    the persistence-dominance problem (Phase 1/2/4): the network only has to
    learn the *deviation* from persistence, which is the useful signal.

The trainer supports: Adam/AdamW, weight decay, ReduceLROnPlateau scheduler,
early stopping (best-on-val restore), gradient clipping, and Huber or MSE loss.

Nothing here touches the test set; selection uses validation only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class GRUConfig:
    n_features: int = 6
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.2
    input_layernorm: bool = True
    delta_head: bool = True          # predict pH change, add last pH back
    bidirectional: bool = False


class ImprovedGRU(nn.Module):
    def __init__(self, cfg: GRUConfig, target_index: int = 0):
        super().__init__()
        self.cfg = cfg
        self.target_index = target_index
        self.norm = (nn.LayerNorm(cfg.n_features)
                     if cfg.input_layernorm else nn.Identity())
        self.gru = nn.GRU(
            input_size=cfg.n_features, hidden_size=cfg.hidden,
            num_layers=cfg.layers, batch_first=True,
            dropout=cfg.dropout if cfg.layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
        )
        out_dim = cfg.hidden * (2 if cfg.bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(out_dim // 2, 1),
        )

    def forward(self, x):
        # x: (batch, seq, features), scaled. Last pH (scaled) is the persistence
        # reference for the delta head.
        last_ph = x[:, -1, self.target_index]
        h = self.norm(x)
        out, _ = self.gru(h)
        y = self.head(out[:, -1, :]).squeeze(-1)
        if self.cfg.delta_head:
            return last_ph + y      # persistence + learned correction
        return y


@dataclass
class TrainConfig:
    epochs: int = 100
    batch: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"        # adam | adamw
    loss: str = "huber"             # huber | mse
    huber_delta: float = 0.01       # in scaled pH units
    grad_clip: float = 1.0
    patience: int = 12              # early stopping on val
    scheduler: bool = True
    seed: int = 42


def _set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def train_improved(model: nn.Module, wd, tc: TrainConfig, device="cpu",
                   verbose=False):
    """Train with early stopping on validation; restore best weights. Returns
    (model, history) and leaves the test set untouched."""
    _set_seed(tc.seed)
    model = model.to(device)

    opt_cls = torch.optim.AdamW if tc.optimizer == "adamw" else torch.optim.Adam
    opt = opt_cls(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(
                 opt, mode="min", factor=0.5, patience=4)
             if tc.scheduler else None)
    loss_fn = (nn.HuberLoss(delta=tc.huber_delta) if tc.loss == "huber"
               else nn.MSELoss())

    tr = DataLoader(TensorDataset(torch.from_numpy(wd.X_train),
                                  torch.from_numpy(wd.y_train)),
                    batch_size=tc.batch, shuffle=True)
    Xv = torch.from_numpy(wd.X_val).to(device)
    yv = torch.from_numpy(wd.y_val).to(device)

    best_val = float("inf")
    best_state = None
    bad = 0
    history = []
    for ep in range(1, tc.epochs + 1):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            if tc.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xv), yv).item()
        history.append(vloss)
        if sched:
            sched.step(vloss)
        if vloss < best_val - 1e-7:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if verbose:
            print(f"    ep{ep:3d} val={vloss:.6f} best={best_val:.6f} bad={bad}")
        if bad >= tc.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val": best_val, "epochs_run": len(history),
                   "val_curve": history}
