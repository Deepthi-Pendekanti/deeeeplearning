"""
CARD — Confidence-Adaptive Residual Dosing-aware forecaster.

A novel FUSED virtual-pH-sensing algorithm. Four components, designed together:

  (1) Physics-anchored ADAPTIVE residual skip
      pH_hat(t) = pH(t-1) + g(t) * delta(t)
      where delta(t) is a learned change and g(t) in [0,1] is a DOSING GATE
      produced from recent actuator activity (pH_reducer, add_water, ...).
      Quiet system  -> g->0 -> prediction anchored to persistence.
      Active dosing -> g->1 -> model free to deviate.
      (A plain delta head is the special case g==1.)

  (2) Tolerance-Weighted Huber (TWH) loss
      Zero cost inside a +/-tau dead-zone (errors that don't matter
      agronomically), Huber-shaped outside it. Directly optimizes the
      within-tolerance objective instead of raw MSE.

  (3) Self-supervised drift / uncertainty head
      The model outputs log-variance s(t); trained with a Gaussian NLL so it
      learns to predict its OWN error. At inference, sigma(t)=exp(0.5 s) is a
      calibrated per-step uncertainty used by the controller and as a drift
      alarm (predicted-sigma vs observed error divergence).

  (4) Confidence-Adaptive Sampling Controller (CASC)  [in card_eval.py]
      An online control law that adjusts the uncertainty threshold to hit a
      target physical-probe usage BUDGET while minimizing tolerance violations.

This file implements (1)-(3) as the model + loss. (4) is the inference policy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class CARDConfig:
    n_sensor: int = 6          # sensor feature count (pH is index target_index)
    n_actuator: int = 5        # actuator/dosing feature count
    hidden: int = 32
    layers: int = 2
    dropout: float = 0.1
    gate_hidden: int = 16      # dosing-gate MLP width
    target_index: int = 0      # pH column index within the sensor block
    uncertainty_head: bool = True


@dataclass
class CARDTrainConfig:
    epochs: int = 120
    batch: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    tau: float = 0.02          # TWH dead-zone half-width (SCALED pH units)
    huber_delta: float = 0.02  # TWH Huber transition (scaled units)
    nll_weight: float = 0.2    # weight of the uncertainty NLL term
    gate_prior_weight: float = 0.1  # (1) push gate to open with dosing activity
    grad_clip: float = 1.0
    patience: int = 15
    seed: int = 42


# ---------------------------------------------------------------------------
# (2) Tolerance-Weighted Huber loss
# ---------------------------------------------------------------------------
def tolerance_weighted_huber(pred, target, tau: float, delta: float,
                             reduction: str = "mean"):
    """
    Zero loss when |err| <= tau (dead-zone). Beyond tau, apply Huber on the
    EXCESS e = |err| - tau: quadratic up to delta, linear after.
    """
    err = torch.abs(pred - target)
    excess = torch.clamp(err - tau, min=0.0)
    quad = 0.5 * excess ** 2
    lin = delta * (excess - 0.5 * delta)
    loss = torch.where(excess <= delta, quad, lin)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


# ---------------------------------------------------------------------------
# (1)+(3) CARD model
# ---------------------------------------------------------------------------
class CARD(nn.Module):
    def __init__(self, cfg: CARDConfig):
        super().__init__()
        self.cfg = cfg
        n_in = cfg.n_sensor + cfg.n_actuator
        self.norm = nn.LayerNorm(n_in)
        self.gru = nn.GRU(
            input_size=n_in, hidden_size=cfg.hidden, num_layers=cfg.layers,
            batch_first=True, dropout=cfg.dropout if cfg.layers > 1 else 0.0,
        )
        # delta (change) head
        self.delta_head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden // 2), nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden // 2, 1),
        )
        # (1) dosing gate: from GRU state + recent actuator activity -> [0,1]
        self.gate_head = nn.Sequential(
            nn.Linear(cfg.hidden + cfg.n_actuator, cfg.gate_hidden), nn.ReLU(),
            nn.Linear(cfg.gate_hidden, 1),
        )
        # (3) uncertainty (log-variance) head
        if cfg.uncertainty_head:
            self.logvar_head = nn.Sequential(
                nn.Linear(cfg.hidden, cfg.gate_hidden), nn.ReLU(),
                nn.Linear(cfg.gate_hidden, 1),
            )

    def forward(self, x):
        """
        x: (B, T, n_sensor+n_actuator) scaled. Sensor block first, then actuators.
        Returns dict with pH_hat, gate, logvar (if enabled), delta.
        """
        cfg = self.cfg
        ns = cfg.n_sensor
        last_ph = x[:, -1, cfg.target_index]                 # persistence anchor
        recent_act = x[:, -1, ns:]                           # last actuator vec

        h = self.norm(x)
        out, _ = self.gru(h)
        z = out[:, -1, :]                                    # (B, hidden)

        delta = self.delta_head(z).squeeze(-1)               # learned change
        gate_in = torch.cat([z, recent_act], dim=-1)
        gate = torch.sigmoid(self.gate_head(gate_in)).squeeze(-1)   # (B,) in [0,1]

        ph_hat = last_ph + gate * delta                      # (1) adaptive skip

        result = {"pH_hat": ph_hat, "gate": gate, "delta": delta}
        if cfg.uncertainty_head:
            logvar = self.logvar_head(z).squeeze(-1)
            result["logvar"] = logvar
        return result


def gaussian_nll(pred, target, logvar):
    """Heteroscedastic Gaussian negative log-likelihood (for the sigma head).
    0.5 * ( (err^2)/var + logvar ). Clamp logvar for stability."""
    logvar = torch.clamp(logvar, min=-10.0, max=6.0)
    inv_var = torch.exp(-logvar)
    return (0.5 * (inv_var * (pred - target) ** 2 + logvar)).mean()


def gate_dosing_prior(gate, recent_act_sum):
    """
    Component (1) prior: encourage the gate to OPEN when dosing is active and
    CLOSE when the system is quiet. `recent_act_sum` is the summed actuator
    activity of the last step (>=0). We normalize it to a [0,1] target and
    apply a light BCE-style penalty so the gate is interpretable and physical.
    Only meaningful when some activity exists in the batch.
    """
    act = (recent_act_sum > 0).float()          # 1 if any dosing, else 0
    if act.sum() == 0 or act.sum() == act.numel():
        return gate.new_tensor(0.0)              # no signal in this batch
    eps = 1e-6
    g = gate.clamp(eps, 1 - eps)
    # target gate = 0.9 when dosing active, 0.1 when quiet
    tgt = 0.1 + 0.8 * act
    return -(tgt * torch.log(g) + (1 - tgt) * torch.log(1 - g)).mean()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def _set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def train_card(model: CARD, wd, tc: CARDTrainConfig, device="cpu",
               verbose=False):
    """Train CARD with TWH loss (+ optional NLL), early stopping on val TWH."""
    _set_seed(tc.seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr,
                            weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5,
                                                       patience=5)

    Xtr = torch.from_numpy(wd.X_train); ytr = torch.from_numpy(wd.y_train)
    tr = DataLoader(TensorDataset(Xtr, ytr), batch_size=tc.batch, shuffle=True)
    Xv = torch.from_numpy(wd.X_val).to(device)
    yv = torch.from_numpy(wd.y_val).to(device)

    best_val = float("inf"); best_state = None; bad = 0; hist = []
    for ep in range(1, tc.epochs + 1):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            o = model(xb)
            loss = tolerance_weighted_huber(o["pH_hat"], yb, tc.tau,
                                            tc.huber_delta)
            if "logvar" in o and tc.nll_weight > 0:
                loss = loss + tc.nll_weight * gaussian_nll(
                    o["pH_hat"].detach(), yb, o["logvar"])
            if tc.gate_prior_weight > 0:
                ns = model.cfg.n_sensor
                recent_act_sum = xb[:, -1, ns:].sum(dim=-1)
                loss = loss + tc.gate_prior_weight * gate_dosing_prior(
                    o["gate"], recent_act_sum)
            loss.backward()
            if tc.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()

        model.eval()
        with torch.no_grad():
            ov = model(Xv)
            vloss = tolerance_weighted_huber(ov["pH_hat"], yv, tc.tau,
                                             tc.huber_delta).item()
        hist.append(vloss); sched.step(vloss)
        if vloss < best_val - 1e-8:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if verbose:
            print(f"    ep{ep:3d} valTWH={vloss:.6e} best={best_val:.6e} bad={bad}")
        if bad >= tc.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_twh": best_val, "epochs_run": len(hist),
                   "val_curve": hist}


def predict(model: CARD, X: np.ndarray):
    """Return dict of numpy arrays: pH_hat, gate, sigma (if available)."""
    model.eval()
    with torch.no_grad():
        o = model(torch.from_numpy(X))
    out = {"pH_hat": o["pH_hat"].numpy(), "gate": o["gate"].numpy()}
    if "logvar" in o:
        out["sigma"] = np.exp(0.5 * o["logvar"].numpy())
    return out
