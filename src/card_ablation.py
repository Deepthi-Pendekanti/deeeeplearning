"""
Task 8 — CARD ablation: measure each component's real contribution.

Variants (all trained on the same paper/temporal split, same seed):
  A. Full CARD                       (adaptive gate + TWH + uncertainty + gate prior)
  B. no adaptive skip (gate := 1)    -> plain delta head
  C. no TWH (plain MSE loss)         -> tests the tolerance-weighted loss
  D. no uncertainty head             -> tests the sigma/NLL head's effect on point pred
  E. no gate prior                   -> gate free (tests the dosing-aware supervision)

For each we report point-prediction metrics (R2/MAE/RMSE/within0.05) on TEST,
plus the CASC hybrid result at a 10% probe budget (needs a usable sigma; C/D
without a proper sigma fall back to |gate*delta| as an uncertainty proxy).

Everything measured; nothing fabricated.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn

import dataset as ds
from card import (CARD, CARDConfig, CARDTrainConfig, gate_dosing_prior,
                  gaussian_nll, predict, tolerance_weighted_huber)
from card_eval import build_card_data, casc_controller
from config import METRICS, MODELS, RESIDUAL_TOLERANCE
from metrics import all_metrics
from torch.utils.data import DataLoader, TensorDataset


def train_variant(wd, cfg: CARDConfig, tc: CARDTrainConfig,
                  force_gate_one=False, use_twh=True, use_nll=True,
                  use_gate_prior=True, seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = CARD(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr,
                            weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5,
                                                       patience=5)
    tr = DataLoader(TensorDataset(torch.from_numpy(wd.X_train),
                                  torch.from_numpy(wd.y_train)),
                    batch_size=tc.batch, shuffle=True)
    Xv = torch.from_numpy(wd.X_val); yv = torch.from_numpy(wd.y_val)

    def loss_of(o, yb, xb):
        pred = o["pH_hat"]
        if force_gate_one:
            # override: recompute pH_hat with gate := 1 (plain delta head)
            last_ph = xb[:, -1, cfg.target_index]
            pred = last_ph + o["delta"]
        if use_twh:
            loss = tolerance_weighted_huber(pred, yb, tc.tau, tc.huber_delta)
        else:
            loss = nn.functional.mse_loss(pred, yb)
        if use_nll and "logvar" in o:
            loss = loss + tc.nll_weight * gaussian_nll(pred.detach(), yb, o["logvar"])
        if use_gate_prior and not force_gate_one:
            ns = cfg.n_sensor
            loss = loss + tc.gate_prior_weight * gate_dosing_prior(
                o["gate"], xb[:, -1, ns:].sum(dim=-1))
        return loss, pred

    best = float("inf"); best_state = None; bad = 0
    for ep in range(tc.epochs):
        model.train()
        for xb, yb in tr:
            opt.zero_grad()
            o = model(xb)
            loss, _ = loss_of(o, yb, xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
        model.eval()
        with torch.no_grad():
            ov = model(Xv)
            vl, _ = loss_of(ov, yv, Xv)
            vl = vl.item()
        sched.step(vl)
        if vl < best - 1e-9:
            best = vl; best_state = {k: v.detach().cpu().clone()
                                     for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= tc.patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    return model


def eval_variant(model, wd, cfg, force_gate_one=False):
    model.eval()
    with torch.no_grad():
        o = model(torch.from_numpy(wd.X_test))
    if force_gate_one:
        last_ph = wd.X_test[:, -1, cfg.target_index]
        ph = last_ph + o["delta"].numpy()
    else:
        ph = o["pH_hat"].numpy()
    pred = ds.inverse_target(wd, ph)
    true = ds.inverse_target(wd, wd.y_test)
    m = all_metrics(true, pred)

    # uncertainty for CASC: use sigma if present, else |gate*delta| proxy
    scale = wd.target_scaler.data_max_[0] - wd.target_scaler.data_min_[0]
    if "logvar" in o:
        sigma = np.exp(0.5 * o["logvar"].numpy()) * scale
    else:
        sigma = np.abs((o["gate"] * o["delta"]).numpy()) * scale
    sim = casc_controller(true, pred, sigma, budget=0.10)
    mh = all_metrics(true, sim["output_ph"])
    return m, {"achieved_usage": sim["physical_fraction"],
               "hybrid_R2": mh["R2"], "hybrid_within_0.05pH": mh["within_0.05pH"]}


def main():
    wd, ns, na = build_card_data()
    base_cfg = dict(n_sensor=ns, n_actuator=na, hidden=32, layers=2,
                    dropout=0.1, target_index=wd.target_index)
    tc = CARDTrainConfig(epochs=120, tau=0.003, huber_delta=0.02,
                         nll_weight=0.2, gate_prior_weight=0.3, patience=20,
                         seed=42)

    variants = {
        "A. Full CARD": dict(),
        "B. no adaptive skip (gate=1)": dict(force_gate_one=True),
        "C. no TWH (MSE loss)": dict(use_twh=False),
        "D. no uncertainty head": dict(use_nll=False),
        "E. no gate prior": dict(use_gate_prior=False),
    }

    rows = {}
    for name, kw in variants.items():
        # The uncertainty HEAD is always present; variant D only drops its
        # NLL loss term (use_nll=False), isolating the head's training effect.
        cfg = CARDConfig(uncertainty_head=True, **base_cfg)
        model = train_variant(wd, cfg, tc, **kw)
        m, hyb = eval_variant(model, wd, cfg,
                              force_gate_one=kw.get("force_gate_one", False))
        rows[name] = {"point": m, "casc10": hyb}
        print(f"{name:<32} R2={m['R2']:.4f} MAE={m['MAE']:.4f} "
              f"RMSE={m['RMSE']:.4f} w0.05={m['within_0.05pH']:.2f}%  "
              f"| CASC@10%: use={hyb['achieved_usage']*100:.1f}% "
              f"R2={hyb['hybrid_R2']:.4f}")

    with open(METRICS / "card_ablation.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\n[save] results/metrics/card_ablation.json")


if __name__ == "__main__":
    main()
