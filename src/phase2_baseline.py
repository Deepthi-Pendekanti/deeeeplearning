"""
Phase 2 — Clean reproducible baseline with extended metrics.

Reproduces the GRU baseline under a fully documented, fixed-seed protocol and
reports the extended metric set requested:
  R2, MAE, RMSE, MAPE, MedAE, MaxAE,
  within +-0.01 / 0.03 / 0.05 / 0.10 pH.

Also computes a PERSISTENCE baseline (predict pH_t = pH_{t-1}) so the GRU's
skill can be judged relative to the trivial predictor (the task is highly
autocorrelated; see Phase 1 audit point B).

Protocol (documented):
  * Source     : paper subset (clean_paper_subset.csv)
  * Features   : 6 sensors (pH, TDS, water_level, DHT_temp, DHT_humidity, water_temp)
  * Window     : look-back 10, one-step-ahead (t+1)
  * Scaling    : MinMax, fit on first 80% only
  * Split      : passed via --split {random,chrono,temporal}
  * Training   : 30 epochs, batch 32, Adam lr 1e-3, MSE, seed 42
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import dataset as ds
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, METRICS
from metrics import all_metrics
from models import build_model, count_parameters


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def persistence_baseline(wd):
    """Predict next pH = last observed pH in the window (scaled -> physical)."""
    last_ph_scaled = wd.X_test[:, -1, wd.target_index]  # pH at t-1 (scaled)
    pred = ds.inverse_target(wd, last_ph_scaled)
    true = ds.inverse_target(wd, wd.y_test)
    return true, pred


def train_gru(wd, seed=42, device="cpu"):
    set_seed(seed)
    model = build_model("gru", wd.n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    tr = DataLoader(TensorDataset(torch.from_numpy(wd.X_train),
                                  torch.from_numpy(wd.y_train)),
                    batch_size=BATCH_SIZE, shuffle=True)
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred_s = model(torch.from_numpy(wd.X_test).to(device)).cpu().numpy()
    return model, ds.inverse_target(wd, wd.y_test), ds.inverse_target(wd, pred_s)


def report(name, true, pred, extra=None):
    m = all_metrics(true, pred)
    if extra:
        m.update(extra)
    print(f"\n=== {name} ===")
    print(f"  R2   = {m['R2']:.4f}")
    print(f"  MAE  = {m['MAE']:.4f} pH")
    print(f"  RMSE = {m['RMSE']:.4f} pH")
    print(f"  MAPE = {m['MAPE']:.3f} %")
    print(f"  MedAE= {m['MedAE']:.4f} pH   MaxAE={m['MaxAE']:.4f} pH")
    print(f"  within +-0.01 pH: {m['within_0.01pH']:.2f}%")
    print(f"  within +-0.03 pH: {m['within_0.03pH']:.2f}%")
    print(f"  within +-0.05 pH: {m['within_0.05pH']:.2f}%")
    print(f"  within +-0.10 pH: {m['within_0.10pH']:.2f}%")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["random", "chrono", "temporal"],
                    default="random")
    ap.add_argument("--source", choices=["full", "paper"], default="paper")
    args = ap.parse_args()

    print(f"[phase2] source={args.source} split={args.split}")
    wd = ds.build(include_actuators=False, source=args.source,
                  split=args.split, verbose=True)

    # Persistence baseline.
    pt, pp = persistence_baseline(wd)
    m_persist = report("Persistence baseline (predict last pH)", pt, pp)

    # GRU baseline.
    model, gt, gp = train_gru(wd)
    m_gru = report("GRU baseline (reproduced)", gt, gp,
                   extra={"params": count_parameters(model)})

    out = {
        "protocol": {
            "source": args.source, "split": args.split,
            "features": wd.feature_names, "lookback": 10, "horizon": 1,
            "epochs": EPOCHS, "batch": BATCH_SIZE, "lr": LEARNING_RATE,
            "loss": "MSE", "seed": 42,
        },
        "persistence": m_persist,
        "gru_baseline": m_gru,
    }
    path = METRICS / f"phase2_baseline_{args.source}_{args.split}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[save] {path}")


if __name__ == "__main__":
    main()
