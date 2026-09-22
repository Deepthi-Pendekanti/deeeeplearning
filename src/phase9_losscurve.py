"""Retrain the Phase-5 winning config once, recording train+val loss curves,
and plot them (Phase 9 item: training vs validation loss)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import dataset as ds
from config import FIGURES, METRICS, MODELS
from improved_gru import GRUConfig, ImprovedGRU, TrainConfig


def main():
    ckpt = torch.load(MODELS / "improved_gru_best.pt", map_location="cpu")
    gcfg = GRUConfig(**ckpt["gru_cfg"])
    tc = TrainConfig(**ckpt["train_cfg"])

    wd = ds.build(include_actuators=False, source="paper", split="temporal",
                  verbose=False)
    import random
    random.seed(tc.seed); np.random.seed(tc.seed); torch.manual_seed(tc.seed)

    model = ImprovedGRU(gcfg, target_index=wd.target_index)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr,
                            weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5,
                                                       patience=4)
    loss_fn = nn.HuberLoss(delta=tc.huber_delta)
    tr = DataLoader(TensorDataset(torch.from_numpy(wd.X_train),
                                  torch.from_numpy(wd.y_train)),
                    batch_size=tc.batch, shuffle=True)
    Xv = torch.from_numpy(wd.X_val); yv = torch.from_numpy(wd.y_val)

    train_curve, val_curve = [], []
    best = float("inf"); bad = 0
    for ep in range(tc.epochs):
        model.train(); run = 0; nb = 0
        for xb, yb in tr:
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step(); run += loss.item() * len(xb); nb += len(xb)
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv), yv).item()
        train_curve.append(run / nb); val_curve.append(vl); sched.step(vl)
        if vl < best - 1e-7:
            best = vl; bad = 0
        else:
            bad += 1
        if bad >= tc.patience:
            break

    plt.figure(figsize=(7, 4.5))
    plt.plot(train_curve, label="train loss (Huber)")
    plt.plot(val_curve, label="val loss (Huber)")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title("Improved GRU — training vs validation loss (early-stopped)")
    plt.tight_layout(); plt.savefig(FIGURES / "improved_loss_curve.png"); plt.close()

    with open(METRICS / "phase9_losscurve.json", "w") as f:
        json.dump({"train": train_curve, "val": val_curve}, f, indent=2)
    print(f"[ok] loss curve ({len(train_curve)} epochs) -> "
          f"improved_loss_curve.png")


if __name__ == "__main__":
    main()
