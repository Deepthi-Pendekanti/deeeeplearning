"""
Stage 9: Training pipeline (Section 3.3.10).

Exact settings from the paper:
  * 30 epochs, batch size 32
  * Adam optimizer, learning rate 0.001
  * MSE loss
  * one-step-ahead pH prediction, look-back 10
  * each model trained 3 times (3 seeds); we report mean +/- std and keep the
    best run (by validation loss) for plotting/deployment.

Metrics (RMSE/MAE/R2) are computed on the TEST set in physical pH units by
inverse-transforming predictions with the target MinMax scaler.

Usage:
  python src/train.py                 # trains GRU, LSTM, Transformer (6-feat)
  python src/train.py --models gru
  python src/train.py --features 8    # use the 8-feature Table A.9 config
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import dataset as ds
from config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    METRICS,
    MODELS,
    N_RUNS,
    PAPER_TARGETS,
    SEEDS,
)
from metrics import all_metrics
from models import build_model, count_parameters


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)  # GRU/LSTM lack deterministic CUDA kernels


def make_loaders(wd: ds.WindowedData):
    def loader(X, y, shuffle):
        tX = torch.from_numpy(X)
        ty = torch.from_numpy(y)
        return DataLoader(TensorDataset(tX, ty), batch_size=BATCH_SIZE,
                          shuffle=shuffle, drop_last=False)
    # Time-series: do NOT shuffle val/test; training may shuffle windows.
    return (loader(wd.X_train, wd.y_train, True),
            loader(wd.X_val, wd.y_val, False),
            loader(wd.X_test, wd.y_test, False))


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            preds.append(np.atleast_1d(out))
            trues.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def measure_latency(model, wd, device, n=200):
    """Average single-sample inference latency in seconds (like Table 5)."""
    model.eval()
    x = torch.from_numpy(wd.X_test[:1]).to(device)
    # warm-up
    with torch.no_grad():
        for _ in range(20):
            model(x)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            model(x)
    return (time.perf_counter() - t0) / n


def train_one(name: str, wd: ds.WindowedData, seed: int, device):
    set_seed(seed)
    train_loader, val_loader, test_loader = make_loaders(wd)

    model = build_model(name, wd.n_features).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optim.step()
            running += loss.item() * len(xb)
            n_seen += len(xb)
        train_loss = running / max(n_seen, 1)

        # validation
        model.eval()
        vloss = 0.0
        vn = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vloss += loss_fn(model(xb), yb).item() * len(xb)
                vn += len(xb)
        val_loss = vloss / max(vn, 1)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        print(f"    [{name} seed={seed}] epoch {epoch:2d}/{EPOCHS} "
              f"train={train_loss:.6f} val={val_loss:.6f}", flush=True)

    # The paper trains for a FIXED 30 epochs with no early stopping, then
    # evaluates the resulting model. We therefore evaluate the FINAL-epoch
    # weights (not best-on-val). best_val is still tracked to pick which of the
    # 3 seed runs to keep for plotting/deployment.
    final_val = history[-1]["val_loss"]

    # Test metrics in physical pH units.
    pred_s, true_s = evaluate(model, test_loader, device)
    pred = ds.inverse_target(wd, pred_s)
    true = ds.inverse_target(wd, true_s)
    m = all_metrics(true, pred)
    m["val_loss"] = final_val
    m["best_val_loss"] = best_val
    m["latency_s"] = measure_latency(model, wd, device)
    m["params"] = count_parameters(model)

    return model, m, history, (true, pred)


def train_model(name: str, wd: ds.WindowedData, device):
    print(f"\n=== Training {name.upper()} ({N_RUNS} runs) ===")
    runs = []
    best = None  # (val_loss, model_state, arrays)
    for seed in SEEDS[:N_RUNS]:
        model, m, hist, arrays = train_one(name, wd, seed, device)
        runs.append(m)
        print(f"  -> seed={seed}: RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} "
              f"R2={m['R2']:.4f} latency={m['latency_s']*1000:.2f}ms")
        if best is None or m["val_loss"] < best[0]:
            best = (m["val_loss"],
                    {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    arrays, m)

    # Aggregate mean +/- std across runs.
    keys = ["RMSE", "MAE", "R2", "MAPE", "within_5pct", "within_10pct",
            "within_15pct", "within_0.05pH", "latency_s"]
    agg = {k: {"mean": float(np.mean([r[k] for r in runs])),
               "std": float(np.std([r[k] for r in runs]))} for k in keys}
    agg["params"] = runs[0]["params"]
    agg["runs"] = runs

    # Save the best run's model weights and its test predictions.
    best_val, best_state, (true, pred), best_metrics = best
    torch.save(best_state, MODELS / f"{name}_best.pt")
    np.savez(MODELS / f"{name}_test_predictions.npz", y_true=true, y_pred=pred)
    agg["best_run"] = best_metrics
    print(f"  BEST {name}: RMSE={best_metrics['RMSE']:.4f} "
          f"MAE={best_metrics['MAE']:.4f} R2={best_metrics['R2']:.4f}")

    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["gru", "lstm", "transformer"])
    parser.add_argument("--features", type=int, choices=[6, 8], default=6,
                        help="6 = all sensors (primary); 8 = Table A.9 config.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override EPOCHS (for quick smoke tests)")
    parser.add_argument("--runs", type=int, default=None,
                        help="override N_RUNS (for quick smoke tests)")
    parser.add_argument("--source", choices=["full", "paper"], default="full",
                        help="full = our cleaned month; paper = authors' subset.")
    parser.add_argument("--split", choices=["chrono", "random"], default="chrono",
                        help="chrono = strict time split; random = window-level "
                             "random split (reproduces paper's joint RMSE/R2).")
    args = parser.parse_args()

    # Optional overrides for fast iteration.
    global EPOCHS, N_RUNS
    if args.epochs is not None:
        EPOCHS = args.epochs
    if args.runs is not None:
        N_RUNS = args.runs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Build the windowed dataset.
    if args.features == 8:
        # 8-feature = 6 sensors + pH_reducer + add_water (matches 24,351 params).
        wd = ds.build(include_actuators=True,
                      exclude=["nutrients_adder", "humidifier", "ex_fan"],
                      source=args.source, split=args.split)
    else:
        wd = ds.build(include_actuators=False, source=args.source,
                      split=args.split)  # 6 sensors

    print(f"[config] feature set ({args.features}): {wd.feature_names}")

    results = {}
    for name in args.models:
        results[name] = train_model(name, wd, device)

    # Persist the summary.
    out = {
        "feature_config": args.features,
        "feature_names": wd.feature_names,
        "paper_targets": PAPER_TARGETS,
        "results": results,
    }
    out_path = (METRICS /
                f"training_results_{args.source}_{args.features}feat_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[save] {out_path}")

    # Console summary vs paper.
    print("\n" + "=" * 68)
    print(f"SUMMARY (feature config = {args.features}) — test set, physical pH")
    print("=" * 68)
    print(f"{'Model':<12}{'RMSE':>10}{'MAE':>10}{'R2':>10}{'params':>10}")
    for name, agg in results.items():
        b = agg["best_run"]
        print(f"{name:<12}{b['RMSE']:>10.4f}{b['MAE']:>10.4f}"
              f"{b['R2']:>10.4f}{agg['params']:>10,}")
    print("\nPaper targets:")
    for name, t in PAPER_TARGETS.items():
        print(f"{name:<12}{t['RMSE']:>10.4f}{t['MAE']:>10.4f}{t['R2']:>10.4f}")


if __name__ == "__main__":
    main()
