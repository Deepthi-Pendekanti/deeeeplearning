"""
Phase 4 — Strong baselines under one identical test protocol.

Models: Persistence, LinearRegression, RandomForest, ExtraTrees, XGBoost,
        LSTM, GRU. (Improved GRU is added in Phase 5.)

Every model uses the SAME windowed data, SAME scaling, SAME split. Classical/
tree models consume the flattened window (lookback x n_features). Recurrent
models consume the 3-D window. Metrics are computed in physical pH units.

We run BOTH splits so generalization is visible:
  * random  : leaky (reproduces paper-style numbers)
  * temporal: leakage-free future test (the honest estimate)

Outputs a comparison table with R2/MAE/RMSE/+-0.05pH/params/inference-time and
saves results/metrics/phase4_baselines_<split>.json.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

import dataset as ds
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, METRICS
from metrics import all_metrics
from models import build_model, count_parameters


def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def flatten(X):
    return X.reshape(X.shape[0], -1)


def time_predict_sklearn(model, x1, n=100):
    for _ in range(10):
        model.predict(x1)
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(x1)
    return (time.perf_counter() - t0) / n


def eval_persistence(wd):
    last = wd.X_test[:, -1, wd.target_index]
    pred = ds.inverse_target(wd, last)
    true = ds.inverse_target(wd, wd.y_test)
    m = all_metrics(true, pred)
    m.update({"params": 0, "latency_ms": 0.0})
    return m, true, pred


def eval_sklearn(name, model, wd):
    Xtr, Xte = flatten(wd.X_train), flatten(wd.X_test)
    t0 = time.perf_counter()
    model.fit(Xtr, wd.y_train)
    fit_t = time.perf_counter() - t0
    pred = ds.inverse_target(wd, model.predict(Xte))
    true = ds.inverse_target(wd, wd.y_test)
    m = all_metrics(true, pred)
    # parameter count proxy for tree/linear models
    if hasattr(model, "coef_"):
        params = int(np.size(model.coef_) + np.size(getattr(model, "intercept_", 0)))
    else:
        params = None
    m.update({"params": params, "fit_time_s": fit_t,
              "latency_ms": time_predict_sklearn(model, Xte[:1]) * 1000})
    return m, true, pred


def eval_torch(name, wd, seed=42, device="cpu"):
    set_seed(seed)
    model = build_model(name, wd.n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    tr = DataLoader(TensorDataset(torch.from_numpy(wd.X_train),
                                  torch.from_numpy(wd.y_train)),
                    batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = ds.inverse_target(wd, model(torch.from_numpy(wd.X_test)).numpy())
    true = ds.inverse_target(wd, wd.y_test)
    m = all_metrics(true, pred)
    # latency
    x1 = torch.from_numpy(wd.X_test[:1])
    with torch.no_grad():
        for _ in range(20):
            model(x1)
        t0 = time.perf_counter()
        for _ in range(200):
            model(x1)
        lat = (time.perf_counter() - t0) / 200
    m.update({"params": count_parameters(model), "latency_ms": lat * 1000})
    return m, true, pred


def run(split, source="paper"):
    wd = ds.build(include_actuators=False, source=source, split=split,
                  verbose=False)
    results = {}

    m, *_ = eval_persistence(wd); results["Persistence"] = m
    m, *_ = eval_sklearn("LinearRegression", LinearRegression(), wd)
    results["LinearRegression"] = m
    m, *_ = eval_sklearn("RandomForest",
                         RandomForestRegressor(n_estimators=200, n_jobs=-1,
                                               random_state=42), wd)
    results["RandomForest"] = m
    m, *_ = eval_sklearn("ExtraTrees",
                         ExtraTreesRegressor(n_estimators=200, n_jobs=-1,
                                             random_state=42), wd)
    results["ExtraTrees"] = m
    m, *_ = eval_sklearn("XGBoost",
                         XGBRegressor(n_estimators=300, max_depth=6,
                                      learning_rate=0.1, random_state=42,
                                      verbosity=0), wd)
    results["XGBoost"] = m
    m, *_ = eval_torch("lstm", wd); results["LSTM"] = m
    m, *_ = eval_torch("gru", wd); results["GRU"] = m
    return results


def print_table(results, split):
    print(f"\n{'='*82}\nPhase 4 baselines — split={split} (test set, physical pH)\n{'='*82}")
    print(f"{'Model':<18}{'R2':>9}{'MAE':>9}{'RMSE':>9}{'±0.05pH%':>11}"
          f"{'params':>10}{'lat(ms)':>10}")
    for name, m in results.items():
        p = m['params'] if m['params'] is not None else -1
        print(f"{name:<18}{m['R2']:>9.4f}{m['MAE']:>9.4f}{m['RMSE']:>9.4f}"
              f"{m['within_0.05pH']:>11.2f}{p:>10}{m['latency_ms']:>10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+",
                    default=["random", "temporal"])
    ap.add_argument("--source", choices=["full", "paper"], default="paper")
    args = ap.parse_args()

    all_out = {}
    for split in args.splits:
        res = run(split, args.source)
        print_table(res, split)
        all_out[split] = res
        with open(METRICS / f"phase4_baselines_{args.source}_{split}.json", "w") as f:
            json.dump(res, f, indent=2)
    print(f"\n[save] results/metrics/phase4_baselines_{args.source}_*.json")


if __name__ == "__main__":
    main()
