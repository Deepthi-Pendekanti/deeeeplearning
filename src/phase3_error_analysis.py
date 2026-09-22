"""
Phase 3 — Investigate the RMSE >> MAE gap.

Retrains the GRU on the paper subset (random split, matching the reproduction),
then locates the largest absolute errors and joins each back to its row context
(timestamp if available, actual/predicted pH, sensor values, actuator states).

Produces:
  * Top-20 largest-error table (console + CSV).
  * A decomposition showing how much of the total squared error (RMSE^2) is
    contributed by the worst k predictions -> confirms/refutes "few outliers".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import dataset as ds
from config import (BATCH_SIZE, DATA_PROCESSED, EPOCHS, LEARNING_RATE, LOOKBACK,
                    METRICS)
from metrics import rmse
from models import build_model


def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def train_and_predict(split="random", source="paper"):
    wd = ds.build(include_actuators=False, source=source, split=split,
                  verbose=False)
    set_seed(42)
    model = build_model("gru", wd.n_features)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    tr = DataLoader(TensorDataset(torch.from_numpy(wd.X_train),
                                  torch.from_numpy(wd.y_train)),
                    batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in tr:
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred_s = model(torch.from_numpy(wd.X_test)).numpy()
    pred = ds.inverse_target(wd, pred_s)
    true = ds.inverse_target(wd, wd.y_test)
    return wd, true, pred


def main():
    # Use the reproduction config (random split) so we analyze the same test
    # predictions that produced the reported RMSE.
    wd, true, pred = train_and_predict(split="random", source="paper")
    err = np.abs(true - pred)

    print(f"[phase3] test samples: {len(err)}")
    print(f"[phase3] RMSE={rmse(true,pred):.4f}  MAE={np.mean(err):.4f}  "
          f"MedAE={np.median(err):.4f}  MaxAE={err.max():.4f}")

    # --- How concentrated is the squared error? ---
    sq = err ** 2
    total_sq = sq.sum()
    order = np.argsort(err)[::-1]
    print("\nShare of total squared error (drives RMSE) from the worst-k:")
    for k in [1, 5, 10, 20, 50]:
        share = sq[order[:k]].sum() / total_sq * 100
        rmse_wo = np.sqrt((total_sq - sq[order[:k]].sum()) / (len(err) - k))
        print(f"  worst {k:>3d}: {share:5.1f}% of SSE | RMSE without them = "
              f"{rmse_wo:.4f}")

    # --- Join top-20 errors to row context ---
    # The random split shuffles windows; map each test window back to its source
    # row so we can show sensor/actuator context. Rebuild the same shuffle.
    df = ds.load_clean(source="paper")
    n = len(df)
    n_train_full = int(n * 0.8)
    # window i predicts row (i + LOOKBACK). Recreate the random test indices.
    from config import TRAIN_RATIO, VAL_RATIO
    feats = wd.feature_names
    data = df[feats].to_numpy()
    total_windows = len(data) - LOOKBACK  # horizon=1
    rng = np.random.default_rng(42)
    idx = rng.permutation(total_windows)
    n_tr = int(total_windows * TRAIN_RATIO)
    test_window_ids = idx[n_tr:]  # aligns with wd.X_test order

    top = order[:20]
    context_cols = [c for c in ["timestamp", "pH", "TDS", "water_level",
                                "DHT_temp", "DHT_humidity", "water_temp",
                                "pH_reducer", "add_water", "nutrients_adder",
                                "humidifier", "ex_fan"] if c in df.columns]
    rows = []
    for rank, ti in enumerate(top, 1):
        src_window = test_window_ids[ti]
        target_row = src_window + LOOKBACK
        prev_row = target_row - 1
        rec = {"rank": rank,
               "actual_pH": round(float(true[ti]), 4),
               "pred_pH": round(float(pred[ti]), 4),
               "abs_err": round(float(err[ti]), 4)}
        if target_row < len(df):
            r = df.iloc[target_row]
            prev = df.iloc[prev_row] if prev_row >= 0 else None
            for c in context_cols:
                rec[c] = r[c]
            if prev is not None:
                rec["prev_pH"] = round(float(prev["pH"]), 4)
                rec["dpH_step"] = round(float(r["pH"] - prev["pH"]), 4)
        rows.append(rec)

    tbl = pd.DataFrame(rows)
    out_csv = METRICS / "phase3_top20_errors.csv"
    tbl.to_csv(out_csv, index=False)
    print(f"\nTop-20 largest errors (saved {out_csv}):")
    show = [c for c in ["rank", "actual_pH", "pred_pH", "abs_err", "prev_pH",
                        "dpH_step", "pH_reducer", "add_water", "nutrients_adder"]
            if c in tbl.columns]
    with pd.option_context("display.width", 200,
                           "display.max_columns", None):
        print(tbl[show].to_string(index=False))

    # How many of the top-20 coincide with a large true pH jump?
    if "dpH_step" in tbl.columns:
        big_jump = (tbl["dpH_step"].abs() > 0.10).sum()
        print(f"\n{big_jump}/20 of the largest errors occur at a true pH jump "
              f"> 0.10 in that step (i.e. dosing/refill transients).")


if __name__ == "__main__":
    main()
