"""
Live demo: load the trained GRU virtual pH sensor and predict on held-out
test windows, showing actual vs predicted pH side by side.

Great for a faculty demonstration -- it proves the saved model works and
reproduces the reported accuracy end to end.

Usage (from src/):
    python demo_predict.py
    python demo_predict.py --n 15
"""
import argparse

import numpy as np
import torch

import dataset as ds
from config import MODELS
from metrics import all_metrics
from models import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="rows to display")
    ap.add_argument("--model", default="gru")
    a = ap.parse_args()

    # Rebuild the exact headline dataset (paper subset, 6 feat, random split).
    wd = ds.build(include_actuators=False, source="paper", split="random",
                  verbose=False)

    model = build_model(a.model, wd.n_features)
    state = torch.load(MODELS / f"{a.model}_best.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        pred_s = model(torch.from_numpy(wd.X_test)).numpy()
    pred = ds.inverse_target(wd, pred_s)
    true = ds.inverse_target(wd, wd.y_test)

    m = all_metrics(true, pred)
    print(f"\nLoaded {a.model.upper()} virtual pH sensor "
          f"({wd.n_features} features, look-back 10).")
    print(f"Test-set metrics: RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
          f"R2={m['R2']:.4f}  within +/-0.05 pH={m['within_0.05pH']:.2f}%")

    print(f"\nSample predictions (first {a.n} test windows):")
    print(f"{'#':>3} {'actual pH':>10} {'predicted pH':>13} {'abs err':>9}")
    for i in range(min(a.n, len(true))):
        e = abs(true[i] - pred[i])
        flag = "" if e <= 0.05 else "  <-- >0.05"
        print(f"{i:>3} {true[i]:>10.3f} {pred[i]:>13.3f} {e:>9.4f}{flag}")


if __name__ == "__main__":
    main()
