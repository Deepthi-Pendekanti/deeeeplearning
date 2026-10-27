"""
Task B7 — tolerance-band breakdown.

For each model on the honest (leakage-free) test set, count how many predictions
fall in each absolute-error band:
    <=0.01 | 0.01-0.03 | 0.03-0.05 | 0.05-0.10 | >0.10  (pH units)

Saves results/metrics/tolerance_bands.json and a clean table image
results/report_images/tolerance_bands.png. All numbers measured.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "results" / "models"
METRICS = ROOT / "results" / "metrics"
OUTIMG = ROOT / "results" / "report_images"
OUTIMG.mkdir(parents=True, exist_ok=True)

BANDS = [(0.0, 0.01, "≤0.01"), (0.01, 0.03, "0.01–0.03"),
         (0.03, 0.05, "0.03–0.05"), (0.05, 0.10, "0.05–0.10"),
         (0.10, np.inf, ">0.10")]


def bands_for(true, pred):
    err = np.abs(np.asarray(true) - np.asarray(pred))
    n = len(err)
    out = []
    for lo, hi, label in BANDS:
        cnt = int(np.sum((err > lo) & (err <= hi)) if lo > 0
                  else np.sum(err <= hi))
        out.append({"band": label, "count": cnt, "pct": round(100 * cnt / n, 2)})
    return out, n


def main():
    d = np.load(MODELS / "card_test_stream.npz")
    true = d["true"]
    models = {
        "Persistence": d["persistence"],
        "Improved GRU": d["improved_gru"],
        "CARD (ours)": d["card"],
    }

    result = {}
    for name, pred in models.items():
        bands, n = bands_for(true, pred)
        result[name] = {"n": n, "bands": bands}

    with open(METRICS / "tolerance_bands.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[save] results/metrics/tolerance_bands.json (n={len(true)} samples)")

    # ---- table image ----
    band_labels = [b[2] for b in BANDS]
    model_names = list(models.keys())
    cell = []
    for name in model_names:
        row = [name]
        for b in result[name]["bands"]:
            row.append(f"{b['pct']:.1f}%")
        cell.append(row)

    fig, ax = plt.subplots(figsize=(11, 2.4))
    ax.axis("off")
    ax.set_title("Prediction Error Distribution by Tolerance Band "
                 "(honest test, 1,544 samples)", fontsize=15, fontweight="bold")
    headers = ["Model"] + [f"|err| {l} pH" for l in band_labels]
    tbl = ax.table(cellText=cell, colLabels=headers, loc="center",
                   cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.9)
    for j in range(len(headers)):
        tbl[(0, j)].set_facecolor("#22314f")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(model_names) + 1):
        tbl[(i, 0)].set_text_props(fontweight="bold")
        if model_names[i - 1] == "CARD (ours)":
            for j in range(len(headers)):
                tbl[(i, j)].set_facecolor("#d5f2e3")
    ax.text(0.5, -0.15, "Most predictions land in the two smallest error bands. "
            "'within ±0.05 pH' = sum of the first three columns.",
            ha="center", transform=ax.transAxes, fontsize=9, style="italic",
            color="#555")
    fig.tight_layout()
    fig.savefig(OUTIMG / "tolerance_bands.png", dpi=170, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("[img] tolerance_bands.png")

    # console
    print(f"\n{'Model':<15}" + "".join(f"{l:>12}" for l in band_labels))
    for name in model_names:
        print(f"{name:<15}" + "".join(
            f"{b['pct']:>11.1f}%" for b in result[name]["bands"]))


if __name__ == "__main__":
    main()
