"""
Task 9 (figures) — CARD publication plots from saved artifacts.

  card_gate_vs_dosing.png   - gate value vs dosing activity (Component 1 works)
  card_casc_budget.png      - achieved usage vs target budget + accuracy (CASC)
  card_uncertainty.png      - predicted sigma vs |error| (Component 3 calibration)
  card_pred.png             - CARD vs true pH on the test stream
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import dataset as ds
from card import CARD, CARDConfig, predict
from config import FIGURES, METRICS, MODELS, SENSOR_FEATURES

plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3})


def load_card():
    ck = torch.load(MODELS / "card_best.pt", map_location="cpu")
    m = CARD(CARDConfig(**ck["cfg"])); m.load_state_dict(ck["state"]); m.eval()
    return m, ck["cfg"]


def main():
    stream = np.load(MODELS / "card_test_stream.npz")
    res = json.load(open(METRICS / "card_results.json"))
    true, card = stream["true"], stream["card"]
    sigma, gate = stream["sigma"], stream["gate"]

    # 1) CARD vs true pH (first 400)
    k = min(400, len(true))
    plt.figure(figsize=(11, 4))
    plt.plot(true[:k], label="True pH", lw=1.3)
    plt.plot(card[:k], label="CARD (virtual)", lw=1.0, alpha=0.85)
    plt.title("CARD — actual vs predicted pH (temporal test)")
    plt.xlabel("Time step"); plt.ylabel("pH"); plt.legend()
    plt.tight_layout(); plt.savefig(FIGURES / "card_pred.png"); plt.close()

    # 2) CASC budget vs achieved + accuracy
    casc = res["casc"]
    b = [c["budget"] * 100 for c in casc]
    a = [c["achieved_usage"] * 100 for c in casc]
    r2 = [c["hybrid_R2"] for c in casc]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(b, a, "o-", color="#4aa3ff", label="achieved usage")
    ax1.plot([0, 25], [0, 25], "--", color="#888", label="ideal (achieved=target)")
    ax1.set_xlabel("Target probe-usage budget (%)")
    ax1.set_ylabel("Achieved physical usage (%)", color="#4aa3ff")
    ax2 = ax1.twinx()
    ax2.plot(b, r2, "s-", color="#2a8a55", label="hybrid R²")
    ax2.set_ylabel("Hybrid R²", color="#2a8a55")
    ax1.set_title("CASC controller: hits the usage budget; accuracy rises with budget")
    fig.tight_layout(); fig.savefig(FIGURES / "card_casc_budget.png"); plt.close()

    # 3) uncertainty vs error (calibration)
    err = np.abs(true - card)
    plt.figure(figsize=(6, 5))
    plt.scatter(sigma, err, s=6, alpha=0.4)
    plt.xlabel("Predicted uncertainty σ (pH)")
    plt.ylabel("|actual − CARD| (pH)")
    plt.title("Component 3: does predicted σ track real error?")
    if sigma.std() > 0:
        c = np.corrcoef(sigma, err)[0, 1]
        plt.annotate(f"Pearson r = {c:.2f}", (0.05, 0.92),
                     xycoords="axes fraction")
    plt.tight_layout(); plt.savefig(FIGURES / "card_uncertainty.png"); plt.close()

    # 4) gate vs dosing on the FULL month (has dosing events)
    wd = ds.build(include_actuators=True, source="full", split="temporal",
                  verbose=False)
    ns = len(SENSOR_FEATURES)
    model, _ = load_card()
    out = predict(model, wd.X_test)
    g = out["gate"]
    act = wd.X_test[:, -1, ns:].sum(axis=1)
    plt.figure(figsize=(7, 4.5))
    active = act > 0
    plt.scatter(np.where(~active)[0], g[~active], s=5, c="#888",
                label="system quiet", alpha=0.4)
    plt.scatter(np.where(active)[0], g[active], s=14, c="#ff5d6c",
                label="dosing active")
    plt.axhline(g[~active].mean(), color="#888", ls=":",
                label=f"mean quiet={g[~active].mean():.2f}")
    plt.axhline(g[active].mean(), color="#ff5d6c", ls=":",
                label=f"mean dosing={g[active].mean():.2f}")
    plt.title("Component 1: dosing gate opens during dosing, closes when quiet")
    plt.xlabel("Time step (full-month test)"); plt.ylabel("gate value")
    plt.legend(fontsize=8); plt.ylim(-0.05, 1.05)
    plt.tight_layout(); plt.savefig(FIGURES / "card_gate_vs_dosing.png"); plt.close()

    print("[ok] card figures: card_pred, card_casc_budget, card_uncertainty, "
          "card_gate_vs_dosing")


if __name__ == "__main__":
    main()
