"""
Phase 9 — Publication-quality figures for the upgraded system.

Consumes saved artifacts from Phases 4/5/7/8 and produces clean plots:
  1. improved_actual_vs_pred.png      - actual vs predicted pH (temporal test)
  2. improved_error_over_time.png     - signed error over the test stream
  3. improved_residual_hist.png       - residual distribution
  4. improved_tolerance.png           - +-0.05 pH tolerance band scatter
  5. model_comparison_temporal.png    - R2/MAE bars across models (honest split)
  6. largest_errors.png               - the biggest errors highlighted
  7. failure_sim.png                  - traditional vs proposed under faults
  8. sensor_usage.png                 - physical vs virtual usage (Phase 7 & 8)
  9. uncertainty_vs_error.png         - MC-dropout uncertainty vs |error|
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES, METRICS, MODELS

plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3})


def _load_npz(name):
    p = MODELS / name
    return np.load(p, allow_pickle=True) if p.exists() else None


def _load_json(name):
    p = METRICS / name
    return json.load(open(p)) if p.exists() else None


def plot_improved_core():
    d = _load_npz("improved_gru_test_predictions.npz")
    if d is None:
        print("[skip] improved predictions missing"); return
    yt, yp = d["y_true"], d["y_pred"]
    err = yt - yp
    k = min(400, len(yt))

    # 1 actual vs predicted
    plt.figure(figsize=(11, 4))
    plt.plot(yt[:k], label="Actual pH", lw=1.3)
    plt.plot(yp[:k], label="Improved GRU (virtual)", lw=1.0, alpha=0.85)
    plt.title("Improved GRU — actual vs predicted pH (temporal test, first 400)")
    plt.xlabel("Time step"); plt.ylabel("pH"); plt.legend()
    plt.tight_layout(); plt.savefig(FIGURES / "improved_actual_vs_pred.png"); plt.close()

    # 2 error over time
    plt.figure(figsize=(11, 3.5))
    plt.plot(err, lw=0.6, color="darkorange")
    plt.axhline(0, color="k", lw=0.6)
    for tol in (0.05, -0.05):
        plt.axhline(tol, color="red", ls=":", lw=0.8)
    plt.title("Prediction error over time (Actual − Predicted)")
    plt.xlabel("Time step"); plt.ylabel("error (pH)")
    plt.tight_layout(); plt.savefig(FIGURES / "improved_error_over_time.png"); plt.close()

    # 3 residual histogram
    plt.figure(figsize=(6, 4))
    plt.hist(err, bins=60, color="steelblue")
    plt.axvline(0, color="k", ls="--", lw=0.8)
    plt.title("Improved GRU residual distribution")
    plt.xlabel("Residual (pH)"); plt.ylabel("Frequency")
    plt.tight_layout(); plt.savefig(FIGURES / "improved_residual_hist.png"); plt.close()

    # 4 tolerance scatter
    within = np.abs(err) <= 0.05
    idx = np.arange(len(err))
    plt.figure(figsize=(11, 3.8))
    plt.scatter(idx[within], err[within], s=5, c="steelblue", label="within ±0.05")
    plt.scatter(idx[~within], err[~within], s=16, c="red", marker="x",
                label="outside ±0.05")
    plt.axhline(0.05, color="red", ls=":"); plt.axhline(-0.05, color="red", ls=":")
    plt.title(f"±0.05 pH tolerance — {100*within.mean():.1f}% within band")
    plt.xlabel("Time step"); plt.ylabel("residual (pH)"); plt.legend()
    plt.tight_layout(); plt.savefig(FIGURES / "improved_tolerance.png"); plt.close()

    # 6 largest errors
    order = np.argsort(np.abs(err))[::-1][:20]
    plt.figure(figsize=(11, 4))
    plt.plot(yt, lw=0.8, label="Actual pH", alpha=0.7)
    plt.scatter(order, yt[order], c="red", s=40, zorder=5,
                label="20 largest errors")
    plt.title("Locations of the 20 largest prediction errors")
    plt.xlabel("Time step"); plt.ylabel("pH"); plt.legend()
    plt.tight_layout(); plt.savefig(FIGURES / "largest_errors.png"); plt.close()
    print("[ok] core improved-GRU figures")


def plot_model_comparison():
    d = _load_json("phase4_baselines_paper_temporal.json")
    if d is None:
        print("[skip] phase4 temporal missing"); return
    # add improved GRU
    imp = _load_json("phase5_search.json")
    names, r2s, maes = [], [], []
    for k, v in d.items():
        names.append(k); r2s.append(v["R2"]); maes.append(v["MAE"])
    if imp:
        names.append("ImprovedGRU")
        r2s.append(imp["test_metrics"]["R2"]); maes.append(imp["test_metrics"]["MAE"])

    order = np.argsort(r2s)
    names = [names[i] for i in order]; r2s = [r2s[i] for i in order]
    maes = [maes[i] for i in order]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = ["#c44" if n in ("GRU", "LSTM") else
              ("#2a2" if n == "ImprovedGRU" else "#69c") for n in names]
    ax[0].barh(names, r2s, color=colors); ax[0].set_title("R² (temporal, honest)")
    ax[0].axvline(0, color="k", lw=0.6)
    ax[1].barh(names, maes, color=colors); ax[1].set_title("MAE (pH) — lower better")
    fig.suptitle("Model comparison on the leakage-free temporal split")
    fig.tight_layout(); fig.savefig(FIGURES / "model_comparison_temporal.png")
    plt.close(fig)
    print("[ok] model comparison figure")


def plot_failure_and_usage():
    d = _load_npz("phase8_failure_stream.npz")
    fj = _load_json("phase8_failure.json")
    if d is not None:
        true, trad, prop = d["true"], d["traditional"], d["proposed"]
        is_fault = d["is_fault"]
        k = len(true)
        plt.figure(figsize=(12, 4.5))
        plt.plot(true, label="True pH", lw=1.4, color="black")
        plt.plot(trad, label="Traditional (corrupted physical)", lw=0.9,
                 color="red", alpha=0.7)
        plt.plot(prop, label="Proposed (virtual during faults)", lw=0.9,
                 color="green", alpha=0.8)
        # shade fault regions
        inf = False
        for t in range(k):
            if is_fault[t] and not inf:
                s = t; inf = True
            elif not is_fault[t] and inf:
                plt.axvspan(s, t, color="orange", alpha=0.12); inf = False
        if inf:
            plt.axvspan(s, k, color="orange", alpha=0.12)
        plt.title("Sensor failure simulation — traditional vs proposed "
                  "(shaded = fault windows)")
        plt.xlabel("Time step"); plt.ylabel("pH"); plt.legend()
        plt.tight_layout(); plt.savefig(FIGURES / "failure_sim.png"); plt.close()
        print("[ok] failure simulation figure")

    # sensor usage (Phase 7 + Phase 8)
    p7 = _load_json("phase7_confidence.json")
    if p7 and fj:
        labels = ["Always physical", "Hybrid (Phase 7)", "Fault-fallback (Phase 8)"]
        usage = [100.0, 100 * p7["physical_fraction"],
                 100 * fj["physical_reads_proposed"] / fj["n"]]
        plt.figure(figsize=(7, 4))
        bars = plt.bar(labels, usage, color=["#c44", "#2a8", "#28a"])
        for b, u in zip(bars, usage):
            plt.text(b.get_x() + b.get_width()/2, u + 1, f"{u:.1f}%",
                     ha="center")
        plt.ylabel("Physical pH reads (% of steps)")
        plt.title("Physical sensor usage: always-on vs proposed")
        plt.ylim(0, 110)
        plt.tight_layout(); plt.savefig(FIGURES / "sensor_usage.png"); plt.close()
        print("[ok] sensor usage figure")


def plot_uncertainty():
    d = _load_npz("phase7_hybrid_stream.npz")
    if d is None:
        print("[skip] phase7 stream missing"); return
    true, virt, unc = d["true"], d["virtual"], d["uncertainty"]
    err = np.abs(true - virt)
    plt.figure(figsize=(6, 5))
    plt.scatter(unc, err, s=6, alpha=0.4)
    plt.xlabel("MC-Dropout uncertainty (pH)")
    plt.ylabel("|actual − virtual| (pH)")
    plt.title("Does model uncertainty track real error?")
    if len(unc) > 2 and np.std(unc) > 0:
        c = np.corrcoef(unc, err)[0, 1]
        plt.annotate(f"Pearson r = {c:.2f}", (0.05, 0.92),
                     xycoords="axes fraction")
    plt.tight_layout(); plt.savefig(FIGURES / "uncertainty_vs_error.png"); plt.close()
    print("[ok] uncertainty vs error figure")


def main():
    plot_improved_core()
    plot_model_comparison()
    plot_failure_and_usage()
    plot_uncertainty()
    print(f"\n[done] figures in {FIGURES}")


if __name__ == "__main__":
    main()
