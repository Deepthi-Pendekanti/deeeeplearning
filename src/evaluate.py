"""
Stages 10-11: Evaluation and figure reproduction.

Consumes the saved best-run test predictions ({model}_test_predictions.npz)
and produces:
  * A metrics table (RMSE/MAE/R2/MAPE/within-tolerance) vs the paper (Table 5/6).
  * Fig. 16: actual vs predicted pH time series (per model).
  * Fig. 18: residual error histograms (GRU/LSTM/Transformer).
  * Fig. 19: threshold-based residual outliers with a +/-0.05 pH band.

Run AFTER train.py.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES, METRICS, MODELS, PAPER_TARGETS, RESIDUAL_TOLERANCE
from metrics import all_metrics

MODEL_ORDER = ["gru", "lstm", "transformer"]
PRETTY = {"gru": "GRU", "lstm": "LSTM", "transformer": "Transformer"}


def load_predictions(name: str):
    path = MODELS / f"{name}_test_predictions.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return d["y_true"], d["y_pred"]


def metrics_table(preds: dict) -> dict:
    table = {}
    for name, (yt, yp) in preds.items():
        table[name] = all_metrics(yt, yp)
    return table


def print_comparison(table: dict):
    print("\n" + "=" * 78)
    print("RESULTS vs PAPER (test set, physical pH units)")
    print("=" * 78)
    print(f"{'Model':<12}{'RMSE':>9}{'MAE':>9}{'R2':>9}{'MAPE%':>9}"
          f"{'<=0.05pH%':>11}  | paper R2/RMSE/MAE")
    for name in MODEL_ORDER:
        if name not in table:
            continue
        m = table[name]
        pt = PAPER_TARGETS.get(PRETTY[name], {})
        print(f"{PRETTY[name]:<12}{m['RMSE']:>9.4f}{m['MAE']:>9.4f}"
              f"{m['R2']:>9.4f}{m['MAPE']:>9.3f}{m['within_0.05pH']:>11.2f}"
              f"  | {pt.get('R2','?')}/{pt.get('RMSE','?')}/{pt.get('MAE','?')}")


def plot_actual_vs_pred(preds: dict, n_points: int = 130):
    """Fig. 16: actual vs predicted for the first n_points test samples."""
    present = [m for m in MODEL_ORDER if m in preds]
    fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 4),
                             squeeze=False)
    for ax, name in zip(axes[0], present):
        yt, yp = preds[name]
        k = min(n_points, len(yt))
        ax.plot(yt[:k], label="Actual pH", linewidth=1.2)
        ax.plot(yp[:k], label="Predicted pH", linewidth=1.0, alpha=0.8)
        ax.set_title(f"{PRETTY[name]}: Actual vs Predicted")
        ax.set_xlabel("Time step")
        ax.set_ylabel("pH")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Fig. 16 reproduction — actual vs predicted pH (first "
                 f"{n_points} test samples)")
    fig.tight_layout()
    out = FIGURES / "fig16_actual_vs_predicted.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig] {out}")


def plot_residual_histograms(preds: dict):
    """Fig. 18: residual (actual - predicted) histograms."""
    present = [m for m in MODEL_ORDER if m in preds]
    fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 4),
                             squeeze=False)
    for ax, name in zip(axes[0], present):
        yt, yp = preds[name]
        res = yt - yp
        ax.hist(res, bins=60, color="steelblue", edgecolor="none")
        ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
        ax.set_title(f"Residual Histogram ({PRETTY[name]})")
        ax.set_xlabel("Residual (Actual pH - Predicted pH)")
        ax.set_ylabel("Frequency")
        ax.set_xlim(-0.15, 0.15)
    fig.suptitle("Fig. 18 reproduction — residual error histograms")
    fig.tight_layout()
    out = FIGURES / "fig18_residual_histograms.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig] {out}")


def plot_threshold_outliers(preds: dict, tol: float = RESIDUAL_TOLERANCE):
    """Fig. 19: pointwise residuals with a +/-tol band; flag outliers."""
    present = [m for m in MODEL_ORDER if m in preds]
    fig, axes = plt.subplots(len(present), 1, figsize=(10, 3.2 * len(present)),
                             squeeze=False)
    summary = {}
    for ax, name in zip(axes[:, 0], present):
        yt, yp = preds[name]
        res = yt - yp
        within = np.abs(res) <= tol
        acc = 100.0 * within.mean()
        summary[name] = {"n": int(len(res)),
                         "within": int(within.sum()),
                         "outliers": int((~within).sum()),
                         "accuracy_pct": acc}
        idx = np.arange(len(res))
        ax.scatter(idx[within], res[within], s=6, c="steelblue",
                   label="Residuals")
        ax.scatter(idx[~within], res[~within], s=20, c="red", marker="x",
                   label=f"Outliers > +/-{tol}")
        ax.axhline(tol, color="red", linestyle=":", linewidth=0.8)
        ax.axhline(-tol, color="red", linestyle=":", linewidth=0.8)
        ax.axhline(0, color="k", linestyle="--", linewidth=0.6)
        ax.set_title(f"Threshold-Based Outliers ({PRETTY[name]}) — "
                     f"within {tol} pH: {acc:.2f}%")
        ax.set_xlabel("Observation Index")
        ax.set_ylabel("Residual")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Fig. 19 reproduction — threshold-based residual outliers")
    fig.tight_layout()
    out = FIGURES / "fig19_threshold_outliers.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig] {out}")
    return summary


def main():
    preds = {}
    for name in MODEL_ORDER:
        p = load_predictions(name)
        if p is not None:
            preds[name] = p
    if not preds:
        print("No predictions found. Run train.py first.")
        return

    table = metrics_table(preds)
    print_comparison(table)

    plot_actual_vs_pred(preds)
    plot_residual_histograms(preds)
    tol_summary = plot_threshold_outliers(preds)

    # Save the evaluation summary.
    out = {
        "metrics": table,
        "paper_targets": PAPER_TARGETS,
        "tolerance_summary": tol_summary,
        "tolerance_pH": RESIDUAL_TOLERANCE,
    }
    with open(METRICS / "evaluation_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[save] {METRICS / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()
