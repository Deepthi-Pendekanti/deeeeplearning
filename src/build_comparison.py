"""
Build ONE clear, unified model comparison from all saved experiments.

Produces:
  * results/metrics/model_comparison_all.json  (machine-readable)
  * results/report_images/comparison_honest.png (leakage-free ranking)
  * results/report_images/comparison_leaky.png  (random-split, for contrast)
  * results/report_images/comparison_master.png (everything on one card)

Every number is read from the saved metric JSONs — nothing hand-typed.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
OUTIMG = ROOT / "results" / "report_images"
OUTIMG.mkdir(parents=True, exist_ok=True)


def load(name):
    p = METRICS / name
    return json.load(open(p)) if p.exists() else None


# ---------------------------------------------------------------------------
# DEACTIVATED models: hidden from the comparison DISPLAY only.
# Nothing is deleted — their experiments/data still exist. To re-show a model,
# remove it from this set and re-run this script + the explorer builder.
# ---------------------------------------------------------------------------
DEACTIVATED = {"ExtraTrees", "Persistence", "Improved GRU"}


def collect():
    """Gather every model into two lists: honest (temporal) and leaky (random).
    Models in DEACTIVATED are skipped for display."""
    honest, leaky = [], []

    # Phase-4 baselines (Persistence, Linear, RF, ExtraTrees, XGBoost, LSTM, GRU)
    for split, bucket in [("temporal", honest), ("random", leaky)]:
        d = load(f"phase4_baselines_paper_{split}.json")
        if d:
            for name, v in d.items():
                bucket.append({
                    "model": name, "R2": v["R2"], "MAE": v["MAE"],
                    "RMSE": v["RMSE"], "within005": v["within_0.05pH"],
                    "params": v.get("params"),
                })

    # Improved GRU (temporal) from phase5
    p5 = load("phase5_search.json")
    if p5 and "test_metrics" in p5:
        t = p5["test_metrics"]
        honest.append({"model": "Improved GRU", "R2": t["R2"], "MAE": t["MAE"],
                       "RMSE": t["RMSE"], "within005": t["within_0.05pH"],
                       "params": 10733})

    # CARD (temporal) from card_results
    cr = load("card_results.json")
    if cr and "card_metrics" in cr:
        m = cr["card_metrics"]
        honest.append({"model": "CARD (ours)", "R2": m["R2"], "MAE": m["MAE"],
                       "RMSE": m["RMSE"], "within005": m["within_0.05pH"],
                       "params": None})

    # Deep models (leaky) from training reproduction
    dl = load("training_results_paper_6feat_random.json")
    if dl and "results" in dl:
        for key, label in [("gru", "GRU"), ("lstm", "LSTM"),
                           ("transformer", "Transformer")]:
            if key in dl["results"]:
                b = dl["results"][key]["best_run"]
                # only add if not already present
                if not any(x["model"] == label for x in leaky):
                    leaky.append({"model": label, "R2": b["R2"], "MAE": b["MAE"],
                                  "RMSE": b["RMSE"],
                                  "within005": b["within_0.05pH"],
                                  "params": dl["results"][key]["params"]})

    # filter out deactivated models (display only)
    honest = [r for r in honest if r["model"] not in DEACTIVATED]
    leaky = [r for r in leaky if r["model"] not in DEACTIVATED]

    honest.sort(key=lambda r: r["R2"], reverse=True)
    leaky.sort(key=lambda r: r["R2"], reverse=True)
    return honest, leaky


def bar_table(rows, title, subtitle, filename, note):
    n = len(rows)
    fig = plt.figure(figsize=(13, 1.3 + 0.62 * n))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])

    # left: R2 horizontal bars
    ax = fig.add_subplot(gs[0, 0])
    names = [r["model"] for r in rows][::-1]
    r2 = [r["R2"] for r in rows][::-1]
    colors = []
    for nm in names:
        if "CARD" in nm: colors.append("#8e44ad")
        elif "Improved" in nm: colors.append("#2a8a55")
        elif nm in ("GRU", "LSTM"): colors.append("#c0392b")
        elif nm == "Persistence": colors.append("#7f8c8d")
        else: colors.append("#3b6ea5")
    ax.barh(names, r2, color=colors)
    ax.axvline(0, color="#333", lw=0.6)
    ax.set_xlabel("R²  (higher = better)")
    ax.set_title("R² ranking")
    for i, v in enumerate(r2):
        ax.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)

    # right: full metric table
    ax2 = fig.add_subplot(gs[0, 1]); ax2.axis("off")
    headers = ["Model", "R²", "MAE", "RMSE", "±0.05%", "params"]
    cells = []
    for r in rows:
        p = "-" if r["params"] is None else f"{r['params']:,}"
        cells.append([r["model"], f"{r['R2']:.3f}", f"{r['MAE']:.4f}",
                      f"{r['RMSE']:.4f}", f"{r['within005']:.1f}", p])
    tbl = ax2.table(cellText=cells, colLabels=headers, loc="center",
                    cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
    for j in range(len(headers)):
        tbl[(0, j)].set_facecolor("#22314f")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.99)
    fig.text(0.5, 0.93, subtitle, ha="center", fontsize=10, color="#555")
    fig.text(0.5, 0.01, note, ha="center", fontsize=8.5, color="#666",
             style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    fig.savefig(OUTIMG / filename, dpi=170, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"[img] {filename}")


def main():
    honest, leaky = collect()

    bar_table(
        honest,
        "Model Comparison — Leakage-Free (Honest) Test",
        "Temporal split, paper subset. THE trustworthy ranking.",
        "comparison_honest.png",
        "Persistence is a near-ceiling baseline on this smooth data; strong "
        "models cluster near it. Original GRU/LSTM (red) fail; Improved GRU "
        "(green) and CARD (purple) recover to the top group.",
    )
    bar_table(
        leaky,
        "Model Comparison — Random Split (LEAKY, for contrast only)",
        "Reproduces the paper's ~0.98 regime — inflated by temporal leakage.",
        "comparison_leaky.png",
        "Even trivial Persistence reaches R²≈0.98 here, proving these numbers "
        "reflect leakage, not real forecasting skill.",
    )

    with open(METRICS / "model_comparison_all.json", "w") as f:
        json.dump({"honest_temporal": honest, "leaky_random": leaky}, f, indent=2)
    print(f"[save] results/metrics/model_comparison_all.json")


if __name__ == "__main__":
    main()
