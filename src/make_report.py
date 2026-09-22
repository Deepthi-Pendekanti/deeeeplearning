"""
Stage 13: Consolidate all results into a single human-readable report
(results/RESULTS.md) for the faculty demo.
"""
from __future__ import annotations

import json

from config import METRICS, PAPER_TARGETS, RESULTS


def load(path):
    p = METRICS / path
    return json.load(open(p)) if p.exists() else None


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def main():
    lines = []
    a = lines.append

    a("# Reproduction Results — GRU Virtual pH Sensor\n")
    a("Paper: *Toward sustainable hydroponic farming: An AI-driven IoT "
      "framework for virtual pH sensing and sensor lifespan extension*, "
      "Moniruzzaman et al., Alexandria Engineering Journal 143 (2026) 40–63.\n")
    a("Dataset: Kaggle `itsmonir31/hydroponics-datasets` "
      "(DOI 10.34740/kaggle/ds/8223904, CC BY 4.0).\n")

    # Headline deep-model table.
    dl = load("training_results_paper_6feat_random.json")
    if dl:
        a("\n## 1. Deep learning models (headline config: paper subset, "
          "6 sensor features, random 80:20 split, 30 epochs, 3 seeds)\n")
        a("Best run per model (test set, physical pH units), vs paper:\n")
        a("| Model | RMSE (ours) | RMSE (paper) | MAE (ours) | MAE (paper) "
          "| R² (ours) | R² (paper) | params (ours) | params (paper) |")
        a("|---|---|---|---|---|---|---|---|---|")
        paper_params = {"gru": 24351, "lstm": 24100, "transformer": 251585}
        for name in ["gru", "lstm", "transformer"]:
            r = dl["results"].get(name)
            if not r:
                continue
            b = r["best_run"]
            pt = PAPER_TARGETS[name.upper() if name != "transformer"
                               else "Transformer"]
            pretty = {"gru": "GRU", "lstm": "LSTM",
                      "transformer": "Transformer"}[name]
            a(f"| {pretty} | {fmt(b['RMSE'])} | {pt['RMSE']} | "
              f"{fmt(b['MAE'])} | {pt['MAE']} | {fmt(b['R2'])} | {pt['R2']} | "
              f"{r['params']:,} | {paper_params[name]:,} |")

        a("\nMean ± std across 3 seeds:\n")
        a("| Model | RMSE | MAE | R² | within ±0.05 pH (%) |")
        a("|---|---|---|---|---|")
        for name in ["gru", "lstm", "transformer"]:
            r = dl["results"].get(name)
            if not r:
                continue
            pretty = {"gru": "GRU", "lstm": "LSTM",
                      "transformer": "Transformer"}[name]
            a(f"| {pretty} | {fmt(r['RMSE']['mean'])}±{fmt(r['RMSE']['std'],4)} "
              f"| {fmt(r['MAE']['mean'])}±{fmt(r['MAE']['std'],4)} "
              f"| {fmt(r['R2']['mean'])}±{fmt(r['R2']['std'],4)} "
              f"| {fmt(r['within_0.05pH']['mean'],2)} |")

    # Baselines.
    bl = load("baselines.json")
    if bl:
        a("\n## 2. Baselines\n")
        ar = bl["arima"]
        pa = bl["paper_arima"]
        a(f"**ARIMA** grid-searched optimum: order {tuple(ar['order'])} "
          f"(paper: {tuple(pa['order'])}).")
        a(f"- Ours: RMSE={fmt(ar['RMSE'])}, MAE={fmt(ar['MAE'])}, "
          f"R²={fmt(ar['R2'])}")
        a(f"- Paper: RMSE={pa['RMSE']}, MAE={pa['MAE']}, R²={pa['R2']}\n")
        a("**ML regressors** (windowed features, Table 6):\n")
        a("| Model | RMSE | MAE | R² | MAPE (%) | latency (ms) |")
        a("|---|---|---|---|---|---|")
        for name, m in bl["ml"].items():
            a(f"| {name} | {fmt(m['RMSE'])} | {fmt(m['MAE'])} | "
              f"{fmt(m['R2'])} | {fmt(m['MAPE'],3)} | "
              f"{fmt(m['latency_s']*1000,3)} |")

    # Correlation note.
    a("\n## 3. Feature correlation (Fig. 9)\n")
    a("See `results/figures/fig9_correlation_matrix.png`. Consistent with the "
      "paper: the exhaust fan shows ≈0 correlation with pH, and pH depends on "
      "multiple interdependent variables (motivating the multivariate GRU over "
      "univariate ARIMA).\n")

    a("\n## 4. Figures\n")
    for f, desc in [
        ("fig9_correlation_matrix.png", "Correlation matrix (Fig. 9)"),
        ("fig16_actual_vs_predicted.png", "Actual vs predicted pH (Fig. 16)"),
        ("fig18_residual_histograms.png", "Residual histograms (Fig. 18)"),
        ("fig19_threshold_outliers.png", "Threshold residual outliers (Fig. 19)"),
    ]:
        a(f"- `results/figures/{f}` — {desc}")

    out = RESULTS / "RESULTS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
