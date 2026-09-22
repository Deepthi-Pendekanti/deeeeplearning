"""
Stage 5: Feature / correlation analysis (Section 3.3.7, reproduces Fig. 9).

Computes the Pearson correlation matrix (Eq. 10) across all sensor and actuator
features and saves a heatmap matching the paper's Fig. 9. Also prints the
pH-vs-feature correlations that the paper discusses in Section 4.2.1
(e.g. DHT_temp r=+0.37, TDS r=+0.31, water_level r=-0.25).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import DATA_PROCESSED, FIGURES, METRICS, TARGET

# Order features to match the paper's Fig. 9 layout as closely as possible.
FIG9_ORDER = [
    "pH", "TDS", "water_level", "DHT_temp", "DHT_humidity", "water_temp",
    "pH_reducer", "add_water", "nutrients_adder", "humidifier", "ex_fan",
]


def load_clean() -> pd.DataFrame:
    path = DATA_PROCESSED / "clean_1min.csv"
    df = pd.read_csv(path)
    return df


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FIG9_ORDER if c in df.columns]
    # Drop constant columns (zero variance) to avoid NaN correlations, but keep
    # them listed so the heatmap resembles the paper.
    corr = df[cols].corr(method="pearson")
    return corr


def plot_heatmap(corr: pd.DataFrame, out_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        vmin=-1, vmax=1, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8}, annot_kws={"size": 8},
    )
    plt.title("Correlation matrix of different features (reproduction of Fig. 9)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    df = load_clean()
    corr = correlation_matrix(df)

    out_fig = FIGURES / "fig9_correlation_matrix.png"
    plot_heatmap(corr, out_fig)
    print(f"[fig] saved {out_fig}")

    corr.round(4).to_csv(METRICS / "correlation_matrix.csv")
    print(f"[csv] saved {METRICS / 'correlation_matrix.csv'}")

    # Report pH correlations (Section 4.2.1).
    ph_corr = corr[TARGET].drop(labels=[TARGET]).sort_values(key=np.abs,
                                                             ascending=False)
    print("\nPearson correlation of each feature with pH (|r| desc):")
    for feat, r in ph_corr.items():
        print(f"  {feat:>16s}: {r:+.3f}")

    print("\nPaper (Sec 4.2.1) reports for reference: "
          "DHT_temp ~ +0.37, TDS ~ +0.31, water_level ~ -0.25, "
          "humidifier ~ +0.22, ex_fan ~ 0.")


if __name__ == "__main__":
    main()
