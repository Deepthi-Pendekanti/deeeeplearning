"""
Diagnostic: quantify why our RMSE differs from the paper.

Compares the pH distribution / volatility of:
  * our full cleaned month (clean_1min.csv), and its train/test split
  * the paper's own cleaned subset (IoTData_25K_with_interpolation...csv)

A naive 'persistence' baseline (predict pH_t = pH_{t-1}) lower-bounds the
achievable one-step RMSE and shows how 'easy' each series is.
"""
import glob

import numpy as np
import pandas as pd

from config import DATA_PROCESSED, DATA_RAW, TRAIN_RATIO


def persistence_rmse(ph: np.ndarray) -> float:
    return float(np.sqrt(np.mean((ph[1:] - ph[:-1]) ** 2)))


def describe(name: str, ph: np.ndarray):
    print(f"\n{name}")
    print(f"  n={len(ph):,}  pH range [{ph.min():.3f}, {ph.max():.3f}]  "
          f"std={ph.std():.4f}")
    print(f"  step-to-step |dpH| mean={np.mean(np.abs(np.diff(ph))):.4f}  "
          f"persistence RMSE={persistence_rmse(ph):.4f}")


def main():
    df = pd.read_csv(DATA_PROCESSED / "clean_1min.csv")
    ph = df["pH"].to_numpy()
    describe("Our full cleaned month (1-min)", ph)

    n_train_full = int(len(ph) * TRAIN_RATIO)
    describe("  -> our TEST split (last 20%)", ph[n_train_full:])

    # Paper's own cleaned subset.
    for f in glob.glob(str(DATA_RAW / "*with_interpolation*.csv")):
        d = pd.read_csv(f)
        describe(f"Paper cleaned subset: {f.split('/')[-1][:40]}", d["pH"].to_numpy())

    print("\nInterpretation: one-step RMSE is bounded below by the persistence "
          "RMSE. A series with small step-to-step change is far easier to "
          "predict, which is why the paper's smooth subset yields ~0.018 while "
          "the full volatile month yields a larger RMSE.")


if __name__ == "__main__":
    main()
