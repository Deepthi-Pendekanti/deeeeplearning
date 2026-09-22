"""
Stage 6: Normalization, windowing, and chronological split (Section 3.3.10).

Steps:
  1. Load the clean 1-minute table.
  2. Select the feature set (sensor-only by default; actuators added for ablation).
  3. Chronologically split into train / validation / test (80:20, with a
     validation subset drawn from the training portion).
  4. Fit a MinMax scaler on the TRAINING data only (no leakage), transform all.
  5. Build sliding windows: X = past LOOKBACK steps of all features,
     y = next-step pH (HORIZON=1). One-step-ahead prediction.

The scalers are returned so predictions can be inverse-transformed back to
physical pH units for metric computation and plotting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import (
    ACTUATOR_FEATURES,
    DATA_PROCESSED,
    HORIZON,
    LOOKBACK,
    SENSOR_FEATURES,
    TARGET,
    TRAIN_RATIO,
    VAL_RATIO,
)


@dataclass
class WindowedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler
    feature_names: list[str]
    target_index: int  # column index of pH within the feature matrix

    @property
    def n_features(self) -> int:
        return self.X_train.shape[2]

    def summary(self) -> str:
        return (
            f"features={self.feature_names}\n"
            f"  X_train={self.X_train.shape} y_train={self.y_train.shape}\n"
            f"  X_val  ={self.X_val.shape} y_val  ={self.y_val.shape}\n"
            f"  X_test ={self.X_test.shape} y_test ={self.y_test.shape}"
        )


def load_clean(source: str = "full") -> pd.DataFrame:
    """
    source='full'  -> our from-scratch cleaned month (clean_1min.csv)
    source='paper' -> authors' cleaned subset (clean_paper_subset.csv),
                      the apples-to-apples reproduction target.
    """
    fname = "clean_paper_subset.csv" if source == "paper" else "clean_1min.csv"
    df = pd.read_csv(DATA_PROCESSED / fname)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def select_features(include_actuators: bool = False,
                    exclude: list[str] | None = None) -> list[str]:
    """
    Feature ordering places pH first so target_index == 0.
    - Sensor-only (paper's final GRU): SENSOR_FEATURES.
    - Ablation configs add actuators (Section 4.4).
    """
    feats = list(SENSOR_FEATURES)  # pH, TDS, DHT_humidity, water_temp, water_level
    if include_actuators:
        actuators = [a for a in ACTUATOR_FEATURES
                     if not (exclude and a in exclude)]
        feats = feats + actuators
    return feats


def _make_windows(arr: np.ndarray, target_col: int,
                  lookback: int, horizon: int):
    """
    Build sliding windows.
      X[i] = arr[i : i+lookback, :]                 (all features)
      y[i] = arr[i+lookback+horizon-1, target_col]  (future pH)
    """
    X, y = [], []
    n = len(arr)
    last = n - lookback - horizon + 1
    for i in range(last):
        X.append(arr[i:i + lookback, :])
        y.append(arr[i + lookback + horizon - 1, target_col])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def build(include_actuators: bool = False,
          exclude: list[str] | None = None,
          verbose: bool = True,
          source: str = "full",
          split: str = "chrono",
          seed: int = 42) -> WindowedData:
    """
    split='chrono' : strict chronological 80/val/20 (train early, test late).
    split='random' : window-level random 80:10:10 assignment. Scaler is still
                     fit only on the chronological first 80% (no target leakage),
                     but windows are shuffled into splits so the test set spans
                     representative pH variation. This reproduces the paper's
                     jointly-consistent (low RMSE, high R^2) reported numbers.
    """
    df = load_clean(source=source)
    feats = select_features(include_actuators, exclude)
    target_index = feats.index(TARGET)  # == 0

    data = df[feats].to_numpy(dtype=np.float64)
    n = len(data)

    n_train_full = int(n * TRAIN_RATIO)
    n_val = int(n_train_full * VAL_RATIO)
    n_train = n_train_full - n_val

    # ---- MinMax scaling fit on the first 80% only (no leakage) ----
    feature_scaler = MinMaxScaler().fit(data[:n_train_full])
    target_scaler = MinMaxScaler().fit(
        data[:n_train_full, target_index:target_index + 1])
    data_s = feature_scaler.transform(data)

    if split == "random":
        # Window the whole scaled series, then randomly assign windows.
        # WARNING: this leaks temporally (adjacent windows share 9/10 rows).
        # Kept ONLY to reproduce the paper's reported numbers, never for
        # generalization claims. Use split='temporal' for a defensible estimate.
        X_all, y_all = _make_windows(data_s, target_index, LOOKBACK, HORIZON)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X_all))
        n_tr = int(len(idx) * TRAIN_RATIO)
        n_v = int(n_tr * VAL_RATIO)
        tr_idx = idx[:n_tr - n_v]
        v_idx = idx[n_tr - n_v:n_tr]
        te_idx = idx[n_tr:]
        X_train, y_train = X_all[tr_idx], y_all[tr_idx]
        X_val, y_val = X_all[v_idx], y_all[v_idx]
        X_test, y_test = X_all[te_idx], y_all[te_idx]
    elif split == "temporal":
        # Leakage-free temporal split: train (early) | gap | val | gap | test
        # (future). An embargo gap of LOOKBACK rows between consecutive splits
        # guarantees NO test/val window shares any row with a training window.
        gap = LOOKBACK
        train_s = data_s[:n_train]
        val_s = data_s[n_train + gap:n_train_full]
        test_s = data_s[n_train_full + gap:]
        X_train, y_train = _make_windows(train_s, target_index, LOOKBACK, HORIZON)
        X_val, y_val = _make_windows(val_s, target_index, LOOKBACK, HORIZON)
        X_test, y_test = _make_windows(test_s, target_index, LOOKBACK, HORIZON)
    else:  # chrono (strict time order, no embargo gap)
        train_s = data_s[:n_train]
        val_s = data_s[n_train:n_train_full]
        test_s = data_s[n_train_full:]
        X_train, y_train = _make_windows(train_s, target_index, LOOKBACK, HORIZON)
        X_val, y_val = _make_windows(val_s, target_index, LOOKBACK, HORIZON)
        X_test, y_test = _make_windows(test_s, target_index, LOOKBACK, HORIZON)

    wd = WindowedData(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_names=feats,
        target_index=target_index,
    )
    if verbose:
        print(f"[dataset] source={source} split={split} total rows={n}")
        print("[dataset] " + wd.summary())
    return wd


def inverse_target(wd: WindowedData, y_scaled: np.ndarray) -> np.ndarray:
    """Inverse-transform scaled pH values back to physical units."""
    y_scaled = np.asarray(y_scaled).reshape(-1, 1)
    return wd.target_scaler.inverse_transform(y_scaled).ravel()


if __name__ == "__main__":
    wd = build(include_actuators=False)
    print("\nSanity: inverse-transform of first 5 y_test (scaled -> pH):")
    print(inverse_target(wd, wd.y_test[:5]).round(3))
