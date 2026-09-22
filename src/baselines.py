"""
Stage 12: Classical and ML baselines (Section 4.2.2, Table 6).

1. ARIMA (Section 4.2.2, "Classical Statistical Model"):
   univariate pH, chronological 80:20, grid-search (p,d,q) by lowest AIC.
   Paper's optimum ARIMA(3,1,3): RMSE 0.2237, MAE 0.1990, R2 -1.96.

2. ML regressors on the flattened look-back window (Table 6):
   MLP, KNN, XGBoost. Same features / split / scaling as the deep models.

Run AFTER preprocess (paper subset) so the smooth series is used, matching the
paper's evaluation setting.
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

import dataset as ds
from config import METRICS
from metrics import all_metrics

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------
# ARIMA
# ----------------------------------------------------------------------------
def run_arima(source: str = "paper", max_order: int = 3):
    from statsmodels.tsa.arima.model import ARIMA

    df = ds.load_clean(source=source)
    ph = df["pH"].to_numpy(dtype=float)
    n = len(ph)
    n_train = int(n * 0.8)
    train, test = ph[:n_train], ph[n_train:]

    print(f"[arima] grid search (p,d,q) up to {max_order} by AIC ...")
    best = None
    for p in range(max_order + 1):
        for d in range(2):
            for q in range(max_order + 1):
                try:
                    res = ARIMA(train, order=(p, d, q)).fit()
                    if best is None or res.aic < best[1]:
                        best = ((p, d, q), res.aic)
                except Exception:
                    continue
    order, aic = best
    print(f"[arima] best order={order} AIC={aic:.2f}")

    # One-step-ahead rolling forecast over the test set.
    res = ARIMA(ph, order=order).fit()
    fc = res.predict(start=n_train, end=n - 1)
    m = all_metrics(test, np.asarray(fc))
    m["order"] = list(order)
    m["aic"] = float(aic)
    print(f"[arima] test RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.4f}")
    return m


# ----------------------------------------------------------------------------
# ML regressors on the flattened window
# ----------------------------------------------------------------------------
def _flatten(wd: ds.WindowedData):
    def flat(X):
        return X.reshape(X.shape[0], -1)
    return flat(wd.X_train), flat(wd.X_val), flat(wd.X_test)


def run_ml_baselines(source: str = "paper", split: str = "random"):
    wd = ds.build(include_actuators=False, source=source, split=split,
                  verbose=False)
    Xtr, Xval, Xte = _flatten(wd)
    ytr, yte = wd.y_train, wd.y_test

    models = {
        "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                            random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=6,
                                learning_rate=0.1, random_state=42,
                                verbosity=0),
    }

    out = {}
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        fit_t = time.perf_counter() - t0

        # inference latency (single sample)
        x1 = Xte[:1]
        for _ in range(10):
            model.predict(x1)
        t0 = time.perf_counter()
        for _ in range(100):
            model.predict(x1)
        lat = (time.perf_counter() - t0) / 100

        pred_s = model.predict(Xte)
        pred = ds.inverse_target(wd, pred_s)
        true = ds.inverse_target(wd, yte)
        m = all_metrics(true, pred)
        m["fit_time_s"] = fit_t
        m["latency_s"] = lat
        out[name] = m
        print(f"[{name:>8}] RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} "
              f"R2={m['R2']:.4f} MAPE={m['MAPE']:.3f}% latency={lat*1000:.3f}ms")
    return out


def main():
    print("=" * 60)
    print("Baselines (paper subset)")
    print("=" * 60)

    print("\n--- ARIMA (univariate pH, chronological) ---")
    arima = run_arima(source="paper")

    print("\n--- ML regressors (windowed features, random split) ---")
    ml = run_ml_baselines(source="paper", split="random")

    out = {"arima": arima, "ml": ml,
           "paper_arima": {"order": [3, 1, 3], "AIC": -63762.23,
                           "RMSE": 0.2237, "MAE": 0.1990, "R2": -1.96}}
    with open(METRICS / "baselines.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[save] {METRICS / 'baselines.json'}")


if __name__ == "__main__":
    main()
