# Phase 4 — Strong baselines under one identical protocol

All models share the same windowed data, MinMax scaling (train-only fit),
look-back 10, one-step-ahead target. Trees/linear use the flattened window;
LSTM/GRU use the 3-D window. Paper subset. Both splits reported.

## Random split (LEAKY — reproduces paper-style numbers)

| Model | R² | MAE | RMSE | ±0.05 pH | params | lat (ms) |
|---|---|---|---|---|---|---|
| Persistence | 0.9815 | 0.0087 | 0.0292 | 97.50 | 0 | 0.000 |
| LinearRegression | 0.9820 | 0.0089 | 0.0288 | 98.27 | 61 | 0.81 |
| RandomForest | 0.9835 | 0.0077 | 0.0276 | 99.23 | – | 183.9 |
| ExtraTrees | 0.9837 | 0.0076 | 0.0274 | 99.17 | – | 221.1 |
| XGBoost | 0.9834 | 0.0081 | 0.0276 | 99.17 | – | 1.37 |
| LSTM | 0.9794 | 0.0163 | 0.0308 | 98.72 | 23,952 | 2.57 |
| GRU | 0.9799 | 0.0147 | 0.0304 | 98.14 | 24,051 | 4.20 |

Under leakage every model reaches R²≈0.98; even Linear Regression and a trivial
persistence predictor match the GRU. **High R² here proves nothing.**

## Temporal split (LEAKAGE-FREE — the honest estimate)

| Model | R² | MAE | RMSE | ±0.05 pH |
|---|---|---|---|---|
| **ExtraTrees** | **0.6755** | 0.0199 | **0.0253** | **95.08** |
| Persistence | 0.6500 | 0.0196 | 0.0263 | 93.78 |
| XGBoost | 0.5504 | 0.0236 | 0.0298 | 90.87 |
| RandomForest | 0.2617 | 0.0309 | 0.0381 | 81.48 |
| LinearRegression | 0.1938 | 0.0326 | 0.0399 | 80.05 |
| LSTM | 0.1720 | 0.0355 | 0.0404 | 77.01 |
| GRU | 0.0304 | 0.0390 | 0.0437 | 72.28 |

## Conclusions (honest)

1. **Is the GRU the best model? Not as configured.** On the honest split the
   GRU is the *worst* of all models tested and does not beat a trivial
   persistence baseline.
2. **ExtraTrees is the only model that clearly beats persistence** (R² 0.676 vs
   0.650; 95.1% vs 93.8% within ±0.05 pH). It is also far cheaper to train.
3. The recurrent models appear to overfit the leaky autocorrelation and fail to
   generalize to a future window. This is the real problem to address in
   Phase 5 (regularization, scheduler, Huber loss, early stopping, feature
   choices) — with the honest temporal split as the yardstick.

Artifacts: `results/metrics/phase4_baselines_paper_random.json`,
`results/metrics/phase4_baselines_paper_temporal.json`.
