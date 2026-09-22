# Reproduction Results — GRU Virtual pH Sensor

Paper: *Toward sustainable hydroponic farming: An AI-driven IoT framework for virtual pH sensing and sensor lifespan extension*, Moniruzzaman et al., Alexandria Engineering Journal 143 (2026) 40–63.

Dataset: Kaggle `itsmonir31/hydroponics-datasets` (DOI 10.34740/kaggle/ds/8223904, CC BY 4.0).


## 1. Deep learning models (headline config: paper subset, 6 sensor features, random 80:20 split, 30 epochs, 3 seeds)

Best run per model (test set, physical pH units), vs paper:

| Model | RMSE (ours) | RMSE (paper) | MAE (ours) | MAE (paper) | R² (ours) | R² (paper) | params (ours) | params (paper) |
|---|---|---|---|---|---|---|---|---|
| GRU | 0.0285 | 0.0181 | 0.0107 | 0.0116 | 0.9824 | 0.9824 | 24,051 | 24,351 |
| LSTM | 0.0279 | 0.019 | 0.0092 | 0.0124 | 0.9831 | 0.9787 | 23,952 | 24,100 |
| Transformer | 0.0369 | 0.0206 | 0.0219 | 0.0128 | 0.9703 | 0.9749 | 351,041 | 251,585 |

Mean ± std across 3 seeds:

| Model | RMSE | MAE | R² | within ±0.05 pH (%) |
|---|---|---|---|---|
| GRU | 0.0289±0.0003 | 0.0121±0.0013 | 0.9819±0.0003 | 98.72 |
| LSTM | 0.0281±0.0002 | 0.0097±0.0003 | 0.9829±0.0002 | 98.85 |
| Transformer | 0.0391±0.0020 | 0.0247±0.0020 | 0.9666±0.0033 | 89.03 |

## 2. Baselines

**ARIMA** grid-searched optimum: order (3, 1, 3) (paper: (3, 1, 3)).
- Ours: RMSE=0.0236, MAE=0.0179, R²=0.7197
- Paper: RMSE=0.2237, MAE=0.199, R²=-1.96

**ML regressors** (windowed features, Table 6):

| Model | RMSE | MAE | R² | MAPE (%) | latency (ms) |
|---|---|---|---|---|---|
| MLP | 0.0324 | 0.0145 | 0.9772 | 0.252 | 0.130 |
| KNN | 0.0327 | 0.0127 | 0.9768 | 0.220 | 1.603 |
| XGBoost | 0.0276 | 0.0081 | 0.9834 | 0.140 | 0.489 |

## 3. Feature correlation (Fig. 9)

See `results/figures/fig9_correlation_matrix.png`. Consistent with the paper: the exhaust fan shows ≈0 correlation with pH, and pH depends on multiple interdependent variables (motivating the multivariate GRU over univariate ARIMA).


## 4. Figures

- `results/figures/fig9_correlation_matrix.png` — Correlation matrix (Fig. 9)
- `results/figures/fig16_actual_vs_predicted.png` — Actual vs predicted pH (Fig. 16)
- `results/figures/fig18_residual_histograms.png` — Residual histograms (Fig. 18)
- `results/figures/fig19_threshold_outliers.png` — Threshold residual outliers (Fig. 19)