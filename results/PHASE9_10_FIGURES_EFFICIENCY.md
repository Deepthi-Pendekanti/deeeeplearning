# Phase 9 — Figures & Phase 10 — Efficiency

## Phase 9 figures (results/figures/)

| File | Shows |
|---|---|
| improved_actual_vs_pred.png | Improved GRU vs true pH on the temporal test |
| improved_error_over_time.png | Signed error stream with ±0.05 band |
| improved_residual_hist.png | Residual distribution (≈ zero-mean) |
| improved_tolerance.png | ±0.05 pH tolerance scatter with outliers flagged |
| improved_loss_curve.png | Train vs validation loss (early-stopped, 17 epochs) |
| model_comparison_temporal.png | R²/MAE across all models on the honest split |
| largest_errors.png | Locations of the 20 largest errors |
| failure_sim.png | Traditional vs proposed under injected sensor faults |
| sensor_usage.png | Physical-read usage: always-on vs proposed |
| uncertainty_vs_error.png | MC-Dropout uncertainty vs actual error |

## Phase 10 — Efficiency & edge readiness

| Metric | GRU baseline | Improved GRU |
|---|---|---|
| Parameters | 24,051 | **10,733** (2.24× smaller) |
| Model size (.pt) | 97.7 KB | **47.0 KB** |
| CPU latency (mean) | 1.11 ms | **0.91 ms** |
| CPU latency (P95) | 1.86 ms | **1.19 ms** |
| Batch-64 throughput | ~21k samples/s | ~24k samples/s |
| ONNX export | 5.9 KB, parity 0.0 | **6.8 KB, parity 0.0** |

**Edge readiness:** both models export to ONNX with exact numerical parity
(max|diff| = 0.0), and the improved GRU is smaller, faster, and generalizes
better than the baseline — well within Raspberry-Pi-class budgets (sub-ms CPU
inference, <10 KB ONNX). ONNX with dynamic batch axis enables batch-independent
inference. Quantization was not needed given the already-tiny footprint;
INT8 quantization is a straightforward further step if required.

Artifacts: `results/metrics/phase10_efficiency.json`,
`results/models/gru_baseline.onnx`, `results/models/improved_gru.onnx`.
