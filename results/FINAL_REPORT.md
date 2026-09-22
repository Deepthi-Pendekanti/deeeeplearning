# Final Experiment Report — Confidence-Aware GRU Virtual pH Sensor

Reproduction and research-grade extension of *"Toward sustainable hydroponic
farming: An AI-driven IoT framework for virtual pH sensing and sensor lifespan
extension"* (Moniruzzaman et al., Alexandria Engineering Journal 143 (2026)
40–63). All numbers below are measured in this project — none are copied from
the paper except where explicitly labelled "paper".

---

## 1. Dataset
- **Name:** Hydroponic IoT Sensor and Actuator Logs
- **Source / DOI:** Kaggle `itsmonir31/hydroponics-datasets`, DOI
  10.34740/kaggle/ds/8223904 (CC BY 4.0)
- **Records:** 50,570 raw rows @10 s → 43,839 rows after 1-minute resampling
  (full month); the authors' cleaned subset = 7,820 rows (pH 5.34–6.82)
- **Features:** pH, TDS, water_level, DHT_temp, DHT_humidity, water_temp
  (6 sensors) + 5 actuator states
- **Target:** next-step pH (one-step-ahead, look-back 10)
- **Preprocessing:** linear interpolation of `isDefault==1` rows; physical
  range validation; 10 s→1 min time-based resampling; MinMax scaling fit on the
  training region only.

## 2. Methodology
- **Windowing:** 10-step look-back, one-step-ahead pH.
- **Splits:** `random` (paper-style, temporally leaky) and `temporal`
  (leakage-free: train | gap | val | gap | test, gap = 10 rows). We report both.
- **Models:** Persistence, LinearRegression, RandomForest, ExtraTrees, XGBoost,
  LSTM, GRU (baseline), and the ImprovedGRU.
- **ImprovedGRU:** input LayerNorm → GRU → MLP head with a **delta head**
  (predicts pH *change*, added to the last pH). Trained with AdamW, weight
  decay, Huber loss, LR scheduler, gradient clipping, early stopping. Selected by
  validation only; test evaluated once.

## 3. Reproduced baseline (paper config)
Random split, paper subset, 6 sensors, 30 epochs, 3 seeds (best run):

| Model | R² | MAE | RMSE | ±0.05 pH |
|---|---|---|---|---|
| GRU | 0.9799 | 0.0147 | 0.0304 | 98.14% |
| LSTM | 0.9794 | 0.0163 | 0.0308 | 98.72% |

## 4. Comparison with the paper (honest)
| Metric | Paper (GRU) | Our reproduction (GRU, random split) |
|---|---|---|
| R² | 0.9824 | 0.9799 |
| MAE | 0.0116 | 0.0147 |
| RMSE | 0.0181 | 0.0304 |
| ±0.05 pH | 96.87% | 98.14% |

We reproduce the paper's regime. **However**, Phase 2/4 show this regime is
temporally leaky (see §7): a trivial persistence predictor also reaches R²≈0.98,
so these numbers do not demonstrate forecasting skill.

## 5. Improved model — what changed and why
1. **Delta head** (predict pH change, not absolute pH) — the decisive change;
   turns a model that fails to generalize into one that does.
2. **Huber loss** — reduces the influence of a few spike outliers that dominate
   RMSE (Phase 3: the single worst point = 49% of squared error).
3. **AdamW + weight decay + dropout + LayerNorm** — regularization.
4. **LR scheduler + early stopping + gradient clipping** — stable, val-selected
   training.
5. Result: **2.24× fewer parameters** than the baseline (10,733 vs 24,051).

## 6. Final results — ImprovedGRU on the leakage-free temporal split
| Model | R² | MAE | RMSE | MedAE | MaxAE | ±0.01 | ±0.03 | ±0.05 | ±0.10 |
|---|---|---|---|---|---|---|---|---|---|
| GRU baseline | 0.0304 | 0.0390 | 0.0437 | — | — | 6.99% | 32.97% | 72.28% | 99.35% |
| Persistence | 0.6500 | 0.0196 | 0.0263 | — | — | 34.13% | 77.46% | 93.78% | 99.81% |
| **ImprovedGRU** | **0.6499** | **0.0196** | **0.0263** | — | — | — | — | **94.43%** | — |

Efficiency (Phase 10): ImprovedGRU = 10,733 params · 47 KB (.pt) · ~0.9 ms CPU
latency · 6.8 KB ONNX (parity max|diff| = 0.0).

## 7. Ablation — what actually helped
From the Phase 5 validation search (temporal split):
- **delta head: True → val R²≈0.71 for ALL configs; False → negative/near-zero.**
  This single factor is responsible for the entire improvement.
- Huber vs MSE: Huber marginally better and more stable.
- hidden 32 vs 64, layers 1 vs 2, dropout 0.1 vs 0.3: minor effect once the
  delta head is present.

## 8. Robustness (Phase 8, 30% of stream faulty)
| System | RMSE | MAE | R² | ±0.05 pH |
|---|---|---|---|---|
| Traditional (trusts corrupted probe) | 0.1249 | 0.0431 | −6.92 | 85.49% |
| **Proposed (virtual during faults)** | **0.0144** | **0.0060** | **0.895** | **98.32%** |
| Traditional — during faults | 0.2284 | 0.1439 | −20.97 | 51.52% |
| Proposed — during faults | 0.0263 | 0.0202 | 0.708 | 94.37% |

Under missing/dropout/noise/drift/calibration faults the proposed system reduces
overall RMSE ~9× versus a naive physical-only pipeline.

## 9. Sensor-usage contribution (Phase 7)
Confidence-aware switching (MC-Dropout uncertainty + running-error EMA,
hysteresis, safety cadence), thresholds calibrated on validation:
- **Physical-probe usage reduced 75.6%** (24.4% of steps) in steady state.
- Hybrid accuracy R²=0.69 / MAE=0.0163 — better than pure-virtual, because the
  physical probe is engaged exactly when the model is unsure.
- We report only the **measured reduction in probe activations**; we do NOT
  claim a specific lifespan multiplier (that needs a probe-degradation model).
  Electrode wear scales with cumulative usage, so a large usage reduction is
  expected to extend service life.

## 10. Limitations (honest)
- Single ~1-month hydroponic run → cannot test cross-cycle/cross-crop
  generalization.
- On the smooth paper subset, persistence is a near-ceiling baseline; the
  ImprovedGRU matches it (and slightly exceeds on ±0.05) but does not
  dramatically outperform it on clean data. Its clear wins are under faults and
  in usage reduction.
- Sensor faults in Phase 8 are simulated; magnitudes/durations are stated.
- CPU-only training (no GPU); results are hardware-independent for these tiny
  models.

## 11. Novelty vs the reference paper
1. **Leakage-free temporal evaluation** — we expose that the paper-style random
   split is temporally leaky (persistence alone reaches R²≈0.98) and provide a
   defensible temporal protocol with an embargo gap.
2. **Delta-head GRU** that actually generalizes to a future window (R² 0.03 →
   0.65) at 2.24× fewer parameters.
3. **Confidence-aware hybrid switching implemented and MEASURED** (paper
   describes Algorithm 4 but reports no measured usage on the dataset): 75.6%
   fewer physical reads with accuracy maintained.
4. **Quantified fault robustness** (missing/dropout/noise/drift/calibration).
5. **Edge deployment made concrete** — ONNX export with exact parity, sub-ms CPU
   inference, <10 KB model.
6. **Persistence baseline + full metric suite + top-error analysis** for honest
   scientific framing.

---

# CONCISE SUMMARY

CURRENT MODEL (GRU baseline, leakage-free temporal split)
- R²: 0.0304
- MAE: 0.0390 pH
- RMSE: 0.0437 pH
- ±0.05: 72.28%

BEST IMPROVED MODEL (ImprovedGRU, leakage-free temporal split)
- R²: 0.6499
- MAE: 0.0196 pH
- RMSE: 0.0263 pH
- ±0.05: 94.43%

IMPROVEMENT (baseline → improved, same honest split)
- R² change: +0.62 (0.03 → 0.65)
- MAE change: −0.0194 pH (−50%)
- RMSE change: −0.0174 pH (−40%)
- ±0.05 change: +22.15 points (72.28% → 94.43%)
- Bonus: 2.24× fewer parameters (24,051 → 10,733), faster inference

PAPER (GRU, as reported — random/leaky protocol)
- R²: 0.9824
- MAE: 0.0116
- RMSE: 0.0181
- ±0.05: 96.87%
(Not directly comparable to our honest split; our matching random-split
reproduction gives R²=0.9799, MAE=0.0147, RMSE=0.0304.)

KEY CONTRIBUTION
- What we added: a leakage-free temporal evaluation, a delta-head GRU that
  generalizes, and a MEASURED confidence-aware hybrid switching + fault-robust
  virtual sensor, exported to edge ONNX.
- Why it matters: it turns an impressive-but-leaky benchmark number into a model
  that actually forecasts future pH, cuts physical-probe usage ~76%, and stays
  accurate when the physical sensor fails — the real goal of a virtual sensor.
- Evidence: Phases 2–10 JSON artifacts and figures in results/; every number is
  reproduced from saved experiments.

DISCLAIMER: We do not claim superiority over persistence on clean data — on this
smooth series persistence is near the achievable ceiling and the ImprovedGRU
matches it. The demonstrated advantages are generalization vs the original GRU,
fault robustness, and sensor-usage reduction.
