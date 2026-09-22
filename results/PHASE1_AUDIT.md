# Phase 1 — Project Audit

Audit of the existing GRU virtual-pH-sensor pipeline before any改. Every point
below was verified by reading the actual source in `src/`.

## What is CORRECT (keep as-is)

| Area | File | Assessment |
|---|---|---|
| Column normalization | `preprocess.py` | Robust mapping of raw headers → canonical names. Correct. |
| Sensor-failure handling | `preprocess.py::interpolate_defaults` | `isDefault==1` rows → NaN → linear interpolation (paper Eq. 8). Correct. |
| Physical range check | `preprocess.py::validate_ranges` | Impossible values (TDS<0, humidity>100) repaired. Sound, defensible. |
| Resampling | `preprocess.py::resample_time` | 10 s → 1 min time-based interpolation (Eq. 9). Correct. |
| Target definition | `dataset.py` | One-step-ahead pH (`t+1`), look-back 10. Matches paper. |
| Scaler leakage guard | `dataset.py::build` | MinMax fit on **first 80% only**, then applied to all. Correct — no scaler leakage. |
| Metrics | `metrics.py` | RMSE/MAE/R²/MAPE correct (Eq. 23–25). |
| Param-count fidelity | `models.py` | GRU with 8 features = 24,351 params, exact match to paper Table A.9. |
| Seeding | `train.py::set_seed` | random/np/torch seeded; 3 seeds. Reasonable reproducibility. |
| Inference latency | `train.py::measure_latency` | Warm-up + timed loop. Correct methodology. |

## What NEEDS improvement (issues found)

### A. TEMPORAL DATA LEAKAGE in the `random` split  ← most important
`dataset.py::build(split="random")` builds all sliding windows from the full
series, then **shuffles windows** into train/val/test. Because window `i` and
window `i+1` share 9 of their 10 input rows and have adjacent targets, a window
in the test set can be almost identical to one the model trained on. This is
classic time-series leakage. It is the reason both the paper and our
reproduction reach R² ≈ 0.98 with a random split.
- **Impact:** inflates reported accuracy; does not measure true forecasting
  generalization to *future* data.
- **Fix (Phase 6):** add a strict, gap-separated temporal split (train = early,
  val = middle, test = future) with an embargo gap of `LOOKBACK` rows between
  splits so no test window overlaps any train window. Report both numbers and
  be explicit about which is which.

### B. Persistence-dominated task → misleading "accuracy"
pH at 1-minute resolution barely changes step to step (mean |ΔpH| ≈ 0.007 on the
paper subset). A trivial "predict last value" baseline already achieves low
error. High R² therefore largely reflects autocorrelation, not learned physics.
- **Fix:** add a **persistence baseline** and report skill *relative* to it, so
  the GRU's real contribution is visible. (Phases 4, 13.)

### C. Model-selection metric mismatch
`train_one` evaluates the **final-epoch** weights but selects the best *seed* by
**final-epoch val loss**. There is no early-stopping/best-checkpoint restore, so
a lucky/unlucky last epoch can dominate.
- **Fix (Phase 5):** proper best-on-validation checkpointing + early stopping +
  LR scheduler, with the test set untouched until the end.

### D. Loss function sensitive to spikes
MSE loss + a few dosing-transient spikes explains the RMSE≫MAE gap. Not a bug,
but Huber loss is worth testing for robustness (Phase 5) and the spikes should
be characterized (Phase 3), not silently dropped.

### E. Reproducibility gaps
- `torch.use_deterministic_algorithms(False)` + non-pinned cuDNN → runs are only
  approximately reproducible. Acceptable on CPU; document it.
- DataLoader shuffling uses the global seed (fine) but `num_workers` default is
  0 (fine for determinism).

### F. Evaluation completeness
Only RMSE/MAE/R²/MAPE + one tolerance band are reported.
- **Fix (Phase 2):** add MedAE, MaxAE, and tolerance bands at ±0.01/0.03/0.05/0.10.

### G. No confidence / no failure simulation / no efficiency-for-edge
The project stops at accuracy. The paper's *purpose* is sensor-lifespan
extension via a hybrid virtual/physical strategy (their Algorithm 4), which is
NOT implemented.
- **Fix (Phases 7, 8, 10):** confidence estimation + switching policy + sensor
  failure simulation + edge efficiency profiling. This is where real novelty
  lives.

## Over/under-fitting, outliers, imbalance
- **Overfitting:** low risk — model is tiny (~24k params), train/val losses track
  closely in logs.
- **Underfitting:** not evident; train loss reaches ~1e-4.
- **Outliers:** a handful of pH dosing transients drive RMSE (quantified in
  Phase 3). Not removed — they are real events.
- **Imbalance:** N/A (regression).

## Decision
Preserve the whole existing pipeline as the **reproducible paper-aligned
baseline** (do not rewrite). Add new, clearly separated modules for: extended
metrics, error analysis, strong baselines, a validation-driven improved GRU, a
leakage-free temporal split, confidence-aware switching, failure simulation,
edge profiling, and a demo dashboard. Always report paper / reproduced-baseline
/ improved separately.
