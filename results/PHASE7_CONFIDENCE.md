# Phase 7 — Confidence-aware hybrid virtual/physical pH sensor

## Mechanism
Two reliability signals drive a hysteresis switching policy:

1. **MC-Dropout uncertainty** — dropout kept active at inference; K=30 stochastic
   passes; prediction std = epistemic uncertainty (converted to pH units).
2. **Running error EMA** — whenever the physical probe is read, update an
   exponential moving average of |virtual − physical| (paper Algorithm 4).

Policy: start virtual-dominant; request the physical probe when uncertainty
> u_high **or** EMA error > e_high; relax back to virtual when both are low;
plus a slow safety cadence (read physical at least every T_base steps).

**Thresholds are calibrated on the VALIDATION set only** (u_low/u_high = 60th/90th
percentile of validation uncertainty; e_low/e_high = 0.03/0.05 pH). The test
stream is untouched during calibration.

## Measured result (test stream, temporal split, 1,544 steps)

| Quantity | Value |
|---|---|
| Physical reads — always-on | 1,544 (100%) |
| Physical reads — hybrid | 376 (24.4%) |
| **Reduction in physical probe use** | **75.6%** |
| Pure-virtual accuracy | R²=0.650, MAE=0.0196, ±0.05=94.3% |
| **Hybrid system accuracy** | **R²=0.690, MAE=0.0163, ±0.05=94.4%** |

## Interpretation (honest)
- The hybrid controller cuts physical pH measurements by **~76%** while
  **improving** accuracy over pure-virtual (it pulls in the real probe exactly
  when the model is unsure or drifting).
- We report only the **measured reduction in probe activations**. We do **not**
  claim a specific lifespan multiplier — that would require a probe-degradation
  model we did not fit. Electrode wear is generally monotonic in cumulative
  usage, so a ~76% usage reduction is expected to extend service life, but the
  exact factor depends on probe chemistry and duty cycle.
- This is the concrete value a virtual sensor adds beyond a trivial persistence
  predictor: persistence cannot tell you *when it is wrong*; the confidence
  mechanism can.

Artifacts: `results/metrics/phase7_confidence.json`,
`results/models/phase7_hybrid_stream.npz`.
