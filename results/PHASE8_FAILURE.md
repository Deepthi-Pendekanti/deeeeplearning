# Phase 8 — Sensor failure / missing-sensor simulation

30% of the test stream is corrupted across six windows, one per fault type:
**missing** (no reading), **dropout** (frozen/stuck value), **noise** (additive
Gaussian σ=0.15 pH), **drift** (slow linear bias), **calibration** (constant
+0.30 pH offset).

Two systems compared against the TRUE pH:
- **Traditional** — trusts the physical reading every step (corrupted in faults;
  holds last value when missing).
- **Proposed** — during a fault the physical reading is untrusted/unavailable,
  so the GRU virtual sensor supplies pH; healthy physical reads are used
  otherwise.

## Measured results (test stream, 1,544 steps)

| System | RMSE | MAE | R² | ±0.05 pH |
|---|---|---|---|---|
| Traditional (overall) | 0.1249 | 0.0431 | −6.92 | 85.49% |
| **Proposed (overall)** | **0.0144** | **0.0060** | **0.895** | **98.32%** |
| Traditional — during faults | 0.2284 | 0.1439 | −20.97 | 51.52% |
| **Proposed — during faults** | **0.0263** | **0.0202** | **0.708** | **94.37%** |
| Proposed — normal periods | 0.0000 | 0.0000 | 1.000 | 100.00% |

Physical reads: traditional 1,544 (100%) → proposed 1,082 (70.1%),
a **29.9% reduction** (fewer because the virtual sensor covers fault windows).

## Interpretation (honest)
- Under realistic pH-sensor faults, a naive always-physical pipeline degrades
  catastrophically (negative R²), because it propagates corrupted readings.
- The proposed virtual-sensor fallback keeps the system usable: overall RMSE
  drops ~9× (0.125 → 0.014) and even *during* fault windows it holds R²≈0.71.
- This is the concrete robustness benefit of a virtual sensor and cannot be
  achieved by a persistence predictor (which would itself freeze on a dropout
  fault and propagate a calibration offset).

Note: these are simulated faults injected for evaluation; the improvement
depends on fault severity/duration. Reported numbers are for the stated 30%
fault fraction and the specific fault parameters above.

Artifacts: `results/metrics/phase8_failure.json`,
`results/models/phase8_failure_stream.npz`.
