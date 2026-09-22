# Phase 3 — Why RMSE ≫ MAE (largest-error analysis)

Config analyzed: GRU, paper subset, random split (the same setup that produced
the reported RMSE≈0.03 / MAE≈0.015). Test set = 1,562 samples.

## Headline: RMSE is dominated by ~2 spike points

| Worst-k removed | Share of total squared error | RMSE without them |
|---|---|---|
| 1 | **49.4%** | 0.0216 |
| 5 | 68.4% | 0.0171 |
| 10 | 69.8% | 0.0168 |
| 20 | 71.9% | 0.0162 |
| 50 | 76.7% | 0.0149 |

The single largest error contributes ~half of the entire sum of squared errors.
Because RMSE squares errors, these few points dominate it, while MAE (which does
not square) stays low. This fully explains the RMSE (0.0304) vs MAE (0.0147) gap
and why the paper's specific test window gave a lower RMSE (0.0181): fewer such
spikes landed in their test partition.

## What the largest errors actually are

The top-2 errors coincide with physically implausible **single-step pH jumps**:

| rank | actual pH | pred pH | abs err | prev pH | ΔpH in 1 min | actuators active? |
|---|---|---|---|---|---|---|
| 1 | 6.59 | 5.74 | 0.846 | 5.76 | **+0.83** | none |
| 2 | 5.87 | 5.37 | 0.499 | 5.38 | **+0.49** | none |
| 3 | 5.97 | 5.86 | 0.106 | 5.86 | +0.11 | none |
| … | … | … | … | … | ≤0.09 | none |

A +0.83 pH jump within a single 1-minute step, with **no dosing/refill actuator
active**, is not chemically realistic — it is consistent with an uncaught sensor
artifact rather than a true event. Ranks 3–20 are ordinary ~0.05 pH errors.

Only 3 of the top-20 correspond to a true |ΔpH| > 0.10; the rest are normal.

## Decision (no silent deletion)

Per the project rules we do **not** drop these points blindly. Instead:
1. **Huber loss** (Phase 5) reduces the training influence of large residuals
   without discarding any data.
2. An **optional, documented** spike-validity filter (reject single-step pH
   jumps above a physically-motivated threshold as sensor artifacts) is offered
   as a cleaning variant and always reported explicitly alongside the unfiltered
   result.

Artifacts: `results/metrics/phase3_top20_errors.csv`.
