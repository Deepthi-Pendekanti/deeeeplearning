# Phase 6 — Temporal robustness (benchmark vs generalization)

The project now reports **two** numbers for every model, never one:

- **Benchmark (random split)** — windows shuffled into train/val/test. This is
  the paper-style protocol; it leaks temporally (adjacent windows share 9/10
  rows) so it *overstates* real forecasting ability. Kept only for
  comparability with the paper.
- **Generalization (temporal split)** — strict time order with a `LOOKBACK`-row
  embargo gap between train | val | test, so **no test window shares any row
  with a training window**. This is the honest estimate of forecasting a
  *future* period.

## Split construction (leakage-free temporal)

```
rows:  [ ---- train (early) ---- ][gap][ val ][gap][ ---- test (future) ---- ]
gap = LOOKBACK (=10) rows  ->  guarantees zero window overlap across splits
scaler: MinMax fit on the first 80% only (train region), applied to all
```

## Results side by side (paper subset)

| Model | Benchmark R² (random, leaky) | Generalization R² (temporal) | Gen. MAE | Gen. ±0.05 pH |
|---|---|---|---|---|
| Persistence | 0.9815 | 0.6500 | 0.0196 | 93.78% |
| ExtraTrees | 0.9837 | 0.6755 | 0.0199 | 95.08% |
| XGBoost | 0.9834 | 0.5504 | 0.0236 | 90.87% |
| GRU baseline | 0.9799 | 0.0304 | 0.0390 | 72.28% |
| **Improved GRU** | — | **0.6499** | **0.0196** | **94.43%** |

## Takeaways
1. The random-split R² (~0.98) is **not** a measure of generalization; every
   model including a trivial persistence predictor reaches it.
2. Under the honest temporal split the ranking changes completely: tree models
   and the improved (delta-head) GRU generalize; the original absolute-pH GRU
   does not.
3. The improved GRU generalizes to a genuinely future window and matches the
   strongest simple baselines, at a fraction of the parameters — this is the
   result we stand behind.

The dataset covers a single ~1-month hydroponic run, so cross-cycle / cross-crop
generalization cannot be tested here; that is stated as a limitation (Phase 13).
