# Honest Assessment, Limitations & Practical Impact

This document exists to make the project **jury-proof**: it states plainly what
the numbers do and do not mean, the limitations, and a clearly-labelled estimate
of practical value. Every measured number is from `results/metrics/`.

---

## 1. "Is the 94% good?" — honest interpretation

**What the number is:** on the leakage-free (honest) test, CARD keeps
**94.5% of predictions within ±0.05 pH** of the true value. It is correctly
computed. But it is a **within-tolerance rate, not classification accuracy**, and
by itself it does **not** prove the model is powerful. Here is why:

| Model | within ±0.05 pH |
|---|---|
| Persistence ("just repeat the last reading") | 93.8% |
| Improved GRU (ours) | 94.4% |
| **CARD (ours)** | **94.5%** |

The trivial **Persistence** predictor — which does nothing but output the
previous pH — already reaches **93.8%**. This is because hydroponic pH changes
very slowly minute-to-minute, so the signal is inherently easy to track. Our
model beats the trivial baseline only slightly on this metric.

**Therefore we do NOT claim the 94% proves superiority.** We state openly that on
this smooth signal Persistence is a near-ceiling baseline. The correct framing:

> "94.5% within ±0.05 pH is measured honestly, but on this slow-moving signal
> even a trivial predictor reaches ~94%, so this number alone is not our
> contribution. Our model's value is that it knows when it is uncertain, cuts
> physical-probe usage by 76%, and survives sensor failures — things a trivial
> predictor cannot do."

**Never present R² (e.g. 0.65) as '65% accuracy'** — R² is 'fraction of variance
explained', a different quantity.

---

## 2. Where the model DOES add real value (measured)

These are capabilities the trivial baseline cannot provide:

| Capability | Measured result |
|---|---|
| Physical-probe usage reduction (confidence switching) | **75.6% fewer reads** |
| Budget-controlled sampling (CASC) | hits 5/10/20% targets within ~0.2% |
| Accuracy under 30% sensor faults — proposed vs traditional | **R² 0.89 vs −6.9** |
| Model size / speed | 10,733 params · ~0.9 ms CPU · 6.8 KB ONNX |

---

## 3. Error distribution (tolerance-band breakdown)

Honest test, 1,544 samples. "within ±0.05" is the sum of the first three bands.

| Model | ≤0.01 | 0.01–0.03 | 0.03–0.05 | 0.05–0.10 | >0.10 |
|---|---|---|---|---|---|
| Persistence | 34.1% | 43.3% | 16.3% | 6.0% | 0.2% |
| Improved GRU | 32.8% | 43.7% | 17.9% | 5.2% | 0.3% |
| CARD (ours) | 33.9% | 42.7% | 17.9% | 5.2% | 0.3% |

Most predictions fall in the two smallest bands, and very few exceed 0.10 pH —
consistent across models, again reflecting the smoothness of the signal.

---

## 4. Limitations (stated openly)

1. **Single hydroponic run.** The dataset is ~one month of one system, so we
   cannot test generalisation across crops, seasons, or hardware. Cross-cycle
   validation is future work.
2. **Persistence is near-ceiling on clean data.** Because pH moves slowly, no
   model can pull far ahead of "repeat last value" on this signal; our models
   match rather than dominate it on raw accuracy.
3. **Simulated faults.** The sensor-failure robustness (Phase 8) uses injected
   faults (missing/frozen/noisy/drift/calibration). They are realistic but
   simulated, not recorded real failures.
4. **One-step-ahead only.** We forecast 1 minute ahead. Multi-step forecasting
   (5/10/30 min), where a smart model should clearly beat persistence, is
   identified as the most valuable next experiment.
5. **Lifespan is estimated, not measured.** See §5 — we report the measured
   usage reduction and a first-order lifespan estimate, not a measured probe
   life.
6. **CPU-only training.** Results are hardware-independent for these tiny
   models, but we did not use the paper's GPU.

---

## 5. Practical impact — cost & lifespan ESTIMATE

> ⚠ **ESTIMATE ONLY.** Built from the *measured* 75.6% usage reduction plus a
> first-order "wear ∝ usage" assumption. Actual probe life depends on chemistry,
> temperature, and handling, which we did not measure.

Assumptions: probe ≈ $45, labour ≈ $15/replacement, baseline life ≈ 6 months
(manufacturer, continuous use), 3-year horizon.

| Quantity | Baseline (always-physical) | Proposed (hybrid) |
|---|---|---|
| Physical-probe usage | 100% | 24.4% (measured) |
| Estimated probe life | 6 months | ~24.6 months (~4.1×) |
| Replacements over 3 yrs | 6.0 | ~1.5 |
| Estimated cost over 3 yrs | $360 | ~$88 |
| **Estimated savings** | — | **~$272 (75.6%)** |

The lifespan multiplier is capped for realism; the point is directional: a large
usage reduction should translate into materially longer probe life and lower
cost, but the exact figure requires a dedicated wear study.

---

## 6. Future work
- Multi-step forecasting (5/10/30 min) and multi-seed significance testing.
- Cross-cycle / cross-crop validation on additional datasets.
- A real (not simulated) fault dataset and a measured probe-wear study.
- On-device deployment trial on a Raspberry Pi with the exported ONNX model.

---

*Sources:* `phase7_confidence.json`, `phase8_failure.json`,
`tolerance_bands.json`, `cost_lifespan_estimate.json`,
`model_comparison_all.json`.
