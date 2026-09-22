# Phase 11 — Final System Architecture

```
                    ┌──────────────────────────────────────────┐
                    │            IoT Sensor Layer                │
                    │  pH · TDS · water level · DHT temp/humid · │
                    │  water temp   (+ actuator states)          │
                    └───────────────────┬────────────────────────┘
                                        │ 10 s samples
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │           Data Acquisition                 │
                    │  Raspberry-Pi edge logger → local store    │
                    └───────────────────┬────────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │            Preprocessing                   │
                    │  • isDefault==1 → linear interpolation      │
                    │  • physical range check (ValidateAndFilter) │
                    │  • 10 s → 1 min time-based resampling        │
                    │  • MinMax scaling (train-fit; no leakage)    │
                    └───────────────────┬────────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │          Feature Engineering               │
                    │  • 6 sensor channels                        │
                    │  • look-back window of 10 steps             │
                    │  • persistence reference (last pH)          │
                    └───────────────────┬────────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │        GRU Virtual pH Sensor               │
                    │  ImprovedGRU (delta head): predicts the pH  │
                    │  CHANGE, added to the last pH. 10,733 params│
                    │  ~0.9 ms CPU inference, 47 KB / 6.8 KB ONNX │
                    └───────────────────┬────────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │     Prediction + Confidence Estimation     │
                    │  • MC-Dropout uncertainty (K passes → std)  │
                    │  • running |virtual−physical| error EMA     │
                    └───────────────────┬────────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │          Reliability Decision              │
                    │  hysteresis thresholds (u_low/u_high,       │
                    │  e_low/e_high) + safety cadence T_base      │
                    └───────┬───────────────────────┬────────────┘
                            │ HIGH confidence        │ LOW confidence
                            ▼                        ▼
                 ┌────────────────────┐    ┌────────────────────────┐
                 │ Use virtual pH      │    │ Activate physical pH    │
                 │ (probe idle → wear  │    │ probe; refresh error    │
                 │  reduction)         │    │ EMA; override if needed │
                 └─────────┬──────────┘    └───────────┬────────────┘
                           └───────────┬───────────────┘
                                       ▼
                    ┌──────────────────────────────────────────┐
                    │        Monitoring / Dashboard              │
                    │  live pH, confidence, mode, sensor status, │
                    │  recent accuracy, low-confidence alerts,    │
                    │  physical-usage reduction                   │
                    └───────────────────┬────────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │        Sensor Usage Optimization           │
                    │  measured ~76% fewer physical reads in     │
                    │  steady state; graceful fault fallback      │
                    └──────────────────────────────────────────┘
```

## Component notes

- **IoT Sensor Layer** — six physicochemical channels sampled every 10 s. pH is
  the target the system aims to serve without constantly reading the fragile
  physical probe.
- **Data Acquisition** — edge logging on a Raspberry-Pi-class device; no cloud
  dependency required for inference.
- **Preprocessing** — repairs sensor-failure rows (linear interpolation),
  removes physically-impossible readings, resamples to 1-minute cadence, and
  scales with a MinMax scaler fit only on the training region (leakage-safe).
- **Feature Engineering** — a 10-step look-back window over the 6 sensors; the
  last observed pH is carried as the persistence reference for the delta head.
- **GRU Virtual Sensor** — the ImprovedGRU predicts the *change* in pH (delta
  head) rather than absolute pH, which is what lets it generalize to future data
  (Phase 5). Compact (10,733 params) and edge-fast (~0.9 ms CPU, 6.8 KB ONNX).
- **Prediction + Confidence** — MC-Dropout provides epistemic uncertainty; a
  running EMA of virtual-vs-physical error tracks drift.
- **Reliability Decision** — hysteresis thresholds (calibrated on validation)
  plus a slow safety cadence decide when the physical probe is actually needed.
- **Monitoring / Dashboard** — surfaces live state for operators and judges
  (Phase 12).
- **Sensor Usage Optimization** — the measured outcome: ~76% fewer physical
  reads in steady state (Phase 7) and robust fallback during faults (Phase 8).
```
