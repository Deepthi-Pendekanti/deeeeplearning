# Phase 12 — Hackathon demo dashboard

A self-contained, dependency-free dashboard that a judge understands in
30–60 seconds.

## How to run
1. Generate the data (once): `python src/phase12_export_demo.py`
   → writes `results/demo_data.json`.
2. Open `results/dashboard.html` in any browser (double-click).
   It replays the leakage-free test stream with injected sensor faults.

No server, no build step, no internet.

## What it shows (all live-updating)
- **Reported pH** — the value the system currently serves.
- **Live chart** — true pH (white) vs virtual GRU (green) vs system output
  (blue); fault windows shaded amber. During faults the blue line follows the
  green virtual sensor and stays on the true signal while a naive physical-only
  reading would diverge.
- **Mode badge** — VIRTUAL vs PHYSICAL, switching in real time.
- **Confidence bar** — model confidence (from MC-Dropout uncertainty).
- **Low-confidence alert** — fires when confidence drops or a fault occurs.
- **Sensor status** — healthy / low-confidence / FAULT (with fault type).
- **Live diagnostics** — physical pH, virtual pH, |Δ| vs true, rolling
  within-±0.05 accuracy, current fault type.
- **KPIs** — physical-probe usage reduction (75.6%), hybrid R² (0.69),
  proposed-vs-traditional R² under faults (0.895 vs −6.9).
- **Model footprint** — 10,733 params · ~0.9 ms CPU · 6.8 KB ONNX.

Every number shown is a **measured** experimental result, not a placeholder.

Artifacts: `results/dashboard.html`, `results/demo_data.json`.
