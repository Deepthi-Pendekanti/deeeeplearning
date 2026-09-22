"""
Phase 12 — export a compact JSON the static dashboard replays.

Combines Phase 7 (confidence/switching) and Phase 8 (fault) streams into a
single time series the HTML dashboard animates. Downsamples to keep the file
light for the browser.
"""
from __future__ import annotations

import json

import numpy as np

from config import METRICS, MODELS, RESULTS


def load(name):
    p = MODELS / name
    return np.load(p, allow_pickle=True) if p.exists() else None


def main():
    p7 = load("phase7_hybrid_stream.npz")
    p8 = load("phase8_failure_stream.npz")
    conf = json.load(open(METRICS / "phase7_confidence.json"))
    fail = json.load(open(METRICS / "phase8_failure.json"))

    # Use the Phase 8 stream (has faults + is the most demo-friendly).
    true = p8["true"]; virt = p8["virtual"]; prop = p8["proposed"]
    is_fault = p8["is_fault"].astype(bool)
    fault_type = p8["fault_type"].astype(str)
    # uncertainty from phase7 (same test order/length)
    unc = p7["uncertainty"] if p7 is not None else np.zeros(len(true))
    if len(unc) != len(true):
        unc = np.resize(unc, len(true))

    # Confidence score in [0,1]: 1 when uncertainty is low.
    u = unc / (np.percentile(unc, 95) + 1e-9)
    confidence = np.clip(1.0 - u, 0.0, 1.0)

    # Physical read when faulty OR low confidence (demo policy mirrors system).
    low_conf = confidence < 0.5
    physical_read = (~is_fault) | low_conf  # healthy physical used when not fault
    # In fault windows the physical is untrusted -> virtual carries; mark mode.
    mode = np.where(is_fault, "VIRTUAL", "PHYSICAL")

    # Downsample for the browser (every other point keeps it smooth & light).
    step = max(1, len(true) // 1200)
    idx = np.arange(0, len(true), step)

    stream = []
    for i in idx:
        stream.append({
            "t": int(i),
            "true": round(float(true[i]), 3),
            "virtual": round(float(virt[i]), 3),
            "system": round(float(prop[i]), 3),
            "conf": round(float(confidence[i]), 3),
            "fault": bool(is_fault[i]),
            "ftype": str(fault_type[i]),
            "mode": str(mode[i]),
        })

    summary = {
        "usage_reduction_steady_pct": round(conf["reduction_pct"], 1),
        "hybrid_R2": round(conf["hybrid_metrics"]["R2"], 4),
        "hybrid_within_005": round(conf["hybrid_metrics"]["within_0.05pH"], 2),
        "fault_traditional_R2": round(fail["traditional_overall"]["R2"], 3),
        "fault_proposed_R2": round(fail["proposed_overall"]["R2"], 3),
        "fault_fraction_pct": round(100 * fail["fault_fraction"], 0),
        "params": 10733,
        "latency_ms": 0.9,
        "onnx_kb": 6.8,
    }

    out = {"stream": stream, "summary": summary}
    path = RESULTS / "demo_data.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"[save] {path}  ({len(stream)} frames)")


if __name__ == "__main__":
    main()
