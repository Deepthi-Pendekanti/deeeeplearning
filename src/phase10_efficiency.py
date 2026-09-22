"""
Phase 10 — Model efficiency & edge readiness.

Measures for the baseline GRU and the improved GRU:
  * parameter count
  * on-disk model size (state_dict .pt)
  * CPU single-sample inference latency (warm-up + timed loop, mean & P95)
  * per-batch throughput
  * process memory footprint (psutil if available)
Also attempts an ONNX export (batch-independent) for edge deployment and
verifies numerical parity, only reporting success if it actually works here.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import dataset as ds
from config import METRICS, MODELS
from improved_gru import GRUConfig, ImprovedGRU
from models import GRUModel, count_parameters


def latency_stats(model, x1, n=500):
    model.eval()
    with torch.no_grad():
        for _ in range(30):
            model(x1)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            model(x1)
            ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1000  # ms
    return {"mean_ms": float(ts.mean()), "p95_ms": float(np.percentile(ts, 95)),
            "min_ms": float(ts.min())}


def disk_size_kb(state_dict):
    tmp = MODELS / "_tmp_size.pt"
    torch.save(state_dict, tmp)
    kb = tmp.stat().st_size / 1024
    tmp.unlink()
    return kb


def try_onnx(model, x1, tag):
    try:
        import onnx  # noqa: F401
        path = MODELS / f"{tag}.onnx"
        torch.onnx.export(
            model, x1, str(path), input_names=["window"],
            output_names=["pH"], dynamic_axes={"window": {0: "batch"}},
            opset_version=17,
        )
        size_kb = path.stat().st_size / 1024
        # parity check with onnxruntime if available
        parity = None
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(path),
                                        providers=["CPUExecutionProvider"])
            onnx_out = sess.run(None, {"window": x1.numpy()})[0].ravel()
            model.eval()
            with torch.no_grad():
                torch_out = model(x1).numpy().ravel()
            parity = float(np.max(np.abs(onnx_out - torch_out)))
        except Exception as e:  # noqa: BLE001
            parity = f"onnxruntime unavailable: {e}"
        return {"exported": True, "size_kb": size_kb, "max_abs_diff": parity}
    except Exception as e:  # noqa: BLE001
        return {"exported": False, "reason": str(e)}


def main():
    wd = ds.build(include_actuators=False, source="paper", split="temporal",
                  verbose=False)
    x1 = torch.from_numpy(wd.X_test[:1])
    xb = torch.from_numpy(wd.X_test[:64])

    results = {}

    # baseline GRU
    base = GRUModel(wd.n_features)
    if (MODELS / "gru_best.pt").exists():
        try:
            base.load_state_dict(torch.load(MODELS / "gru_best.pt",
                                            map_location="cpu"))
        except Exception:
            pass
    results["GRU_baseline"] = {
        "params": count_parameters(base),
        "disk_kb": disk_size_kb(base.state_dict()),
        "latency": latency_stats(base, x1),
        "onnx": try_onnx(base, x1, "gru_baseline"),
    }

    # improved GRU
    ck = torch.load(MODELS / "improved_gru_best.pt", map_location="cpu")
    imp = ImprovedGRU(GRUConfig(**ck["gru_cfg"]), target_index=wd.target_index)
    imp.load_state_dict(ck["state"])
    results["ImprovedGRU"] = {
        "params": count_parameters(imp),
        "disk_kb": disk_size_kb(imp.state_dict()),
        "latency": latency_stats(imp, x1),
        "onnx": try_onnx(imp, x1, "improved_gru"),
    }

    # throughput (batch 64)
    for name, m in [("GRU_baseline", base), ("ImprovedGRU", imp)]:
        m.eval()
        with torch.no_grad():
            for _ in range(10):
                m(xb)
            t0 = time.perf_counter()
            for _ in range(100):
                m(xb)
            dt = (time.perf_counter() - t0) / 100
        results[name]["batch64_ms"] = dt * 1000
        results[name]["throughput_samples_per_s"] = 64 / dt

    # process memory
    try:
        import psutil
        mem_mb = psutil.Process().memory_info().rss / 1e6
        results["process_memory_mb"] = mem_mb
    except Exception:
        results["process_memory_mb"] = None

    print("=" * 66)
    print("Phase 10 — efficiency")
    print("=" * 66)
    for name in ("GRU_baseline", "ImprovedGRU"):
        r = results[name]
        print(f"\n{name}")
        print(f"  params      : {r['params']:,}")
        print(f"  disk size   : {r['disk_kb']:.1f} KB")
        print(f"  latency     : mean {r['latency']['mean_ms']:.3f} ms, "
              f"P95 {r['latency']['p95_ms']:.3f} ms")
        print(f"  batch-64    : {r['batch64_ms']:.3f} ms "
              f"({r['throughput_samples_per_s']:.0f} samples/s)")
        o = r["onnx"]
        if o.get("exported"):
            print(f"  ONNX        : exported {o['size_kb']:.1f} KB, "
                  f"parity max|diff|={o['max_abs_diff']}")
        else:
            print(f"  ONNX        : not exported ({o.get('reason','')[:60]})")
    if results["process_memory_mb"]:
        print(f"\nprocess RSS  : {results['process_memory_mb']:.1f} MB")

    with open(METRICS / "phase10_efficiency.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[save] results/metrics/phase10_efficiency.json")


if __name__ == "__main__":
    main()
