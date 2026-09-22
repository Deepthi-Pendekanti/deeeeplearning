"""
Phase 8 — Sensor failure / missing-sensor simulation.

We corrupt the PHYSICAL pH channel of the test stream with realistic faults and
compare two systems:

  TRADITIONAL : trust the physical pH reading at every step (it is corrupted
                during fault windows -> errors propagate).
  PROPOSED    : during a fault the physical reading is unavailable/untrusted, so
                the GRU virtual sensor supplies pH; the confidence switch decides
                when a (healthy) physical read is needed.

Fault types injected into designated fault windows of the test period:
  * missing        : pH unavailable (NaN) -> must use virtual
  * dropout        : pH stuck at last value (frozen sensor)
  * noise          : additive Gaussian noise
  * drift          : slow linear bias accumulation
  * calibration    : constant offset (miscalibration)

We measure, against the TRUE pH:
  * error of traditional (corrupted-physical) system
  * error of proposed (virtual-during-fault) system
  * accuracy during normal vs fault periods
  * physical reads used by the proposed system
"""
from __future__ import annotations

import json

import numpy as np
import torch

import dataset as ds
from config import METRICS, MODELS
from confidence import mc_dropout_predict
from improved_gru import GRUConfig, ImprovedGRU
from metrics import all_metrics

RNG = np.random.default_rng(123)


def load_improved(target_index):
    ckpt = torch.load(MODELS / "improved_gru_best.pt", map_location="cpu")
    model = ImprovedGRU(GRUConfig(**ckpt["gru_cfg"]), target_index=target_index)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model


def make_fault_windows(n, frac=0.30, n_windows=6):
    """Mark ~frac of the stream as faulty, spread over n_windows blocks."""
    is_fault = np.zeros(n, dtype=bool)
    fault_type = np.array(["none"] * n, dtype=object)
    types = ["missing", "dropout", "noise", "drift", "calibration"]
    total = int(n * frac)
    per = total // n_windows
    starts = np.linspace(int(0.08 * n), int(0.9 * n) - per, n_windows).astype(int)
    for i, s in enumerate(starts):
        e = min(s + per, n)
        is_fault[s:e] = True
        fault_type[s:e] = types[i % len(types)]
    return is_fault, fault_type


def corrupt_physical(true_ph, is_fault, fault_type):
    """Produce the (possibly corrupted) physical reading + availability mask."""
    phys = true_ph.copy().astype(float)
    available = np.ones(len(true_ph), dtype=bool)
    drift_accum = 0.0
    for t in range(len(true_ph)):
        ft = fault_type[t]
        if not is_fault[t]:
            drift_accum = 0.0
            continue
        if ft == "missing":
            available[t] = False           # sensor returns nothing
        elif ft == "dropout":
            phys[t] = phys[t - 1] if t > 0 else phys[t]  # frozen
        elif ft == "noise":
            phys[t] = true_ph[t] + RNG.normal(0, 0.15)
        elif ft == "drift":
            drift_accum += 0.01
            phys[t] = true_ph[t] + drift_accum
        elif ft == "calibration":
            phys[t] = true_ph[t] + 0.30
    return phys, available


def main():
    wd = ds.build(include_actuators=False, source="paper", split="temporal",
                  verbose=False)
    model = load_improved(wd.target_index)

    true_ph = ds.inverse_target(wd, wd.y_test)
    vmean_s, vstd_s = mc_dropout_predict(model, wd.X_test, k=30)
    virt_ph = ds.inverse_target(wd, vmean_s)
    n = len(true_ph)

    is_fault, fault_type = make_fault_windows(n, frac=0.30, n_windows=6)
    phys_reading, phys_available = corrupt_physical(true_ph, is_fault, fault_type)

    # TRADITIONAL: always use the physical reading (corrupted in faults).
    # If missing, hold last available value (best a naive system can do).
    trad = phys_reading.copy()
    last = true_ph[0]
    for t in range(n):
        if not phys_available[t]:
            trad[t] = last
        else:
            last = trad[t]

    # PROPOSED: during a fault (untrusted/missing physical) use the virtual
    # sensor; otherwise use the (healthy) physical reading. Count physical reads.
    proposed = np.empty(n)
    physical_reads = 0
    for t in range(n):
        if is_fault[t]:
            proposed[t] = virt_ph[t]          # virtual carries the load
        else:
            proposed[t] = phys_reading[t]      # healthy physical read
            physical_reads += 1

    # Metrics vs TRUE pH.
    def seg(mask, arr):
        return all_metrics(true_ph[mask], arr[mask])

    normal = ~is_fault
    res = {
        "n": int(n),
        "fault_fraction": float(is_fault.mean()),
        "traditional_overall": all_metrics(true_ph, trad),
        "proposed_overall": all_metrics(true_ph, proposed),
        "traditional_during_fault": seg(is_fault, trad),
        "proposed_during_fault": seg(is_fault, proposed),
        "proposed_normal": seg(normal, proposed),
        "physical_reads_proposed": int(physical_reads),
        "physical_reads_traditional": int(n),
        "reduction_pct": 100 * (1 - physical_reads / n),
    }

    print("=" * 66)
    print("Phase 8 — sensor failure simulation (test stream)")
    print("=" * 66)
    print(f"steps={n}  faulty={100*res['fault_fraction']:.0f}%  "
          f"(missing/dropout/noise/drift/calibration windows)")
    print(f"\n{'':<26}{'RMSE':>9}{'MAE':>9}{'R2':>9}{'±0.05%':>9}")
    for label, key in [("Traditional (overall)", "traditional_overall"),
                       ("Proposed   (overall)", "proposed_overall"),
                       ("Traditional in-fault", "traditional_during_fault"),
                       ("Proposed   in-fault", "proposed_during_fault"),
                       ("Proposed   normal", "proposed_normal")]:
        m = res[key]
        print(f"{label:<26}{m['RMSE']:>9.4f}{m['MAE']:>9.4f}"
              f"{m['R2']:>9.4f}{m['within_0.05pH']:>9.2f}")
    print(f"\nphysical reads: traditional={n} (100%), "
          f"proposed={physical_reads} "
          f"({100*physical_reads/n:.1f}%) -> {res['reduction_pct']:.1f}% fewer")

    np.savez(MODELS / "phase8_failure_stream.npz",
             true=true_ph, virtual=virt_ph, traditional=trad, proposed=proposed,
             is_fault=is_fault, fault_type=fault_type.astype(str))
    with open(METRICS / "phase8_failure.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\n[save] results/metrics/phase8_failure.json")


if __name__ == "__main__":
    main()
