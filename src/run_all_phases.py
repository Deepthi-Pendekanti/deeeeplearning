"""
One-command runner for the full 13-phase upgraded pipeline.

Assumes data is already downloaded + preprocessed (run src/run_all.py first, or
at least download_data.py + preprocess.py [--paper-subset]).

Order:
  Phase 2  reproducible baseline + persistence (random + temporal)
  Phase 3  top-error analysis
  Phase 4  strong baselines (random + temporal)
  Phase 5  improved-GRU validation search + test
  Phase 7  confidence-aware hybrid switching
  Phase 8  sensor-failure simulation
  Phase 9  figures + loss curve
  Phase 10 efficiency + ONNX export
  Phase 12 export demo data for the dashboard

Phases 1/6/11/13 are written documents in results/ (no compute).
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
PY = sys.executable


def step(title, args):
    print("\n" + "#" * 72 + f"\n# {title}\n" + "#" * 72, flush=True)
    r = subprocess.run([PY, "-u"] + args, cwd=str(SRC))
    if r.returncode != 0:
        print(f"[run_all_phases] FAILED: {title} (exit {r.returncode})")
        sys.exit(r.returncode)


def main():
    step("Phase 2 — baseline (paper/random)", ["phase2_baseline.py", "--source", "paper", "--split", "random"])
    step("Phase 2 — baseline (paper/temporal)", ["phase2_baseline.py", "--source", "paper", "--split", "temporal"])
    step("Phase 3 — error analysis", ["phase3_error_analysis.py"])
    step("Phase 4 — strong baselines", ["phase4_baselines.py", "--source", "paper", "--splits", "random", "temporal"])
    step("Phase 5 — improved GRU search", ["phase5_improve.py"])
    step("Phase 7 — confidence switching", ["phase7_confidence.py"])
    step("Phase 8 — failure simulation", ["phase8_failure_sim.py"])
    step("Phase 9 — figures", ["phase9_plots.py"])
    step("Phase 9 — loss curve", ["phase9_losscurve.py"])
    step("Phase 10 — efficiency + ONNX", ["phase10_efficiency.py"])
    step("Phase 12 — export demo data", ["phase12_export_demo.py"])
    print("\n[run_all_phases] DONE. See results/FINAL_REPORT.md, results/*.md, "
          "results/figures/, and open results/dashboard.html")


if __name__ == "__main__":
    main()
