"""Pretty-print a phase4-style results JSON as a table."""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
d = json.load(open(path))
print(f"\n{path.name}")
print(f"{'Model':<18}{'R2':>9}{'MAE':>9}{'RMSE':>9}{'+-0.05%':>10}"
      f"{'params':>10}{'lat(ms)':>10}")
for k, v in d.items():
    p = v.get("params")
    p = -1 if p is None else p
    print(f"{k:<18}{v['R2']:>9.4f}{v['MAE']:>9.4f}{v['RMSE']:>9.4f}"
          f"{v['within_0.05pH']:>10.2f}{p:>10}{v.get('latency_ms', 0):>10.3f}")
