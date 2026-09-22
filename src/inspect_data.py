"""Quick inspection of the raw Kaggle CSVs to confirm schema and pick the source."""
import glob
import os

import pandas as pd

for f in sorted(glob.glob("data/raw/*.csv")):
    d = pd.read_csv(f)
    print("=" * 72)
    print(os.path.basename(f))
    print(f"  rows: {len(d):,} | cols: {len(d.columns)}")
    if "isDefault" in d.columns:
        print(f"  isDefault==1: {int((d['isDefault'] == 1).sum()):,}")
    if "timestamp" in d.columns:
        ts = pd.to_datetime(d["timestamp"], errors="coerce")
        print(f"  time span: {ts.min()} -> {ts.max()} | bad ts: {int(ts.isna().sum())}")
    else:
        print("  (no timestamp column)")
    print(f"  pH range: {d['pH'].min():.3f} - {d['pH'].max():.3f} | pH NaN: {int(d['pH'].isna().sum())}")
    for c in ["pH_reducer", "add_water", "nutrients_adder", "humidifier", "ex_fan"]:
        if c in d.columns:
            print(f"  {c}: dtype={d[c].dtype}, sample uniques={list(pd.Series(d[c].dropna().unique())[:4])}")
