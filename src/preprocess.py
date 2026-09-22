"""
Stage 3-4: Data cleaning and preprocessing, following the paper exactly.

Pipeline (Sections 3.3.5, 3.3.6, 3.3.10):
  1. Load raw 10-second-interval logs.
  2. Normalize column names to canonical internal names.
  3. Parse the timestamp and sort chronologically.
  4. Handle sensor-failure rows (isDefault == 1) with LINEAR interpolation (Eq. 8).
  5. Resample/interpolate 10 s -> 1 min using TIME-BASED interpolation (Eq. 9).
  6. Save the cleaned 1-minute dataset to data/processed/.

This module only produces the clean, resampled table. Normalization (MinMax)
and windowing happen later in dataset.py so that scalers can be fit on the
training split only (avoiding leakage).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DATA_RAW,
    DATA_PROCESSED,
    IS_DEFAULT_COLUMN,
    RESAMPLE_INTERVAL,
    VALID_RANGES,
)

# ----------------------------------------------------------------------------
# Column name normalization
# ----------------------------------------------------------------------------
# Maps many possible raw header spellings -> canonical internal names.
# Canonical names match src/config.py (SENSOR_FEATURES / ACTUATOR_FEATURES).
CANONICAL_MAP = {
    # timestamp
    "timestamp": "timestamp", "time": "timestamp", "datetime": "timestamp",
    "date_time": "timestamp", "created_at": "timestamp", "ts": "timestamp",
    # pH
    "ph": "pH", "ph_value": "pH", "phvalue": "pH", "ph_sensor": "pH",
    # TDS
    "tds": "TDS", "tds_value": "TDS", "ppm": "TDS",
    # water level
    "water_level": "water_level", "waterlevel": "water_level",
    "level": "water_level", "water_lvl": "water_level",
    # DHT temperature (environment/air temperature)
    "dht_temp": "DHT_temp", "dhttemp": "DHT_temp", "env_temp": "DHT_temp",
    "environment_temperature": "DHT_temp", "air_temp": "DHT_temp",
    "dht_temperature": "DHT_temp", "temperature": "DHT_temp",
    # DHT humidity
    "dht_humidity": "DHT_humidity", "dhthumidity": "DHT_humidity",
    "humidity": "DHT_humidity",
    # water temperature (DS18B20)
    "water_temp": "water_temp", "watertemp": "water_temp",
    "water_temperature": "water_temp", "ds18b20": "water_temp",
    # actuators
    "ph_reducer": "pH_reducer", "phreducer": "pH_reducer", "acid": "pH_reducer",
    "acid_pump": "pH_reducer", "ph_reduce": "pH_reducer",
    "add_water": "add_water", "addwater": "add_water", "water_pump": "add_water",
    "nutrients_adder": "nutrients_adder", "nutrient_adder": "nutrients_adder",
    "nutrients": "nutrients_adder", "nutrient": "nutrients_adder",
    "nutrient_pump": "nutrients_adder",
    "humidifier": "humidifier",
    "ex_fan": "ex_fan", "exfan": "ex_fan", "exhaust_fan": "ex_fan",
    "exhaustfan": "ex_fan", "fan": "ex_fan",
    # default / failure flag
    "isdefault": IS_DEFAULT_COLUMN, "is_default": IS_DEFAULT_COLUMN,
    "default": IS_DEFAULT_COLUMN, "isdefault_flag": IS_DEFAULT_COLUMN,
    # id
    "id": "id", "sl_no": "id", "sl": "id", "index": "id",
}


def _slug(name: str) -> str:
    """Lowercase, strip, collapse non-alphanumerics to underscores."""
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names where recognized; leave others slugged."""
    rename = {}
    for col in df.columns:
        key = _slug(col)
        rename[col] = CANONICAL_MAP.get(key, key)
    out = df.rename(columns=rename)
    # If duplicate canonical names appear, keep the first.
    out = out.loc[:, ~out.columns.duplicated()]
    return out


# Preferred source, in priority order. We want the RAW 50,570-row log that ALSO
# carries the authors' own isDefault labels and a timestamp, so we interpolate
# exactly the rows they flagged (Sec 3.3.5/3.3.6). Fallback to the plain raw log.
PREFERRED_SOURCES = [
    "IoTData_IsDefaultInterpolate",     # raw values + isDefault + timestamp (best)
    "IoTData --Raw--",                  # raw values + timestamp (no isDefault)
]


def find_raw_csv() -> Path:
    csvs = sorted(DATA_RAW.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSV found in {DATA_RAW}. Run src/download_data.py first."
        )
    # Honor the preferred-source ordering first.
    for stem in PREFERRED_SOURCES:
        for c in csvs:
            if c.name.startswith(stem):
                return c
    # Otherwise prefer the largest CSV (the main log).
    csvs.sort(key=lambda p: p.stat().st_size, reverse=True)
    return csvs[0]


def binarize_actuators(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ON/OFF actuator strings to 1/0 floats (leave numerics as-is)."""
    df = df.copy()
    on_off = {"on": 1.0, "off": 0.0, "true": 1.0, "false": 0.0,
              "1": 1.0, "0": 0.0}
    from config import ACTUATOR_FEATURES
    for c in ACTUATOR_FEATURES:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = (
                df[c].astype(str).str.strip().str.lower().map(on_off)
            )
    return df


def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = path or find_raw_csv()
    print(f"[load] reading {path.name}")
    df = pd.read_csv(path)
    print(f"[load] raw shape: {df.shape}")
    print(f"[load] raw columns: {list(df.columns)}")
    df = normalize_columns(df)
    df = binarize_actuators(df)
    print(f"[load] canonical columns: {list(df.columns)}")
    return df


def ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a proper datetime index, chronologically sorted."""
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        n_bad = df["timestamp"].isna().sum()
        if n_bad:
            print(f"[time] dropping {n_bad} rows with unparseable timestamps")
            df = df.dropna(subset=["timestamp"])
    else:
        # No timestamp column -> synthesize a 10-second cadence (Sec 3.3.5).
        print("[time] no timestamp column found; synthesizing 10s cadence")
        df = df.reset_index(drop=True)
        df["timestamp"] = pd.date_range(
            "2024-01-01", periods=len(df), freq="10s"
        )
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def numeric_sensor_columns(df: pd.DataFrame) -> list[str]:
    """Columns that carry sensor/actuator numeric signal (exclude id/flag/time)."""
    exclude = {"timestamp", "id", IS_DEFAULT_COLUMN}
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def interpolate_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eq. (8): replace sensor-failure values (isDefault == 1) via linear
    interpolation between the nearest valid points.

    We treat isDefault==1 rows as missing for the sensor signal columns, then
    apply pandas linear interpolation (index-position based == Eq. 8).
    """
    df = df.copy()
    signal_cols = numeric_sensor_columns(df)

    if IS_DEFAULT_COLUMN in df.columns:
        mask = df[IS_DEFAULT_COLUMN].fillna(0).astype(float) == 1
        n_default = int(mask.sum())
        print(f"[clean] isDefault==1 rows: {n_default} "
              f"({100 * n_default / max(len(df),1):.2f}%)")
        # Set the failed sensor readings to NaN so interpolation can fill them.
        df.loc[mask, signal_cols] = np.nan
    else:
        print("[clean] no isDefault column; interpolating existing NaNs only")

    # Also count pre-existing NaNs.
    pre_nan = int(df[signal_cols].isna().sum().sum())

    # Eq. (8) linear interpolation (position-based), fill ends by nearest valid.
    df[signal_cols] = (
        df[signal_cols]
        .interpolate(method="linear", limit_direction="both")
    )
    post_nan = int(df[signal_cols].isna().sum().sum())
    print(f"[clean] NaNs before/after linear interpolation: {pre_nan} -> {post_nan}")
    return df


def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Algorithm 2 'ValidateAndFilter' range check (Sec 3.2.4): readings outside
    physically-plausible bounds are sensor faults. Mark them NaN and repair via
    the same linear interpolation used for isDefault (Eq. 8).

    Bounds are broad physical limits (see VALID_RANGES), not crop setpoints, so
    this removes only impossible spikes (e.g. TDS<0, humidity>100%) without
    biasing pH toward the target range.
    """
    df = df.copy()
    total_flagged = 0
    for col, (lo, hi) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        bad = (df[col] < lo) | (df[col] > hi)
        n = int(bad.sum())
        if n:
            print(f"[range] {col}: {n} out-of-range readings "
                  f"(outside [{lo}, {hi}]) -> interpolated")
            df.loc[bad, col] = np.nan
            total_flagged += n
    if total_flagged:
        signal_cols = numeric_sensor_columns(df)
        df[signal_cols] = df[signal_cols].interpolate(
            method="linear", limit_direction="both")
    else:
        print("[range] no out-of-range readings found")
    return df


def resample_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eq. (9): time-based interpolation, resampling the 10 s stream to 1 minute.

    We set the timestamp as the index and use pandas time-based interpolation,
    which linearly interpolates against the actual time delta (Eq. 9), then
    sample on a regular 1-minute grid.
    """
    df = df.copy()
    signal_cols = numeric_sensor_columns(df)
    df = df.set_index("timestamp")

    # Keep only numeric signal columns for the resampled table.
    signal = df[signal_cols].astype(float)

    # Build a regular 1-minute grid spanning the data.
    grid = pd.date_range(signal.index.min(), signal.index.max(),
                         freq=RESAMPLE_INTERVAL)

    # Union original + grid index, time-interpolate, then select the grid.
    union_idx = signal.index.union(grid)
    resampled = (
        signal.reindex(union_idx)
        .interpolate(method="time")   # Eq. (9): time-based interpolation
        .reindex(grid)
    )
    resampled.index.name = "timestamp"
    print(f"[resample] {len(signal):,} rows @10s -> {len(resampled):,} rows @1min")
    return resampled.reset_index()


def run() -> pd.DataFrame:
    df = load_raw()
    df = ensure_timestamp(df)
    df = interpolate_defaults(df)
    df = validate_ranges(df)
    clean = resample_time(df)

    # Drop any residual NaNs at edges (shouldn't happen after both fills).
    before = len(clean)
    clean = clean.dropna().reset_index(drop=True)
    if len(clean) != before:
        print(f"[clean] dropped {before - len(clean)} edge rows with NaN")

    out_path = DATA_PROCESSED / "clean_1min.csv"
    clean.to_csv(out_path, index=False)
    print(f"[save] wrote {out_path} shape={clean.shape}")
    print(f"[save] columns: {list(clean.columns)}")
    return clean


def run_paper_subset() -> pd.DataFrame:
    """
    Produce a processed file from the AUTHORS' OWN cleaned subset
    (IoTData_25K_with_interpolation...csv): already isDefault-interpolated,
    already 1-min-equivalent, actuators already 0-1. This gives an
    apples-to-apples reproduction of the paper's headline metrics, which were
    computed on this smoother subset (pH ~5.34-6.82).

    We only normalize column names and (since it has no timestamp) keep row
    order. Output: data/processed/clean_paper_subset.csv
    """
    matches = sorted(DATA_RAW.glob("*with_interpolation*.csv"))
    if not matches:
        raise FileNotFoundError(
            "Authors' cleaned subset (*with_interpolation*.csv) not found in "
            f"{DATA_RAW}. Run download_data.py first.")
    path = matches[0]
    print(f"[paper-subset] reading {path.name}")
    df = pd.read_csv(path)
    df = normalize_columns(df)
    df = binarize_actuators(df)

    # Keep the sensor + actuator numeric columns (drop id).
    drop = [c for c in ("id", IS_DEFAULT_COLUMN) if c in df.columns]
    df = df.drop(columns=drop)

    # Repair any residual physical-range violations (defensive; usually none).
    df = validate_ranges(df)
    df = df.dropna().reset_index(drop=True)

    out_path = DATA_PROCESSED / "clean_paper_subset.csv"
    df.to_csv(out_path, index=False)
    print(f"[save] wrote {out_path} shape={df.shape}")
    print(f"[save] columns: {list(df.columns)}")
    print(f"[save] pH range: [{df['pH'].min():.3f}, {df['pH'].max():.3f}]")
    return df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--paper-subset", action="store_true",
                   help="process the authors' cleaned subset instead of raw.")
    a = p.parse_args()
    try:
        if a.paper_subset:
            run_paper_subset()
        else:
            run()
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)
