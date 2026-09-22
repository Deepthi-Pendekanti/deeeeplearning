"""
Stage 2: Download the Kaggle dataset used in the paper.

Dataset: "Hydroponic IoT Sensor and Actuator Logs"
DOI: 10.34740/kaggle/ds/8223904  (CC BY 4.0)

This script supports two paths:

1. Kaggle API (preferred, fully automated)
   Requires a Kaggle API token at %USERPROFILE%\\.kaggle\\kaggle.json
   (Kaggle -> Settings -> API -> Create New Token).

2. Manual ZIP fallback
   If you download the dataset ZIP manually from the Kaggle page and drop it
   into data/raw/, this script will detect and extract it. No token needed.

Usage:
    python src/download_data.py
    python src/download_data.py --ref owner/dataset-slug   # if you know the slug
"""
import argparse
import sys
import zipfile
from pathlib import Path

from config import DATA_RAW

DATASET_NUMERIC_ID = "8223904"  # from the DOI 10.34740/kaggle/ds/8223904


def extract_local_zips() -> bool:
    """Extract any .zip files already present in data/raw/. Returns True if any found."""
    zips = list(DATA_RAW.glob("*.zip"))
    if not zips:
        return False
    for z in zips:
        print(f"[manual] Extracting {z.name} ...")
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(DATA_RAW)
    return True


def list_raw():
    files = sorted(p for p in DATA_RAW.rglob("*") if p.is_file())
    print(f"\nContents of {DATA_RAW}:")
    for f in files:
        print(f"  {f.relative_to(DATA_RAW)}  ({f.stat().st_size:,} bytes)")
    return files


def try_kaggle_api(ref: str | None) -> bool:
    """Attempt download via the Kaggle API. Returns True on success."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:  # noqa: BLE001
        print(f"[kaggle] Kaggle package import failed: {e}")
        return False

    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:  # noqa: BLE001
        print(f"[kaggle] Authentication failed (no valid kaggle.json?): {e}")
        return False

    refs_to_try = []
    if ref:
        refs_to_try.append(ref)

    # If no ref given, try to resolve the slug by searching Kaggle datasets.
    if not ref:
        try:
            print("[kaggle] Searching Kaggle for the dataset slug ...")
            for term in ("Hydroponic IoT Sensor and Actuator Logs",
                         "hydroponic sensor actuator logs",
                         "hydroponic pH virtual sensor"):
                results = api.dataset_list(search=term)
                for d in results:
                    print(f"    candidate: {d.ref}")
                    refs_to_try.append(str(d.ref))
        except Exception as e:  # noqa: BLE001
            print(f"[kaggle] Search failed: {e}")

    # De-duplicate while preserving order.
    seen = set()
    refs_to_try = [r for r in refs_to_try if not (r in seen or seen.add(r))]

    for r in refs_to_try:
        try:
            print(f"[kaggle] Trying download: {r}")
            api.dataset_download_files(r, path=str(DATA_RAW), unzip=True, quiet=False)
            print(f"[kaggle] Success: {r}")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[kaggle]   failed for {r}: {e}")

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default=None,
                        help="Kaggle dataset ref 'owner/slug' if known.")
    args = parser.parse_args()

    print("=" * 70)
    print("Stage 2: dataset acquisition")
    print(f"DOI: 10.34740/kaggle/ds/{DATASET_NUMERIC_ID}")
    print("=" * 70)

    # 1) If a ZIP is already sitting in data/raw, just extract it.
    if extract_local_zips():
        print("[ok] Extracted local ZIP(s).")
        list_raw()
        return

    # 2) If CSVs already extracted, we're done.
    existing_csv = list(DATA_RAW.rglob("*.csv"))
    if existing_csv:
        print("[ok] CSV files already present, skipping download.")
        list_raw()
        return

    # 3) Try the Kaggle API.
    if try_kaggle_api(args.ref):
        list_raw()
        return

    # 4) Nothing worked -> print manual instructions.
    print("\n" + "!" * 70)
    print("Could not download automatically.")
    print("Please do ONE of the following:")
    print("  A) Create a Kaggle API token (Kaggle > Settings > API > Create New")
    print("     Token) and place kaggle.json at:")
    print("       %USERPROFILE%\\.kaggle\\kaggle.json")
    print("     then re-run this script.")
    print("  B) Open https://doi.org/10.34740/kaggle/ds/8223904 , click Download,")
    print(f"     and drop the ZIP into: {DATA_RAW}")
    print("     then re-run this script.")
    print("!" * 70)
    sys.exit(1)


if __name__ == "__main__":
    main()
