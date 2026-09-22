"""
One-command pipeline runner for the faculty demo.

Runs, in order:
  1. download_data     (Kaggle dataset -> data/raw)
  2. preprocess (full) (clean + resample -> data/processed/clean_1min.csv)
  3. preprocess (paper subset) -> data/processed/clean_paper_subset.csv
  4. feature_analysis  (Fig. 9 correlation heatmap)
  5. train             (GRU/LSTM/Transformer, paper subset, 6 feat, random)
  6. evaluate          (metrics + Fig. 16/18/19)
  7. baselines         (ARIMA + MLP/KNN/XGBoost)
  8. make_report       (results/RESULTS.md)

Usage (from the src/ directory):
    python run_all.py
    python run_all.py --skip-download    # if data already present
"""
import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
PY = sys.executable


def step(title, args, cont_on_error=False):
    print("\n" + "#" * 72)
    print(f"# {title}")
    print("#" * 72, flush=True)
    r = subprocess.run([PY, "-u"] + args, cwd=str(SRC))
    if r.returncode != 0 and not cont_on_error:
        print(f"[run_all] step failed: {title} (exit {r.returncode})")
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    a = ap.parse_args()

    if not a.skip_download:
        step("1/8 Download dataset", ["download_data.py"], cont_on_error=True)
    step("2/8 Preprocess (full month)", ["preprocess.py"])
    step("3/8 Preprocess (paper subset)", ["preprocess.py", "--paper-subset"])
    step("4/8 Feature correlation (Fig. 9)", ["feature_analysis.py"])
    if not a.skip_train:
        step("5/8 Train GRU/LSTM/Transformer",
             ["train.py", "--models", "gru", "lstm", "transformer",
              "--features", "6", "--source", "paper", "--split", "random"])
    step("6/8 Evaluate + figures", ["evaluate.py"])
    step("7/8 Baselines (ARIMA + ML)", ["baselines.py"])
    step("8/8 Consolidated report", ["make_report.py"])

    print("\n[run_all] DONE. See results/RESULTS.md and results/figures/.")


if __name__ == "__main__":
    main()
