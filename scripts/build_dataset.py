# scripts/build_dataset.py

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_ml.config import TRAIN_TICKERS, DATA_DIR
from stock_ml.dataset import build_dataset


def main():
    print("[build_dataset] Building dataset...")
    df = build_dataset(TRAIN_TICKERS)
    out_path = DATA_DIR / "dataset.parquet"
    df.to_parquet(out_path)
    print(f"[build_dataset] Saved dataset to {out_path}")
    print(f"[build_dataset] Shape: {df.shape}")


if __name__ == "__main__":
    main()
