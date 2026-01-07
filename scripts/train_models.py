# scripts/train_models.py

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_ml.config import DATA_DIR
from stock_ml.modeling import (
    cross_validate_logistic,
    train_final_logistic,
)


def main():
    dataset_path = DATA_DIR / "dataset.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            f"Run scripts/build_dataset.py first."
        )

    df = pd.read_parquet(dataset_path)

    # Ensure DateTimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    print("[train_models] Cross-validating short-term (label_st) model...")
    cv_st = cross_validate_logistic(df, label_col="label_st")

    print("[train_models] Cross-validating swing (label_sw) model...")
    cv_sw = cross_validate_logistic(df, label_col="label_sw")

    print("[train_models] Training final short-term model on full data...")
    train_final_logistic(df, label_col="label_st", model_name_prefix="short")

    print("[train_models] Training final swing model on full data...")
    train_final_logistic(df, label_col="label_sw", model_name_prefix="swing")

    print("[train_models] Done.")
    print("[train_models] Short-term CV average:", cv_st)
    print("[train_models] Swing CV average:", cv_sw)


if __name__ == "__main__":
    main()
