# stock_ml/dataset.py

from __future__ import annotations

from typing import List

import pandas as pd

from .config import DATE_START, DATE_END
from .features import fetch_history, add_technical_features, add_fundamental_features
from .labeling import add_labels


def build_dataset(
    tickers: List[str],
    start: str = DATE_START,
    end: str | None = DATE_END,
) -> pd.DataFrame:
    """
    Build a full multi-ticker dataset with features and labels.

    Returns
    -------
    df : pd.DataFrame
        Columns include technical features, fundamental features,
        labels (label_st, label_sw), and 'ticker'.
        Index is DateTimeIndex.
    """
    frames = []
    for ticker in tickers:
        print(f"[build_dataset] Processing {ticker}...")
        df = fetch_history(ticker, start=start, end=end)
        df = add_technical_features(df)
        df = add_fundamental_features(df, ticker)
        df = add_labels(df)
        df["ticker"] = ticker
        frames.append(df)

    full = pd.concat(frames, axis=0)
    # Ensure DateTimeIndex & sorted (multi-ticker)
    if not isinstance(full.index, pd.DatetimeIndex):
        full.index = pd.to_datetime(full.index)
    full.sort_index(inplace=True)

    # Drop rows with any missing features (safe, simple option)
    full.dropna(inplace=True)

    return full
