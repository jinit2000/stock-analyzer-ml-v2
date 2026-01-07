# stock_ml/labeling.py

from __future__ import annotations

import pandas as pd

from .config import (
    SHORT_HORIZON_DAYS,
    SHORT_RETURN_THRESHOLD,
    SWING_HORIZON_DAYS,
    SWING_RETURN_THRESHOLD,
)


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add short-term and swing binary labels based on future returns.

    label_st = 1 if future N-day return >= SHORT_RETURN_THRESHOLD, else 0
    label_sw = 1 if future M-day return >= SWING_RETURN_THRESHOLD, else 0
    """
    df = df.copy()

    # Ensure DateTimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Short-term horizon
    df["future_close_st"] = df["Close"].shift(-SHORT_HORIZON_DAYS)
    df["ret_st"] = (df["future_close_st"] - df["Close"]) / df["Close"]
    df["label_st"] = (df["ret_st"] >= SHORT_RETURN_THRESHOLD).astype(int)

    # Swing horizon
    df["future_close_sw"] = df["Close"].shift(-SWING_HORIZON_DAYS)
    df["ret_sw"] = (df["future_close_sw"] - df["Close"]) / df["Close"]
    df["label_sw"] = (df["ret_sw"] >= SWING_RETURN_THRESHOLD).astype(int)

    # Drop rows where we don't have future data
    df.dropna(subset=["future_close_st", "future_close_sw"], inplace=True)

    return df
