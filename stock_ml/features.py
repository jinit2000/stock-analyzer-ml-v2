# stock_ml/features.py

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import ta  # Technical Analysis library, built on pandas & numpy

from .config import DATA_DIR


# ---------------------------------------------------------------------
# Price history download & caching
# ---------------------------------------------------------------------
def fetch_history(
    ticker: str,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download historical OHLCV data for a ticker and cache to disk as Parquet.

    Returns a DataFrame indexed by Date with columns:
    ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    """
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    cache_file = DATA_DIR / f"{ticker}.parquet"

    def _download_fresh() -> pd.DataFrame:
        df_dl = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            group_by="ticker",  # make MultiIndex more predictable
            auto_adjust=False,
        )
        if df_dl.empty:
            raise ValueError(f"No data downloaded for ticker {ticker}")

        # If yfinance gives MultiIndex, flatten it
        if isinstance(df_dl.columns, pd.MultiIndex):
            level0 = df_dl.columns.get_level_values(0)
            level1 = df_dl.columns.get_level_values(-1)
            standard = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

            # Case 1: OHLCV are in level 0
            if standard.issubset(set(level0)):
                df_dl.columns = level0
            # Case 2: OHLCV are in level 1 (common: level 0 is ticker name)
            elif standard.issubset(set(level1)):
                df_dl.columns = level1
            else:
                # Fallback: join levels with underscore
                df_dl.columns = [
                    "_".join(str(x) for x in tup if str(x) != "")
                    for tup in df_dl.columns.to_list()
                ]
        return df_dl

    # Step 1: load from cache or download
    if use_cache and cache_file.exists():
        df = pd.read_parquet(cache_file)
    else:
        df = _download_fresh()
        df.to_parquet(cache_file)

    # Step 2: if parquet kept MultiIndex, flatten again
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        level1 = df.columns.get_level_values(-1)
        standard = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

        if standard.issubset(set(level0)):
            df.columns = level0
        elif standard.issubset(set(level1)):
            df.columns = level1
        else:
            df.columns = [
                "_".join(str(x) for x in tup if str(x) != "")
                for tup in df.columns.to_list()
            ]

    # Step 3: if expected columns are missing (e.g., old bad cache), re-download fresh
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(set(df.columns)):
        # Try once more by ignoring cache and overwriting
        df = _download_fresh()

        if isinstance(df.columns, pd.MultiIndex):
            level0 = df.columns.get_level_values(0)
            level1 = df.columns.get_level_values(-1)
            standard = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

            if standard.issubset(set(level0)):
                df.columns = level0
            elif standard.issubset(set(level1)):
                df.columns = level1
            else:
                df.columns = [
                    "_".join(str(x) for x in tup if str(x) != "")
                    for tup in df.columns.to_list()
                ]

        if not required_cols.issubset(set(df.columns)):
            # Give a clear error if something is really off
            raise ValueError(
                f"Missing expected columns {required_cols} in price history for {ticker}. "
                f"Available columns after re-download: {list(df.columns)}"
            )

        # Overwrite the bad cache with the good one
        df.to_parquet(cache_file)

    # Ensure DateTimeIndex and sorted
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)

    return df


# ---------------------------------------------------------------------
# Technical features
# ---------------------------------------------------------------------
def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators as feature columns.

    Uses 'ta' library: RSI, MACD, ATR, etc.
    """
    df = df.copy()

    # 🔧 Ensure key columns are 1D Series, not DataFrames
    def _ensure_series(col: str) -> pd.Series:
        if col not in df.columns:
            raise KeyError(
                f"Expected column '{col}' not found in DataFrame. "
                f"Got columns: {list(df.columns)}"
            )
        c = df[col]
        if isinstance(c, pd.DataFrame):
            # If multiple columns with same name, take the first
            return c.iloc[:, 0]
        return c

    close = _ensure_series("Close")
    high = _ensure_series("High")
    low = _ensure_series("Low")
    volume = _ensure_series("Volume")

    # Overwrite with cleaned Series (keeps things consistent downstream)
    df["Close"] = close
    df["High"] = high
    df["Low"] = low
    df["Volume"] = volume

    # Basic returns
    df["ret_1"] = close.pct_change()
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)
    df["ret_20"] = close.pct_change(20)

    # Simple moving averages
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_200"] = close.rolling(200).mean()

    # Exponential moving average
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()

    # Price position relative to averages
    sma_20 = df["sma_20"]
    sma_50 = df["sma_50"]
    sma_200 = df["sma_200"]

    df["price_over_sma_20"] = close / sma_20
    df["price_over_sma_50"] = close / sma_50
    df["price_over_sma_200"] = close / sma_200

    # Momentum: RSI
    rsi_indicator = ta.momentum.RSIIndicator(close, window=14)
    df["rsi_14"] = rsi_indicator.rsi()

    # Momentum: MACD
    macd_indicator = ta.trend.MACD(
        close, window_slow=26, window_fast=12, window_sign=9
    )
    df["macd"] = macd_indicator.macd()
    df["macd_signal"] = macd_indicator.macd_signal()

    # Volatility: rolling std of returns
    df["vol_20"] = df["ret_1"].rolling(20).std()

    # Volatility: Average True Range (ATR)
    atr_indicator = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    )
    df["atr_14"] = atr_indicator.average_true_range()

    # Volume features
    df["vol_ma_20"] = volume.rolling(20).mean()
    df["vol_ratio_20"] = volume / df["vol_ma_20"]

    # Distance to recent high/low (support/resistance proxy)
    df["roll_min_20"] = low.rolling(20).min()
    df["roll_max_20"] = high.rolling(20).max()
    df["dist_to_20_low"] = (close - df["roll_min_20"]) / df["roll_min_20"]
    df["dist_to_20_high"] = (df["roll_max_20"] - close) / df["roll_max_20"]

    return df


# ---------------------------------------------------------------------
# Fundamental features
# ---------------------------------------------------------------------
def add_fundamental_features(
    df: pd.DataFrame,
    ticker: str,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Attach basic fundamental features from yfinance.info to each row.

    These are static for a given download (do not change daily here).
    """
    df = df.copy()
    stock = yf.Ticker(ticker)
    info = stock.info or {}

    pe = info.get("trailingPE", np.nan)
    eps = info.get("trailingEps", np.nan)
    roe = info.get("returnOnEquity", np.nan)

    if overwrite or "fund_pe" not in df.columns:
        df["fund_pe"] = pe
        df["fund_eps"] = eps
        df["fund_roe"] = roe

    return df
