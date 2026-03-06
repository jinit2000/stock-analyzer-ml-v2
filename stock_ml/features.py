# stock_ml/features.py

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
import time
import random

import numpy as np
import pandas as pd
import yfinance as yf
import ta  # Technical Analysis library, built on pandas & numpy

from .config import DATA_DIR


# ---------------------------------------------------------------------
# In-memory caches (to avoid hitting Yahoo too often)
# ---------------------------------------------------------------------
_HISTORY_MEM_CACHE: Dict[Tuple[str, str, Optional[str]], Tuple[float, pd.DataFrame]] = {}
_FUNDAMENTALS_MEM_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}

# Tune these for Render free tier (safe defaults)
HISTORY_MEM_TTL_SECONDS = 120       # 2 minutes
FUNDAMENTALS_MEM_TTL_SECONDS = 600  # 10 minutes

# Disk cache: keep parquet and re-use if not too old
DISK_CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours


def _is_rate_limited_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("too many requests" in msg) or ("rate limit" in msg) or ("429" in msg)


def _sleep_backoff(attempt: int) -> None:
    # Exponential backoff with jitter: 0.8s, 1.6s, 3.2s (+ jitter)
    base = 0.8 * (2 ** (attempt - 1))
    time.sleep(base + random.uniform(0.0, 0.25))


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance sometimes returns MultiIndex columns (ticker + OHLCV).
    This normalizes into standard OHLCV columns.
    """
    if df is None or df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        level1 = df.columns.get_level_values(-1)
        standard = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

        # Case 1: OHLCV in level 0
        if standard.issubset(set(level0)):
            df.columns = level0
        # Case 2: OHLCV in level 1 (common: level 0 is ticker)
        elif standard.issubset(set(level1)):
            df.columns = level1
        else:
            # Fallback: join levels with underscore
            df.columns = [
                "_".join(str(x) for x in tup if str(x) != "")
                for tup in df.columns.to_list()
            ]

    return df


def _ensure_datetime_sorted(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    return df


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

    Key upgrades:
    - In-memory TTL cache (prevents repeated Yahoo calls on Render)
    - Disk cache TTL (parquet reused if still fresh)
    - Retry/backoff when Yahoo rate-limits
    """
    t = ticker.strip().upper()
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    cache_file = DATA_DIR / f"{t}.parquet"

    # ---------
    # 0) In-memory cache
    # ---------
    mem_key = (t, start, end)
    if use_cache:
        item = _HISTORY_MEM_CACHE.get(mem_key)
        if item:
            ts, df_cached = item
            if (time.time() - ts) < HISTORY_MEM_TTL_SECONDS and not df_cached.empty:
                return df_cached.copy()

    # ---------
    # 1) Disk cache (optional + TTL)
    # ---------
    def _disk_cache_is_fresh(path) -> bool:
        try:
            age = time.time() - path.stat().st_mtime
            return age < DISK_CACHE_TTL_SECONDS
        except Exception:
            return False

    if use_cache and cache_file.exists() and _disk_cache_is_fresh(cache_file):
        try:
            df_disk = pd.read_parquet(cache_file)
            df_disk = _flatten_yf_columns(df_disk)
            df_disk = _ensure_datetime_sorted(df_disk)

            required_cols = {"Open", "High", "Low", "Close", "Volume"}
            if required_cols.issubset(set(df_disk.columns)) and not df_disk.empty:
                # store to memory and return
                _HISTORY_MEM_CACHE[mem_key] = (time.time(), df_disk.copy())
                return df_disk
        except Exception:
            # If disk cache is corrupted, fall through to fresh download
            pass

    # ---------
    # 2) Download with retry/backoff
    # ---------
    def _download_fresh() -> pd.DataFrame:
        last_err: Optional[Exception] = None

        for attempt in range(1, 4):  # 3 tries
            try:
                df_dl = yf.download(
                    t,
                    start=start,
                    end=end,
                    progress=False,
                    group_by="ticker",
                    auto_adjust=False,
                    threads=False,  # slightly gentler on Yahoo
                )

                if df_dl is None or df_dl.empty:
                    raise ValueError(f"No data downloaded for ticker {t}")

                df_dl = _flatten_yf_columns(df_dl)
                df_dl = _ensure_datetime_sorted(df_dl)

                required_cols = {"Open", "High", "Low", "Close", "Volume"}
                if not required_cols.issubset(set(df_dl.columns)):
                    raise ValueError(
                        f"Missing expected columns {required_cols} in price history for {t}. "
                        f"Available columns: {list(df_dl.columns)}"
                    )

                return df_dl

            except Exception as e:
                last_err = e
                if _is_rate_limited_error(e):
                    _sleep_backoff(attempt)
                    continue
                break

        # If we got here, retries failed
        if last_err and _is_rate_limited_error(last_err):
            raise ValueError(
                f"Too Many Requests. Rate limited while downloading history for '{t}'. "
                f"Please try again in ~30–60 seconds."
            )
        raise ValueError(f"Failed to download price history for '{t}': {last_err}")

    df = _download_fresh()

    # Save disk cache
    try:
        df.to_parquet(cache_file)
    except Exception:
        # disk write failure shouldn't break the request
        pass

    # Save mem cache
    _HISTORY_MEM_CACHE[mem_key] = (time.time(), df.copy())

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
    df["price_over_sma_20"] = close / df["sma_20"]
    df["price_over_sma_50"] = close / df["sma_50"]
    df["price_over_sma_200"] = close / df["sma_200"]

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

    Upgrade:
    - In-memory TTL cache to reduce rate-limits from repeated .info calls
    - Retry/backoff for rate limit
    """
    df = df.copy()
    t = ticker.strip().upper()

    # 0) Use cached fundamentals if still fresh
    cached = _FUNDAMENTALS_MEM_CACHE.get(t)
    if cached:
        ts, info = cached
        if (time.time() - ts) < FUNDAMENTALS_MEM_TTL_SECONDS:
            pe = info.get("trailingPE", np.nan)
            eps = info.get("trailingEps", np.nan)
            roe = info.get("returnOnEquity", np.nan)

            if overwrite or "fund_pe" not in df.columns:
                df["fund_pe"] = pe
                df["fund_eps"] = eps
                df["fund_roe"] = roe
            return df

    # 1) Fetch with retry/backoff
    last_err: Optional[Exception] = None
    info: Dict[str, Any] = {}
    for attempt in range(1, 4):
        try:
            stock = yf.Ticker(t)
            info = stock.info or {}
            break
        except Exception as e:
            last_err = e
            if _is_rate_limited_error(e):
                _sleep_backoff(attempt)
                continue
            break

    # If rate-limited hard, just proceed with NaNs (don’t fail the whole request)
    if last_err and _is_rate_limited_error(last_err) and not info:
        info = {}

    # Cache whatever we got (even empty) to avoid repeated calls
    _FUNDAMENTALS_MEM_CACHE[t] = (time.time(), info)

    pe = info.get("trailingPE", np.nan)
    eps = info.get("trailingEps", np.nan)
    roe = info.get("returnOnEquity", np.nan)

    if overwrite or "fund_pe" not in df.columns:
        df["fund_pe"] = pe
        df["fund_eps"] = eps
        df["fund_roe"] = roe

    return df