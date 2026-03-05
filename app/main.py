# app/main.py

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import AnalyzeResponse, HorizonPrediction, Reason
from stock_ml.config import (
    FEATURE_COLUMNS,
    MODELS_DIR,
    SHORT_HORIZON_DAYS,
    SHORT_RETURN_THRESHOLD,
    SWING_HORIZON_DAYS,
    SWING_RETURN_THRESHOLD,
)
from stock_ml.explain import explain_instance, prob_to_label, summarize_prediction
from stock_ml.features import add_fundamental_features, add_technical_features, fetch_history

# -----------------------------------------------------------------------------
# APP SETUP
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Stock Analyzer v2.0",
    description=(
        "ML-powered stock analyzer with short-term and swing predictions.\n\n"
        f"Short-term = {SHORT_HORIZON_DAYS} trading days, {SHORT_RETURN_THRESHOLD:.2%} target (high-conviction only).\n"
        f"Swing = {SWING_HORIZON_DAYS} trading days, {SWING_RETURN_THRESHOLD:.2%} target (primary signal)."
    ),
    version="2.0.0",
)

# -----------------------------------------------------------------------------
# CORS (for web UI)
# ALLOWED_ORIGINS="http://localhost:5173,https://your-ui.vercel.app"
# -----------------------------------------------------------------------------

allowed_origins_env = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
)
ALLOWED_ORIGINS = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
BACKTEST_PATH = BASE_DIR / "models" / "backtest_demo.json"

# -----------------------------------------------------------------------------
# QUOTE (LIVE PRICE) — cached, small TTL
# -----------------------------------------------------------------------------

_QUOTE_CACHE: dict[str, tuple[float, dict]] = {}
_QUOTE_TTL_SECONDS = 20


def _get_cached_quote(ticker: str) -> Optional[dict]:
    now = time.time()
    item = _QUOTE_CACHE.get(ticker)
    if item and (now - item[0] < _QUOTE_TTL_SECONDS):
        return item[1]
    return None


def _set_cached_quote(ticker: str, payload: dict) -> None:
    _QUOTE_CACHE[ticker] = (time.time(), payload)


@app.get("/quote/{ticker}", tags=["market"])
def quote(ticker: str):
    """
    Get near-real-time quote (may be delayed depending on source/market).
    Returns: price, prev_close, change, change_pct, as_of timestamp.
    """
    t = ticker.strip().upper()
    if not t:
        raise HTTPException(status_code=400, detail="Ticker required")

    cached = _get_cached_quote(t)
    if cached:
        return cached

    try:
        tk = yf.Ticker(t)

        info = getattr(tk, "fast_info", {}) or {}
        last_price = info.get("last_price") or info.get("lastPrice")
        prev_close = info.get("previous_close") or info.get("previousClose")

        # Fallback if fast_info doesn't provide enough
        if last_price is None or prev_close is None:
            hist = tk.history(period="5d", interval="1d")
            if hist is None or hist.empty:
                raise HTTPException(status_code=404, detail=f"No price data for {t}")
            last_price = float(hist["Close"].iloc[-1])
            prev_close = (
                float(hist["Close"].iloc[-2]) if len(hist) >= 2 else float(hist["Close"].iloc[-1])
            )

        last_price = float(last_price)
        prev_close = float(prev_close)

        change = last_price - prev_close
        change_pct = (change / prev_close) if prev_close else 0.0

        payload = {
            "ticker": t,
            "price": last_price,
            "prev_close": prev_close,
            "change": change,
            "change_pct": change_pct,
            "as_of": datetime.now(timezone.utc).isoformat(),
            "source": "yfinance",
            "note": "Price may be delayed depending on market/data source.",
        }

        _set_cached_quote(t, payload)
        return payload

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Quote fetch failed for {t}: {e}")


# -----------------------------------------------------------------------------
# METRICS + BACKTEST (static JSON served to UI)
# -----------------------------------------------------------------------------

@app.get("/metrics", tags=["meta"])
def get_metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="metrics.json not found. Run scripts/train_models.py to generate it.",
        )
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.get("/backtest/demo", tags=["meta"])
def get_backtest_demo():
    if not BACKTEST_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="backtest_demo.json not found. Run scripts/backtest_demo.py to generate it.",
        )
    return json.loads(BACKTEST_PATH.read_text(encoding="utf-8"))


# -----------------------------------------------------------------------------
# MODEL LOADING HELPERS
# -----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_model_pair(prefix: str):
    """
    Load logistic regression model and scaler for given prefix ("short" or "swing").
    Uses LRU cache so they are loaded only once per process.
    """
    model_path = MODELS_DIR / f"{prefix}_logreg.pkl"
    scaler_path = MODELS_DIR / f"{prefix}_scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        raise RuntimeError(
            f"Model or scaler not found for prefix '{prefix}'. "
            f"Expected: {model_path} and {scaler_path}. "
            f"Run scripts/train_models.py to create them."
        )

    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return clf, scaler


def build_latest_features_row(ticker: str) -> pd.Series:
    """
    Fetch latest OHLCV data for ticker and compute feature row for the most recent day.
    """
    df = fetch_history(ticker)
    df = add_technical_features(df)
    df = add_fundamental_features(df, ticker)
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError("No rows with complete features for this ticker.")

    return df.iloc[-1]


# -----------------------------------------------------------------------------
# ROUTES
# -----------------------------------------------------------------------------

@app.get("/", tags=["meta"])
def root():
    return {"message": "Welcome to Stock Analyzer v2.0 API. See /docs for documentation."}


@app.get("/health", tags=["meta"])
def health_check():
    return {"status": "ok"}


@app.get("/analyze/{ticker}", response_model=AnalyzeResponse, tags=["analysis"])
def analyze_ticker(ticker: str):
    """
    Analyze a stock ticker and return short-term and swing predictions.
    """
    ticker = ticker.upper().strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker must not be empty.")

    # 1) Build latest feature row
    try:
        latest = build_latest_features_row(ticker)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not fetch or compute features for ticker '{ticker}': {e}",
        )

    # Ensure we have all required features
    missing = [f for f in FEATURE_COLUMNS if f not in latest.index]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing required features for ticker '{ticker}': {missing}",
        )

    x_raw = latest[FEATURE_COLUMNS].values.astype(float)

    # Latest date as ISO string
    as_of_date = latest.name
    as_of_date_str = (
        as_of_date.date().isoformat() if isinstance(as_of_date, pd.Timestamp) else str(as_of_date)
    )

    # 2) Load models
    try:
        clf_short, scaler_short = load_model_pair("short")
    except RuntimeError:
        clf_short, scaler_short = None, None

    clf_swing, scaler_swing = load_model_pair("swing")

    # 3) Predict probabilities
    short_prob: Optional[float] = None
    short_reasons = []
    if clf_short is not None and scaler_short is not None:
        short_prob = float(
            clf_short.predict_proba(scaler_short.transform(x_raw.reshape(1, -1)))[0, 1]
        )
        short_reasons = explain_instance(clf_short, scaler_short, x_raw, top_k=8)

    swing_prob = float(
        clf_swing.predict_proba(scaler_swing.transform(x_raw.reshape(1, -1)))[0, 1]
    )
    swing_reasons = explain_instance(clf_swing, scaler_swing, x_raw, top_k=8)

    # 4) Convert reasons to Pydantic models
    short_reasons_models = [
        Reason(
            feature=r["feature"],
            contribution=r["contribution"],
            direction=r["direction"],
            text=r["text"],
        )
        for r in short_reasons
    ]

    swing_reasons_models = [
        Reason(
            feature=r["feature"],
            contribution=r["contribution"],
            direction=r["direction"],
            text=r["text"],
        )
        for r in swing_reasons
    ]

    # 5) Labels
    if clf_short is not None and short_prob is not None:
        summary = summarize_prediction(
            p_short=short_prob,
            p_swing=swing_prob,
            reasons_short=short_reasons,
            reasons_swing=swing_reasons,
        )
        short_label = summary["short_term"]["label"]
        swing_label = summary["swing"]["label"]
    else:
        short_label = None
        swing_label = prob_to_label(swing_prob, horizon="swing")

    # 6) Build response objects
    short_pred: Optional[HorizonPrediction] = None
    if short_prob is not None:
        short_pred = HorizonPrediction(
            horizon_days=SHORT_HORIZON_DAYS,
            target_return=SHORT_RETURN_THRESHOLD,
            probability=short_prob,
            label=short_label,
            reasons=short_reasons_models,
        )

    swing_pred = HorizonPrediction(
        horizon_days=SWING_HORIZON_DAYS,
        target_return=SWING_RETURN_THRESHOLD,
        probability=swing_prob,
        label=swing_label,
        reasons=swing_reasons_models,
    )

    return AnalyzeResponse(
        ticker=ticker,
        as_of_date=as_of_date_str,
        short_term=short_pred,
        swing=swing_pred,
    )