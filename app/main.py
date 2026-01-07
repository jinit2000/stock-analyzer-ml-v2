# app/main.py

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from stock_ml.config import (
    FEATURE_COLUMNS,
    SHORT_HORIZON_DAYS,
    SHORT_RETURN_THRESHOLD,
    SWING_HORIZON_DAYS,
    SWING_RETURN_THRESHOLD,
    MODELS_DIR,
)
from stock_ml.features import (
    fetch_history,
    add_technical_features,
    add_fundamental_features,
)
from stock_ml.explain import explain_instance, summarize_prediction, prob_to_label
from app.schemas import AnalyzeResponse, HorizonPrediction, Reason

app = FastAPI(
    title="Stock Analyzer v2.0",
    description=(
        "ML-powered stock analyzer with short-term and swing predictions.\n\n"
        "Short-term = 10 trading days, 1.5% target (high-conviction only).\n"
        "Swing = 60 trading days, 5% target (primary signal)."
    ),
    version="2.0.0",
)


# ---------- MODEL LOADING HELPERS ----------


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

    latest_row = df.iloc[-1]
    return latest_row


# ---------- ROUTES ----------
@app.get("/", tags=["meta"])
def root():
    return {
        "message": "Welcome to Stock Analyzer v2.0 API. See /docs for interactive documentation."
    }


@app.get("/health", tags=["meta"])
def health_check():
    return {"status": "ok"}


@app.get("/analyze/{ticker}", response_model=AnalyzeResponse, tags=["analysis"])
def analyze_ticker(ticker: str):
    """
    Analyze a given stock ticker and return short-term and swing predictions.

    - Short-term: horizon = SHORT_HORIZON_DAYS, target = SHORT_RETURN_THRESHOLD
    - Swing: horizon = SWING_HORIZON_DAYS, target = SWING_RETURN_THRESHOLD
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
    if isinstance(as_of_date, pd.Timestamp):
        as_of_date_str = as_of_date.date().isoformat()
    else:
        as_of_date_str = str(as_of_date)

    # 2) Load models
    try:
        clf_short, scaler_short = load_model_pair("short")
    except RuntimeError:
        # If short model isn't available yet, we treat it as None in response
        clf_short, scaler_short = None, None

    clf_swing, scaler_swing = load_model_pair("swing")

    # 3) Predict probabilities
    short_prob: Optional[float] = None
    short_reasons = []
    if clf_short is not None and scaler_short is not None:
        short_prob = float(
            clf_short.predict_proba(
                scaler_short.transform(x_raw.reshape(1, -1))
            )[0, 1]
        )
        short_reasons = explain_instance(clf_short, scaler_short, x_raw, top_k=8)
    else:
        short_prob = None
        short_reasons = []

    swing_prob = float(
        clf_swing.predict_proba(
            scaler_swing.transform(x_raw.reshape(1, -1))
        )[0, 1]
    )
    swing_reasons = explain_instance(clf_swing, scaler_swing, x_raw, top_k=8)

    # 4) Convert reasons from raw dicts to Pydantic models
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

    # 5) Use summarize_prediction / prob_to_label to get labels
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
        # Only swing available
        short_label = None
        swing_label = prob_to_label(swing_prob, horizon="swing")

    # 6) Build HorizonPrediction objects
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

    # 7) Final response
    return AnalyzeResponse(
        ticker=ticker,
        as_of_date=as_of_date_str,
        short_term=short_pred,
        swing=swing_pred,
    )
