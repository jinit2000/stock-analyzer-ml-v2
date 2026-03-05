# scripts/analyze_ticker.py

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

from stock_ml.config import (
    FEATURE_COLUMNS,
    MODELS_DIR,
)
from stock_ml.features import fetch_history, add_technical_features, add_fundamental_features
from stock_ml.explain import explain_instance, summarize_prediction
from stock_ml.labeling import add_labels  # optional if you want to see future returns


def load_model_pair(prefix: str):
    """Load logistic model and scaler for given prefix ('short' or 'swing')."""
    model_path = MODELS_DIR / f"{prefix}_logreg.pkl"
    scaler_path = MODELS_DIR / f"{prefix}_scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"Model or scaler not found for prefix '{prefix}'. "
            f"Run scripts/train_models.py first."
        )

    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return clf, scaler


def build_feature_row_for_latest(ticker: str) -> pd.Series:
    """
    Fetch latest data for ticker and compute feature row for the most recent day.
    """
    df = fetch_history(ticker)
    df = add_technical_features(df)
    df = add_fundamental_features(df, ticker)
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError("No rows with complete features for this ticker.")

    latest_row = df.iloc[-1]
    return latest_row


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.analyze_ticker TICKER")
        sys.exit(1)

    ticker = sys.argv[1].upper().strip()
    print(f"[analyze_ticker] Analyzing {ticker}...")

    # Build feature row
    latest = build_feature_row_for_latest(ticker)
    x_raw = latest[FEATURE_COLUMNS].values.astype(float)

    # Load models
    clf_short, scaler_short = load_model_pair("short")
    clf_swing, scaler_swing = load_model_pair("swing")

    # Predict probabilities
    p_short = float(clf_short.predict_proba(scaler_short.transform(x_raw.reshape(1, -1)))[0, 1])
    p_swing = float(clf_swing.predict_proba(scaler_swing.transform(x_raw.reshape(1, -1)))[0, 1])

    # Explain
    reasons_short = explain_instance(clf_short, scaler_short, x_raw, top_k=8)
    reasons_swing = explain_instance(clf_swing, scaler_swing, x_raw, top_k=8)

    summary = summarize_prediction(p_short, p_swing, reasons_short, reasons_swing)

    print(f"\n=== Stock Analyzer v2.0 – ML Prediction for {ticker} ===")
    print(f"Short-term (10 days, +3% target):")
    print(f"  Probability of success: {summary['short_term']['probability']:.3f}")
    print(f"  Recommendation: {summary['short_term']['label']}")
    print("  Key reasons:")
    for r in summary["short_term"]["reasons"]:
        print(f"   - ({r['direction']}) {r['text']}  [contrib={r['contribution']:.3f}]")

    print(f"\nSwing (40 days, +8% target):")
    print(f"  Probability of success: {summary['swing']['probability']:.3f}")
    print(f"  Recommendation: {summary['swing']['label']}")
    print("  Key reasons:")
    for r in summary["swing"]["reasons"]:
        print(f"   - ({r['direction']}) {r['text']}  [contrib={r['contribution']:.3f}]")

    print("\n⚠️  This is a research tool only, not financial advice.")


if __name__ == "__main__":
    main()
