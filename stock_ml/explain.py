# stock_ml/explain.py

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import (
    FEATURE_COLUMNS,
    SHORT_PROB_THRESHOLDS,
    SWING_PROB_THRESHOLDS,
)


# Human-readable explanation templates for features
EXPLANATION_TEMPLATES: Dict[str, Dict[str, str]] = {
    "rsi_14": {
        "bull": "RSI is relatively low compared to history, often preceding gains.",
        "bear": "RSI is elevated (overbought), often preceding pullbacks.",
    },
    "price_over_sma_20": {
        "bull": "Price is above its 20-day average, indicating short-term upward momentum.",
        "bear": "Price is below its 20-day average, indicating short-term weakness.",
    },
    "price_over_sma_50": {
        "bull": "Price is above its 50-day average, supporting an ongoing uptrend.",
        "bear": "Price is below its 50-day average, suggesting trend weakness.",
    },
    "price_over_sma_200": {
        "bull": "Price is above its 200-day average, consistent with a longer-term uptrend.",
        "bear": "Price is below its 200-day average, consistent with a longer-term downtrend.",
    },
    "dist_to_20_low": {
        "bull": "Price is trading close to recent 20-day lows, improving risk/reward if trend reverses.",
        "bear": "Price is far above recent lows, leaving less margin of safety.",
    },
    "dist_to_20_high": {
        "bull": "Price is not too close to recent 20-day highs, leaving room for upside.",
        "bear": "Price is close to recent 20-day highs where reversals often occur.",
    },
    "vol_20": {
        "bull": "Volatility is moderate, which historically leads to more stable patterns.",
        "bear": "Volatility is high, which historically leads to noisy and less predictable moves.",
    },
    "atr_14": {
        "bull": "Average True Range is not extremely high, indicating manageable day-to-day swings.",
        "bear": "Average True Range is elevated, indicating wide and risky price swings.",
    },
    "vol_ratio_20": {
        "bull": "Volume is above its 20-day average, strengthening recent price action.",
        "bear": "Volume is below its 20-day average, weakening recent price action.",
    },
    "ret_5": {
        "bull": "Recent 5-day returns are positive, indicating short-term upward pressure.",
        "bear": "Recent 5-day returns are negative, indicating short-term selling pressure.",
    },
    "ret_20": {
        "bull": "Recent 20-day returns are positive, indicating sustained momentum.",
        "bear": "Recent 20-day returns are negative, indicating sustained weakness.",
    },
    "macd": {
        "bull": "MACD is favorable, consistent with bullish momentum.",
        "bear": "MACD is weak, consistent with bearish or fading momentum.",
    },
    "fund_pe": {
        "bull": "Valuation (P/E) is not extremely high compared to history.",
        "bear": "Valuation (P/E) is elevated, increasing downside risk if growth slows.",
    },
    "fund_eps": {
        "bull": "Earnings per share are solid, which historically supports price appreciation.",
        "bear": "Earnings per share are weak or negative, which historically limits upside.",
    },
    "fund_roe": {
        "bull": "Return on equity is strong, indicating efficient use of capital.",
        "bear": "Return on equity is weak, indicating less efficient use of capital.",
    },
}


def prob_to_label(
    p: float,
    horizon: str = "short",
) -> str:
    """
    Convert a probability into a discrete recommendation label.
    horizon = "short" or "swing".
    """
    if horizon == "short":
        th = SHORT_PROB_THRESHOLDS
    elif horizon == "swing":
        th = SWING_PROB_THRESHOLDS
    else:
        raise ValueError("horizon must be 'short' or 'swing'")

    if p >= th["strong_buy"]:
        return "STRONG BUY"
    elif p >= th["buy"]:
        return "BUY"
    elif p >= th["hold"]:
        return "HOLD"
    else:
        return "SELL"


def explain_instance(
    clf: LogisticRegression,
    scaler: StandardScaler,
    x_raw: np.ndarray,
    top_k: int = 8,
) -> List[Dict[str, object]]:
    """
    Explain a single instance prediction using logistic regression coefficients.

    Returns a list of dicts with:
      - feature
      - contribution
      - direction ("bull" or "bear")
      - text (human-readable explanation)
    """
    if x_raw.ndim != 1 or x_raw.shape[0] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"x_raw must be 1D array of length {len(FEATURE_COLUMNS)} "
            f"(got shape {x_raw.shape})"
        )

    x_scaled = scaler.transform(x_raw.reshape(1, -1))[0]
    coefs = clf.coef_[0]  # shape: (n_features,)

    # Contribution approximation = coefficient * scaled feature
    contributions = coefs * x_scaled

    # Sort by absolute importance
    idx_sorted = np.argsort(-np.abs(contributions))

    reasons: List[Dict[str, object]] = []
    for idx in idx_sorted[:top_k]:
        fname = FEATURE_COLUMNS[idx]
        contrib = float(contributions[idx])
        template = EXPLANATION_TEMPLATES.get(fname)
        if template is None:
            # Skip features without explicit templates (to keep output clean)
            continue

        direction = "bull" if contrib > 0 else "bear"
        text = template[direction]

        reasons.append(
            {
                "feature": fname,
                "contribution": contrib,
                "direction": direction,
                "text": text,
            }
        )

    return reasons


def summarize_prediction(
    p_short: float,
    p_swing: float,
    reasons_short: List[Dict[str, object]],
    reasons_swing: List[Dict[str, object]],
) -> Dict[str, object]:
    """
    Combine probabilities and explanations for short-term and swing horizons
    into a single structured output.
    """
    label_short = prob_to_label(p_short, horizon="short")
    label_swing = prob_to_label(p_swing, horizon="swing")

    return {
        "short_term": {
            "probability": p_short,
            "label": label_short,
            "reasons": reasons_short,
        },
        "swing": {
            "probability": p_swing,
            "label": label_swing,
            "reasons": reasons_swing,
        },
    }
