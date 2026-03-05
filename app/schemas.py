# app/schemas.py

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class Reason(BaseModel):
    """
    One explanation item for the model's decision.
    Mirrors the dicts returned by stock_ml.explain.explain_instance().
    """
    feature: str          # name of the feature (e.g. "rsi_14")
    contribution: float   # numeric contribution (e.g. SHAP value / log-odds contrib)
    direction: str        # "positive"/"negative" or "up"/"down"
    text: str             # human-readable explanation string


class HorizonPrediction(BaseModel):
    """
    Prediction for a particular holding horizon (short-term or swing).
    """
    horizon_days: int         # e.g. 10 or 60
    target_return: float      # e.g. 0.015 or 0.05
    probability: float        # model P(BUY) for this horizon
    label: str                # e.g. "BUY", "NO_BUY", "AVOID"
    reasons: List[Reason]     # top feature contributions


class AnalyzeResponse(BaseModel):
    """
    Full response for /analyze/{ticker}.
    """
    ticker: str
    as_of_date: str                  # ISO date (last data point used)
    short_term: Optional[HorizonPrediction]
    swing: HorizonPrediction
