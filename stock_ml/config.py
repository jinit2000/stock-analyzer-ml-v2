# stock_ml/config.py

from pathlib import Path

# Root directory = project root (two levels up from this file)
ROOT_DIR = Path(__file__).resolve().parents[1]

# Data and models directories
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

DATA_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Training universe (you can extend this list)
TRAIN_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "JNJ", "NFLX"
]

# Historical data range
DATE_START = "2010-01-01"
DATE_END = None  # None = up to latest

# Label definitions (binary classification)
# Short-term: will price go up by at least +1.5% within 10 trading days?
SHORT_HORIZON_DAYS = 10
SHORT_RETURN_THRESHOLD = 0.015

# Swing: will price go up by at least +5% within 60 trading days?
SWING_HORIZON_DAYS = 60
SWING_RETURN_THRESHOLD = 0.05

# Feature columns (we keep them in one place so modeling + explain can share)
FEATURE_COLUMNS = [
    "ret_1", "ret_5", "ret_10", "ret_20",
    "price_over_sma_20", "price_over_sma_50", "price_over_sma_200",
    "rsi_14", "macd", "macd_signal",
    "vol_20", "atr_14",
    "vol_ratio_20",
    "dist_to_20_low", "dist_to_20_high",
    # fundamentals (filled if available)
    "fund_pe", "fund_eps", "fund_roe"
]

# Thresholds for turning probabilities into recommendations
SHORT_PROB_THRESHOLDS = {
    "strong_buy": 0.8,
    "buy": 0.60,
    "hold": 0.45  # below this = sell/avoid
}

SWING_PROB_THRESHOLDS = {
    "strong_buy": 0.7,
    "buy": 0.55,
    "hold": 0.45
}
