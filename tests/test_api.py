# tests/test_api.py

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_analyze_ticker_swing_basic():
    """
    Basic integration test for /analyze/{ticker}.

    Assumes:
      - models have been trained and saved into models/ (short_* and swing_*),
      - internet access is available to fetch latest market data from yfinance.
    """
    resp = client.get("/analyze/AAPL")
    assert resp.status_code == 200

    data = resp.json()
    assert data["ticker"] == "AAPL"

    # Swing prediction should always be present
    assert "swing" in data
    swing = data["swing"]

    assert swing["horizon_days"] == 60
    assert swing["target_return"] == 0.05
    assert 0.0 <= swing["probability"] <= 1.0
    assert swing["label"] in ["STRONG BUY", "BUY", "HOLD", "SELL"]

    # Reasons should be a non-empty list
    assert isinstance(swing["reasons"], list)
    assert len(swing["reasons"]) > 0
    reason = swing["reasons"][0]
    assert "feature" in reason
    assert "direction" in reason
    assert reason["direction"] in ["bull", "bear"]
    assert "text" in reason
