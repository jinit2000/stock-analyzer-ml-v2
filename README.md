# Stock Analyzer ML v2

A full-stack machine learning application that estimates the probability a stock will hit a target return within a given time horizon.

Built with a **FastAPI backend**, **React + Vite frontend**, **Docker**, and a **Jenkins CI/CD pipeline**.

> Evolution of [stock-analyzer](https://github.com/jinit2000/stock-analyzer) - rebuilt with an API, web UI, explainable model outputs, and production-ready infrastructure.

---

## What It Does

Enter a ticker symbol and get two ML-powered predictions:

| Horizon | Window | Focus |
|---|---|---|
| Short-term | ~10 trading days | Momentum and mean-reversion signals |
| Swing | ~60 trading days | Medium-term trend signals |

Each prediction returns a **probability score**, a **BUY / HOLD / SELL label**, and the **top features driving the decision** - so the output is fully explainable, not a black box.

---

## Tech Stack

| Layer | Tech |
|---|---|
| Backend | Python, FastAPI, scikit-learn |
| Frontend | React, TypeScript, Vite |
| ML | Logistic Regression, pandas, ta |
| Data | yfinance (Yahoo Finance) |
| DevOps | Docker, Jenkins CI/CD |
| Deployment | Render (API) + Vercel (UI) |

---

## Features

- Live stock data via Yahoo Finance
- Technical indicators: RSI, MACD, SMA 50/200, volatility, Bollinger Bands
- Fundamental indicators: P/E Ratio, EPS, Return on Equity
- Two prediction horizons: short-term (10d) and swing (60d)
- Explainable output - top contributing features shown per prediction
- REST API with auto-generated docs at `/docs`
- Dockerized backend
- Jenkins pipeline for automated build and test

---

## Project Structure

```
stock-analyzer-ml-v2/
├── app/          # FastAPI application and routes
├── stock_ml/     # Feature engineering and model utilities
├── scripts/      # Dataset building and model training scripts
├── models/       # Saved trained models and metrics
├── data/         # Example datasets
├── frontend/     # React + Vite web UI
├── tests/        # Unit tests
├── Dockerfile
├── Jenkinsfile
└── requirements.txt
```

---

## API Example

```
GET /analyze/AAPL
```

Response:

```json
{
  "short_term": {
    "probability": 0.63,
    "label": "BUY",
    "top_features": ["RSI", "MACD", "SMA_50"]
  },
  "swing": {
    "probability": 0.51,
    "label": "HOLD",
    "top_features": ["P/E", "ROE", "SMA_200"]
  }
}
```

Full API docs available at `http://localhost:8000/docs` after running locally.

---

## Running Locally

**1. Clone the repo**

```bash
git clone https://github.com/jinit2000/stock-analyzer-ml-v2.git
cd stock-analyzer-ml-v2
```

**2. Start the API**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
export ALLOWED_ORIGINS="http://localhost:5173"
uvicorn app.main:app --reload --port 8000
```

API docs available at: `http://127.0.0.1:8000/docs`

**3. Start the frontend**

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Running with Docker

```bash
docker build -t stock-analyzer-ml .
docker run -p 8000:8000 stock-analyzer-ml
```

---

## Training the Models

```bash
python scripts/build_dataset.py
python scripts/train_models.py
```

---

## Deployment

| Service | Platform |
|---|---|
| Backend (FastAPI) | Render |
| Frontend (React) | Vercel |

Environment variables:

```
# Frontend (Vercel)
VITE_API_BASE_URL=https://your-api.onrender.com

# Backend (Render)
ALLOWED_ORIGINS=https://your-ui.vercel.app
```

---

## Planned Improvements

- [ ] Candlestick chart visualization
- [ ] TSX (Canadian) ticker support
- [ ] Sector-relative scoring
- [ ] Upgrade to XGBoost model
- [ ] Portfolio-level analysis (multiple tickers at once)

---

> **Disclaimer:** This project is for educational purposes only. Always do your own research before making investment decisions.
