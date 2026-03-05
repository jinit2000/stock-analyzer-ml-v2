📈 Stock Analyzer ML v2

A machine-learning based stock analysis tool that provides probability signals for short-term and swing trading horizons.
The project includes a FastAPI backend, a React web interface, and scripts for data preparation, model training, and evaluation.

This project is an evolution of an earlier GUI prototype (stock-analyzer).
Version 2 focuses on a cleaner architecture with an API, a web UI, and explainable model outputs.

🚀 Overview

The system estimates the probability that a stock will reach a target return within a given time horizon.

Two horizons are supported:

Short-term: ~10 trading days (quick momentum signals)

Swing: ~60 trading days (multi-week trend signals)

Predictions are based on a combination of:

Historical market data

Technical indicators (RSI, SMA, MACD, volatility, etc.)

Basic fundamental indicators (P/E, EPS, ROE)

Logistic regression classification models

The API returns both the prediction probability and feature contributions so the output is explainable.

🏗 Project Structure
stock-analyzer-ml-v2/
 ├── app/            # FastAPI application
 ├── stock_ml/       # Feature engineering and model utilities
 ├── scripts/        # Dataset building and training scripts
 ├── models/         # Saved models and metrics
 ├── data/           # Example datasets
 ├── frontend/       # React + Vite web UI
 ├── tests/          # Basic tests
 ├── Dockerfile
 ├── requirements.txt
 └── README.md
📡 API Example

Example request:

GET /analyze/AAPL

Example response (simplified):

{
  "short_term": {
    "probability": 0.48,
    "label": "HOLD"
  },
  "swing": {
    "probability": 0.51,
    "label": "HOLD"
  }
}
🖥 Web UI

The repository includes a small React frontend located in frontend/.

The UI allows you to:

Enter a ticker symbol

View prediction probabilities

See the most important features affecting the prediction

Display the latest price data

▶ Running Locally
1. Start the API
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export ALLOWED_ORIGINS="http://localhost:5173"

uvicorn app.main:app --reload --port 8000

API documentation will be available at:

http://127.0.0.1:8000/docs
2. Start the Web UI
cd frontend
npm install
npm run dev

Open the UI in your browser:

http://localhost:5173
🔎 Model Horizons (UI Explanation)

The UI shows two prediction horizons:

Short-term

Horizon: ~10 trading days

Focus: short momentum or mean-reversion signals

Swing

Horizon: ~60 trading days

Focus: medium-term trends

The interface also displays top contributing features so the prediction is interpretable.

🧪 Training and Backtesting

The repository includes scripts for:

Building datasets

Training models

Cross-validation

Exporting metrics

Example:

python scripts/build_dataset.py
python scripts/train_models.py
🐳 Docker

You can run the API using Docker.

docker build -t stock-analyzer-ml .
docker run -p 8000:8000 stock-analyzer-ml
🌍 Deployment

A simple deployment setup:

Backend

Render (FastAPI service)

Frontend

Vercel (static React app)

Example environment variables:

Frontend (Vercel):

VITE_API_BASE_URL=https://your-api.onrender.com

Backend (Render):

ALLOWED_ORIGINS=https://your-ui.vercel.app
⚠ Disclaimer

This project is for educational purposes only and should not be used as financial advice.