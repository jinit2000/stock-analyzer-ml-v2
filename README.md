# 📈 Stock Analyzer ML --- v2

Machine-learning powered stock analysis API with FastAPI, Docker, CI/CD,
and explainable predictions.

> Evolution of my earlier GUI tool (`stock-analyzer`).\
> v2 introduces real ML models, a REST API, back-testing, DevOps
> tooling, and production-ready packaging.

## 🚀 Overview

This project predicts whether a stock is likely to gain value over:

-   **Short-term** (10--20 trading days)
-   **Swing trading** (40--60 trading days)

It combines:

-   Historical stock data\
-   Technical indicators (RSI, SMA, MACD, volatility, etc.)\
-   Fundamental signals (P/E, EPS, ROE)\
-   Binary machine-learning classification\
-   Explainable output (reasons behind predictions)

## 🏗 Project Structure

    stock-analyzer-ml-v2/
     ├── app/
     ├── scripts/
     ├── stock_ml/
     ├── models/
     ├── data/
     ├── tests/
     ├── Dockerfile
     ├── Jenkinsfile
     ├── requirements.txt
     └── README.md

## 📡 API Example

GET `/analyze/AAPL` returns prediction probabilities and explanation.

## 🧪 Backtesting

Includes scripts for dataset building, model training and evaluation.

## 🐳 Docker Support

Application runs easily inside Docker.

## 🤖 CI/CD

Jenkins pipeline builds, tests and publishes Docker images
automatically.

## ⚠ Disclaimer

Educational project --- not financial advice.
