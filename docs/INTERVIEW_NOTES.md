# Interview talking points (Stock Analyzer ML v2)

Use these as short bullets when you demo the project.

## Product framing
- The app provides **two horizons**:
  - **Short-term (10 trading days)**: tighter threshold, "high-conviction" signal.
  - **Swing (60 trading days)**: primary signal for multi-week trend.
- Output is **probabilistic** (not a single yes/no) so the consumer can decide risk tolerance.

## Engineering decisions
- **FastAPI**: clear contract, automatic docs (`/docs`), typed Pydantic response.
- **Model loading** uses `lru_cache` to avoid re-loading pickles per request.
- **Explainability**: returns top feature contributions (good for trust + debugging).
- **Separation of concerns**:
  - `stock_ml/` contains ML/feature code
  - `app/` is the API layer
  - `frontend/` is a static UI that can be deployed independently

## Deployable architecture (what makes it "real")
- Frontend is a static build (CDN friendly) and calls the API via HTTPS.
- Backend exposes health checks (`/health`) and has CORS controlled via env var.
- Dockerfile enables containerized deployment.

## Future improvements (nice to mention)
- Add **caching** for ticker history to reduce external calls (TTL cache).
- Add **rate limiting** to protect API keys and avoid abuse.
- Add **batch endpoint** `/analyze` that accepts multiple tickers.
- Add **model monitoring**: track prediction drift and data quality checks.
