import json
from pathlib import Path
import pandas as pd

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())

def run_demo_backtest(prices: pd.Series) -> dict:
    # Simple demo strategy: hold when daily return > 0 (momentum-ish)
    # Replace this later with your model-driven signals if you want.
    rets = prices.pct_change().fillna(0.0)
    signal = (rets.shift(1) > 0).astype(int)  # trade on yesterday's move

    strategy_rets = signal * rets
    equity = (1 + strategy_rets).cumprod() * 100
    buyhold = (1 + rets).cumprod() * 100

    out = pd.DataFrame({
        "date": prices.index.strftime("%Y-%m-%d"),
        "equity": equity.round(4),
        "buyhold": buyhold.round(4),
    })

    return {
        "summary": {
            "strategy_total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
            "buyhold_total_return": float(buyhold.iloc[-1] / buyhold.iloc[0] - 1),
            "strategy_max_drawdown": max_drawdown(equity),
            "buyhold_max_drawdown": max_drawdown(buyhold),
        },
        "series": out.to_dict(orient="records")
    }

if __name__ == "__main__":
    # For demo JSON: load a CSV if you already have price history
    # Or replace with yfinance fetch in your project.
    BASE_DIR = Path(__file__).resolve().parents[1]
    data_path = BASE_DIR / "data" / "demo_prices.csv"

    if not data_path.exists():
        raise SystemExit("Missing data/demo_prices.csv. Create it or wire yfinance fetching.")

    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values("date")
    df.set_index("date", inplace=True)

    # Use Close column
    result = run_demo_backtest(df["close"])

    out_path = BASE_DIR / "models" / "backtest_demo.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")