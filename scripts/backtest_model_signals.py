# scripts/backtest_model_signals.py

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_ml.config import (
    DATA_DIR,
    SHORT_HORIZON_DAYS,
    SHORT_RETURN_THRESHOLD,
    SWING_HORIZON_DAYS,
    SWING_RETURN_THRESHOLD,
    SHORT_PROB_THRESHOLDS,
    SWING_PROB_THRESHOLDS,
)
from stock_ml.backtesting import backtest_logistic_signals


def main():
    dataset_path = DATA_DIR / "dataset.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            f"Run scripts/build_dataset.py first."
        )

    df = pd.read_parquet(dataset_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    print("=== BACKTEST: Short-term horizon ===")
    print(
        f" Target: +{SHORT_RETURN_THRESHOLD * 100:.1f}% "
        f"in {SHORT_HORIZON_DAYS} trading days"
    )

    result_short = backtest_logistic_signals(
        df=df,
        label_col="label_st",
        ret_col="ret_st",
        prob_thresholds=SHORT_PROB_THRESHOLDS,
        target_return_threshold=SHORT_RETURN_THRESHOLD,
        horizon_name=f"{SHORT_HORIZON_DAYS}d",
        C=1.0,
        max_iter=1000,
        train_years=5,
        test_years=1,
    )

    print("\n=== BACKTEST: Swing horizon ===")
    print(
        f" Target: +{SWING_RETURN_THRESHOLD * 100:.1f}% "
        f"in {SWING_HORIZON_DAYS} trading days"
    )

    result_swing = backtest_logistic_signals(
        df=df,
        label_col="label_sw",
        ret_col="ret_sw",
        prob_thresholds=SWING_PROB_THRESHOLDS,
        target_return_threshold=SWING_RETURN_THRESHOLD,
        horizon_name=f"{SWING_HORIZON_DAYS}d",
        C=1.0,
        max_iter=1000,
        train_years=5,
        test_years=1,
    )

    print("\n=== SUMMARY (Short-term) ===")
    print(f" Horizon: {result_short.horizon_name}")
    print(
        f" Baseline avg return (all days): "
        f"{result_short.avg_return_all:.4f}"
    )
    print(
        f" Avg return when true label=1: "
        f"{result_short.avg_return_positive_labels:.4f}"
    )
    print(
        f" STRONG BUY signals: n={result_short.stats_strong_buy.n_signals}, "
        f"avg_ret={result_short.stats_strong_buy.avg_return:.4f}, "
        f"win_rate={result_short.stats_strong_buy.win_rate:.3f}, "
        f"hit_rate_target={result_short.stats_strong_buy.hit_rate_target:.3f}"
    )
    print(
        f" BUY signals:        n={result_short.stats_buy.n_signals}, "
        f"avg_ret={result_short.stats_buy.avg_return:.4f}, "
        f"win_rate={result_short.stats_buy.win_rate:.3f}, "
        f"hit_rate_target={result_short.stats_buy.hit_rate_target:.3f}"
    )

    print("\n=== SUMMARY (Swing) ===")
    print(f" Horizon: {result_swing.horizon_name}")
    print(
        f" Baseline avg return (all days): "
        f"{result_swing.avg_return_all:.4f}"
    )
    print(
        f" Avg return when true label=1: "
        f"{result_swing.avg_return_positive_labels:.4f}"
    )
    print(
        f" STRONG BUY signals: n={result_swing.stats_strong_buy.n_signals}, "
        f"avg_ret={result_swing.stats_strong_buy.avg_return:.4f}, "
        f"win_rate={result_swing.stats_strong_buy.win_rate:.3f}, "
        f"hit_rate_target={result_swing.stats_strong_buy.hit_rate_target:.3f}"
    )
    print(
        f" BUY signals:        n={result_swing.stats_buy.n_signals}, "
        f"avg_ret={result_swing.stats_buy.avg_return:.4f}, "
        f"win_rate={result_swing.stats_buy.win_rate:.3f}, "
        f"hit_rate_target={result_swing.stats_buy.hit_rate_target:.3f}"
    )


if __name__ == "__main__":
    main()
