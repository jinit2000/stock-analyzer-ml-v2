# stock_ml/backtesting.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS
from .modeling import time_slices  # reuse time-based splits


@dataclass
class SignalStats:
    n_signals: int
    avg_return: float
    median_return: float
    win_rate: float
    hit_rate_target: float


@dataclass
class BacktestResult:
    label_col: str
    ret_col: str
    horizon_name: str
    avg_return_all: float
    avg_return_positive_labels: float
    stats_buy: SignalStats
    stats_strong_buy: SignalStats


def _compute_signal_stats(
    returns: np.ndarray,
    target_threshold: float,
) -> SignalStats:
    """Compute stats for a set of trade returns."""
    if returns.size == 0:
        return SignalStats(
            n_signals=0,
            avg_return=float("nan"),
            median_return=float("nan"),
            win_rate=float("nan"),
            hit_rate_target=float("nan"),
        )

    avg_ret = float(np.mean(returns))
    med_ret = float(np.median(returns))
    win_rate = float(np.mean(returns >= 0.0))
    hit_rate = float(np.mean(returns >= target_threshold))

    return SignalStats(
        n_signals=int(returns.size),
        avg_return=avg_ret,
        median_return=med_ret,
        win_rate=win_rate,
        hit_rate_target=hit_rate,
    )


def backtest_logistic_signals(
    df: pd.DataFrame,
    label_col: str,
    ret_col: str,
    prob_thresholds: Dict[str, float],
    target_return_threshold: float,
    horizon_name: str,
    C: float = 1.0,
    max_iter: int = 1000,
    train_years: int = 5,
    test_years: int = 1,
) -> BacktestResult:
    """
    Walk-forward backtest for logistic regression signals.

    For each time slice:
      - Train on past years.
      - Predict probabilities on next year.
      - Classify each day as STRONG BUY / BUY / HOLD / SELL using prob_thresholds.
      - Collect realized returns for ret_col over the horizon.

    We then compute:
      - average return of all test days,
      - average return of days where predicted BUY/STRONG BUY,
      - stats for BUY and STRONG BUY separately.
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame.")
    if ret_col not in df.columns:
        raise ValueError(f"Return column '{ret_col}' not found in DataFrame.")

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    df = df.sort_index().copy()

    X_all = df[FEATURE_COLUMNS].values
    y_all = df[label_col].values
    ret_all = df[ret_col].values

    slices = time_slices(df, train_years=train_years, test_years=test_years)

    # Collect returns & probabilities from all test folds
    all_test_returns: List[float] = []
    all_test_probs: List[float] = []

    for fold_idx, (train_mask, test_mask) in enumerate(slices, start=1):
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test = X_all[test_mask]
        ret_test = ret_all[test_mask]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(C=C, max_iter=max_iter)
        clf.fit(X_train_scaled, y_train)

        probas = clf.predict_proba(X_test_scaled)[:, 1]

        all_test_returns.extend(ret_test.tolist())
        all_test_probs.extend(probas.tolist())

        print(
            f"[backtest] Fold {fold_idx}: "
            f"n_train={train_mask.sum()}, n_test={test_mask.sum()}"
        )

    all_test_returns = np.array(all_test_returns)
    all_test_probs = np.array(all_test_probs)

    # Average return of ALL test days (baseline, no signals)
    avg_return_all = float(np.mean(all_test_returns))

    # Average return when the true label was 1 (for reference)
    positive_mask = (df[label_col].values == 1)
    avg_return_pos_labels = float(
        np.mean(df.loc[df.index[positive_mask], ret_col].values)
    ) if positive_mask.any() else float("nan")

    # Convert probabilities into STRONG BUY / BUY / HOLD / SELL
    strong_thr = prob_thresholds["strong_buy"]
    buy_thr = prob_thresholds["buy"]
    hold_thr = prob_thresholds["hold"]

    is_strong_buy = all_test_probs >= strong_thr
    is_buy = (all_test_probs >= buy_thr) & (all_test_probs < strong_thr)
    # Holds and sells not directly used in stats, but you could add later.

    returns_strong_buy = all_test_returns[is_strong_buy]
    returns_buy = all_test_returns[is_buy]

    stats_strong_buy = _compute_signal_stats(
        returns_strong_buy, target_threshold=target_return_threshold
    )
    stats_buy = _compute_signal_stats(
        returns_buy, target_threshold=target_return_threshold
    )

    print(f"[backtest] Horizon={horizon_name}")
    print(f"  Baseline avg return (all days): {avg_return_all:.4f}")
    print(
        f"  Avg return when true label=1 (ground truth): "
        f"{avg_return_pos_labels:.4f}"
    )
    print(
        f"  STRONG BUY: n={stats_strong_buy.n_signals}, "
        f"avg_ret={stats_strong_buy.avg_return:.4f}, "
        f"win_rate={stats_strong_buy.win_rate:.3f}, "
        f"hit_rate_target={stats_strong_buy.hit_rate_target:.3f}"
    )
    print(
        f"  BUY:        n={stats_buy.n_signals}, "
        f"avg_ret={stats_buy.avg_return:.4f}, "
        f"win_rate={stats_buy.win_rate:.3f}, "
        f"hit_rate_target={stats_buy.hit_rate_target:.3f}"
    )

    return BacktestResult(
        label_col=label_col,
        ret_col=ret_col,
        horizon_name=horizon_name,
        avg_return_all=avg_return_all,
        avg_return_positive_labels=avg_return_pos_labels,
        stats_buy=stats_buy,
        stats_strong_buy=stats_strong_buy,
    )
