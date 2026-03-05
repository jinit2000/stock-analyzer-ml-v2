# scripts/train_models.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from stock_ml.config import DATA_DIR, MODELS_DIR
from stock_ml.modeling import cross_validate_logistic, train_final_logistic


def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _infer_feature_cols(df: pd.DataFrame, label_cols: set[str]) -> list[str]:
    """
    Pick numeric feature columns automatically:
    - numeric dtypes
    - excludes labels and common non-features
    """
    exclude = set(label_cols) | {"ticker", "symbol", "date"}
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError(
            "No numeric feature columns detected. "
            "Check your dataset.parquet columns."
        )
    return numeric_cols


def _evaluate_timeseries_metrics(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: list[str],
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    TimeSeriesSplit evaluation for honest backtesting-style metrics.
    Trains on past, tests on future.
    Returns averaged metrics across folds.
    """
    X = df[feature_cols].astype(float).values
    y = df[label_col].astype(int).values

    tscv = TimeSeriesSplit(n_splits=n_splits)

    accs, precs, recs, f1s, aucs = [], [], [], [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features fold-wise to avoid leakage
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(
            max_iter=2000,
            n_jobs=1,
            random_state=random_state,
            solver="lbfgs",
        )
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)

        # Probabilities for AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_s)[:, 1]
        else:
            # Fallback (shouldn't happen for LogisticRegression)
            y_prob = y_pred.astype(float)

        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, zero_division=0))
        recs.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))

        # AUC requires both classes to be present in test fold
        try:
            aucs.append(roc_auc_score(y_test, y_prob))
        except ValueError:
            # If a fold has only one class, skip it
            pass

    metrics: Dict[str, float] = {
        "accuracy": float(np.mean(accs)) if accs else 0.0,
        "precision": float(np.mean(precs)) if precs else 0.0,
        "recall": float(np.mean(recs)) if recs else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "roc_auc": float(np.mean(aucs)) if aucs else 0.0,
    }
    return metrics


def _write_metrics_json(metrics: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main():
    dataset_path = DATA_DIR / "dataset.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            f"Run scripts/build_dataset.py first."
        )

    df = pd.read_parquet(dataset_path)
    df = _ensure_dt_index(df)

    # --- Existing CV (your original behavior) ---
    print("[train_models] Cross-validating short-term (label_st) model...")
    cv_st = cross_validate_logistic(df, label_col="label_st")

    print("[train_models] Cross-validating swing (label_sw) model...")
    cv_sw = cross_validate_logistic(df, label_col="label_sw")

    # --- NEW: Backtest-style metrics for UI accuracy section ---
    label_cols = {"label_st", "label_sw"}
    feature_cols = _infer_feature_cols(df, label_cols=label_cols)

    print("[train_models] Computing time-series evaluation metrics (short-term)...")
    st_metrics = _evaluate_timeseries_metrics(df, "label_st", feature_cols=feature_cols, n_splits=5)

    print("[train_models] Computing time-series evaluation metrics (swing)...")
    sw_metrics = _evaluate_timeseries_metrics(df, "label_sw", feature_cols=feature_cols, n_splits=5)

    metrics_payload = {
        "short_term": {
            "horizon_days": 10,
            "target_return": 0.02,
            **st_metrics,
            "last_trained": datetime.utcnow().strftime("%Y-%m-%d"),
            "notes": "TimeSeriesSplit metrics (train on past, test on future).",
        },
        "swing": {
            "horizon_days": 60,
            "target_return": 0.05,
            **sw_metrics,
            "last_trained": datetime.utcnow().strftime("%Y-%m-%d"),
            "notes": "TimeSeriesSplit metrics (train on past, test on future).",
        },
    }

    metrics_path = MODELS_DIR / "metrics.json"
    _write_metrics_json(metrics_payload, metrics_path)

    # --- Train final models as before ---
    print("[train_models] Training final short-term model on full data...")
    train_final_logistic(df, label_col="label_st", model_name_prefix="short")

    print("[train_models] Training final swing model on full data...")
    train_final_logistic(df, label_col="label_sw", model_name_prefix="swing")

    print("[train_models] Done.")
    print("[train_models] Short-term CV average:", cv_st)
    print("[train_models] Swing CV average:", cv_sw)
    print(f"[train_models] Wrote metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()