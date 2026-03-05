# stock_ml/modeling.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS, MODELS_DIR


@dataclass
class CVMetrics:
    auc: float
    precision: float
    recall: float


@dataclass
class CVSummary:
    per_fold: List[CVMetrics]
    avg_auc: float
    avg_precision: float
    avg_recall: float


def time_slices(
    df: pd.DataFrame,
    train_years: int = 5,
    test_years: int = 1,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate boolean masks for rolling train/test splits based on calendar years.

    Example:
      train: 2012-2016, test: 2017
      train: 2013-2017, test: 2018
      ...
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex for time_slices.")

    years = sorted(df.index.year.unique())
    slices: List[Tuple[np.ndarray, np.ndarray]] = []

    for i in range(len(years) - train_years - test_years + 1):
        train_start = years[i]
        train_end = years[i + train_years - 1]
        test_start = years[i + train_years]
        test_end = years[i + train_years + test_years - 1]

        train_mask = (df.index.year >= train_start) & (df.index.year <= train_end)
        test_mask = (df.index.year >= test_start) & (df.index.year <= test_end)

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        slices.append((train_mask, test_mask))

    if not slices:
        raise ValueError("Not enough years in data to create time-based slices.")

    return slices


def cross_validate_logistic(
    df: pd.DataFrame,
    label_col: str,
    C: float = 1.0,
    max_iter: int = 1000,
) -> CVSummary:
    """
    Perform walk-forward cross-validation for a logistic regression model
    for a given label column (label_st or label_sw).
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

    df = df.copy()
    # Ensure sorted
    df.sort_index(inplace=True)

    X_all = df[FEATURE_COLUMNS].values
    y_all = df[label_col].values

    slices = time_slices(df)

    metrics_list: List[CVMetrics] = []

    for idx, (train_mask, test_mask) in enumerate(slices, start=1):
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(C=C, max_iter=max_iter)
        clf.fit(X_train_scaled, y_train)

        probas = clf.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, probas)

        y_pred = (probas >= 0.5).astype(int)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        metrics = CVMetrics(auc=auc, precision=prec, recall=rec)
        metrics_list.append(metrics)

        print(
            f"[CV] Fold {idx}: AUC={auc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}"
        )

    avg_auc = float(np.mean([m.auc for m in metrics_list]))
    avg_prec = float(np.mean([m.precision for m in metrics_list]))
    avg_rec = float(np.mean([m.recall for m in metrics_list]))

    print(
        f"[CV] Averages: AUC={avg_auc:.3f}, "
        f"Precision={avg_prec:.3f}, Recall={avg_rec:.3f}"
    )

    return CVSummary(
        per_fold=metrics_list,
        avg_auc=avg_auc,
        avg_precision=avg_prec,
        avg_recall=avg_rec,
    )


def train_final_logistic(
    df: pd.DataFrame,
    label_col: str,
    model_name_prefix: str,
    C: float = 1.0,
    max_iter: int = 1000,
) -> Tuple[LogisticRegression, StandardScaler]:
    """
    Train a final logistic regression model on ALL available data for a label,
    and persist the model + scaler to disk.
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

    df = df.copy()
    df.sort_index(inplace=True)

    X = df[FEATURE_COLUMNS].values
    y = df[label_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(C=C, max_iter=max_iter)
    clf.fit(X_scaled, y)

    model_path = MODELS_DIR / f"{model_name_prefix}_logreg.pkl"
    scaler_path = MODELS_DIR / f"{model_name_prefix}_scaler.pkl"

    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"[train_final_logistic] Saved model to: {model_path}")
    print(f"[train_final_logistic] Saved scaler to: {scaler_path}")

    return clf, scaler
