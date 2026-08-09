# -*- coding: utf-8 -*-
"""
Load the trained model bundle (scripts/22_train_rul_models.py's
models/rul_model_bundle.joblib) and score one live cycle. Mirrors
build_matrix() and the live-inference block in scripts/22 exactly, so a
prediction made here matches what that script would have printed.
"""
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


def load_bundle(path: Union[str, Path]) -> dict:
    import joblib
    return joblib.load(path)


def build_matrix(data: pd.DataFrame, feature_cols: list, fill_values: pd.Series, is_carbon: bool) -> pd.DataFrame:
    X = data[feature_cols].copy()
    X["is_carbon_filter"] = int(is_carbon)
    return X.fillna(fill_values)


def predict_hours(model, X: pd.DataFrame) -> np.ndarray:
    """Models are fit on log1p(RUL_hours); quantiles are preserved under
    this monotonic transform, so expm1 inverts straight back to hours."""
    return np.expm1(model.predict(X))


def score_latest_row(latest: pd.DataFrame, bundle: dict, is_carbon: bool) -> dict:
    """latest: single-row DataFrame -- the most recent hourly feature row
    for one filter's current cycle (output of engineer_features().tail(1)).
    Returns P10/median/P90 RUL in hours and days for the primary model, plus
    the trend-only model's median as a cross-check, exactly as scripts/22
    reports for live forecasts."""
    fill = bundle["fill_values"]
    feature_cols = bundle["feature_cols"]
    trend_cols = bundle["trend_only_cols"]

    X = build_matrix(latest, feature_cols, fill, is_carbon)
    p10 = float(predict_hours(bundle["final_models"]["p10"], X)[0])
    median = float(predict_hours(bundle["final_models"]["median"], X)[0])
    p90 = float(predict_hours(bundle["final_models"]["p90"], X)[0])
    p10, median, p90 = sorted([p10, median, p90])

    X_trend = build_matrix(latest, trend_cols, fill[trend_cols], is_carbon)
    trend_median = float(predict_hours(bundle["trend_only_models"]["median"], X_trend)[0])

    return dict(
        rul_p10_hours=p10, rul_median_hours=median, rul_p90_hours=p90,
        rul_p10_days=round(p10 / 24, 1), rul_median_days=round(median / 24, 1), rul_p90_days=round(p90 / 24, 1),
        trend_only_median_days=round(trend_median / 24, 1),
    )