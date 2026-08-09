# -*- coding: utf-8 -*-
"""
Train RUL (Remaining Useful Life) models: a point-estimate regressor (median,
via quantile loss alpha=0.5) plus P10/P90 quantile regressors for a
confidence interval, using Gradient Boosting on the hourly cycle-relative
dataset from script 21.

Validation: leave-one-cycle-out (LOCO) -- a random row-level split would be
badly optimistic (adjacent hourly rows within a cycle are highly
autocorrelated), so each fold holds out one whole cycle and trains on the
rest. This gives an honest, if wide, estimate of how the model performs on
a cycle it has never seen -- which is the actual deployment scenario
(forecasting the live, still-running cycle).

Three error-reduction changes from the first pass:
  1. `specific_resistance` (DP/Flow) and its 24h slope, engineered in
     script 21 -- isolates true fouling state from flow-rate-driven noise
     in the raw DP reading (for the same fouling level, DP rises with flow,
     so DP alone conflates "more fouled" with "flowing harder right now").
  2. The two cycles flagged `is_anomalous_cycle` in script 21 (both filters
     showing an implausible ~3-day cycle in the same Apr11-16 2026 window,
     more consistent with a shared plant event than independent fouling)
     are excluded from training here, though loaded and inspectable.
  3. Models are fit on log1p(RUL_hours) rather than raw hours. The raw
     target spans 2-68 days (~30x range), so squared/absolute error in raw
     hours implicitly weights errors on long cycles far more than on short
     ones. Log-space balances this -- and since quantiles are preserved
     under a monotonic transform, predicting quantiles of log(RUL) and
     inverting with expm1 still gives valid quantiles of RUL in hours.

IMPORTANT caveat carried through to the report: even 14 cycles (12 after
excluding the anomalous pair) is thin for a production ML model. This is a
legitimate, improving first-pass, not a claim of production-grade accuracy.
See eda/RUL_Model_Report.md.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
EDA_DIR = PROJECT_DIR / "eda"

training_data_full = pd.read_pickle(DATA_DIR / "rul_training_data.pkl")
live_data = pd.read_pickle(DATA_DIR / "rul_live_data.pkl")

n_excluded = training_data_full["is_anomalous_cycle"].sum()
training_data = training_data_full[~training_data_full["is_anomalous_cycle"]].copy()
print(f"Excluding {n_excluded} rows from {training_data_full.loc[training_data_full['is_anomalous_cycle'], 'cycle_id'].nunique()} "
      f"anomalous cycle(s); training on {len(training_data)} rows across "
      f"{training_data['cycle_id'].nunique()} cycles")

FEATURE_COLS = [
    "dp", "dp_slope_6h", "dp_slope_24h", "dp_slope_7d", "dp_std_24h", "hours_since_start",
    "specific_resistance", "specific_resistance_slope_24h",
    "PLANT_AMINE_P_DPIT_7502_VAL", "PLANT_AMINE_P_DPIT_6502_VAL",
    "PLANT_AMINE_P_FIT_2512_VAL", "flow_cummean_this_cycle",
    "PLANT_AMINE_CIRC514VFD21_RPM_FB",
    "PLANT_AMINE_CIRC515VFD22_RPM_FB", "PLANT_AMINE_P_TIT_1504_VAL",
    "PLANT_AMINE_P_PIT_6503_VAL",
    "PLANT_AMINE_P_TIT_3507_VAL", "tit3507_cummean_this_cycle", "tit3507_slope_24h",
]


# Fill values are computed once from the full training set and reused for
# every train/test/live matrix -- computing medians from a single-row live
# inference frame would just return NaN again for any missing feature.
_fill_values = training_data[FEATURE_COLS].median(numeric_only=True)


def build_matrix(data):
    X = data[FEATURE_COLS].copy()
    X["is_carbon_filter"] = (data["filter_type"] == "F-510 Carbon Filter").astype(int)
    X = X.fillna(_fill_values)
    return X


GB_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42)


def train_models(X, y_hours):
    """Fits on log1p(RUL_hours); quantiles are preserved under this monotonic
    transform, so p10/median/p90 predictions just need expm1 to convert back."""
    y_log = np.log1p(y_hours)
    models = {}
    for name, loss_kwargs in [("p10", dict(loss="quantile", alpha=0.1)),
                              ("median", dict(loss="quantile", alpha=0.5)),
                              ("p90", dict(loss="quantile", alpha=0.9))]:
        m = GradientBoostingRegressor(**GB_PARAMS, **loss_kwargs)
        m.fit(X, y_log)
        models[name] = m
    return models


def predict_hours(model, X):
    return np.expm1(model.predict(X))


# ---------------------------------------------------------------------------
# Leave-one-cycle-out validation
# ---------------------------------------------------------------------------
cycle_ids = sorted(training_data["cycle_id"].unique())
loco_results = []
for held_out in cycle_ids:
    train = training_data[training_data["cycle_id"] != held_out]
    test = training_data[training_data["cycle_id"] == held_out]

    X_train, y_train = build_matrix(train), train["RUL_hours"]
    X_test, y_test = build_matrix(test), test["RUL_hours"]

    models = train_models(X_train, y_train)
    pred_median = predict_hours(models["median"], X_test)
    pred_p10 = predict_hours(models["p10"], X_test)
    pred_p90 = predict_hours(models["p90"], X_test)

    mae_hours = mean_absolute_error(y_test, pred_median)
    mae_days = mae_hours / 24
    coverage = np.mean((y_test.values >= pred_p10) & (y_test.values <= pred_p90))
    filter_name = test["filter_type"].iloc[0]

    loco_results.append(dict(
        cycle_id=held_out, filter=filter_name,
        n_test_rows=len(test), mae_hours=round(mae_hours, 1), mae_days=round(mae_days, 2),
        p10_p90_coverage=round(coverage, 3),
        actual_cycle_length_days=round(test["RUL_hours"].max() / 24, 1),
    ))
    print(f"Held-out cycle {held_out} ({filter_name}): MAE = {mae_days:.2f} days, "
          f"P10-P90 coverage = {coverage*100:.1f}% (target ~80%)")

loco_df = pd.DataFrame(loco_results)
loco_df.to_csv(EDA_DIR / "rul_loco_validation.csv", index=False)

# ---------------------------------------------------------------------------
# Final models trained on ALL complete cycles (for live inference)
# ---------------------------------------------------------------------------
X_all, y_all = build_matrix(training_data), training_data["RUL_hours"]
final_models = train_models(X_all, y_all)

# Feature importance (from the median model)
importances = pd.Series(final_models["median"].feature_importances_, index=X_all.columns).sort_values(ascending=False)
importances.to_csv(EDA_DIR / "rul_feature_importance.csv", header=["importance"])
print("\nFeature importance (median model):")
print(importances.to_string())

# A DP-trend-only variant (excluding hours_since_start) is trained alongside
# as a sanity-check comparison -- useful whenever a live cycle's age exceeds
# what the primary model saw in training, since "elapsed time" based
# reasoning breaks down under extrapolation while trend-based reasoning
# degrades more gracefully.
TREND_ONLY_COLS = [c for c in FEATURE_COLS if c != "hours_since_start"]


def build_matrix_trend_only(data):
    X = data[TREND_ONLY_COLS].copy()
    X["is_carbon_filter"] = (data["filter_type"] == "F-510 Carbon Filter").astype(int)
    X = X.fillna(_fill_values[TREND_ONLY_COLS])
    return X


X_all_trend, y_all_trend = build_matrix_trend_only(training_data), training_data["RUL_hours"]
trend_only_models = train_models(X_all_trend, y_all_trend)

max_train_age_days = training_data.groupby("cycle_id")["hours_since_start"].max() / 24
max_train_age_by_filter = training_data.assign(
    age_days=training_data["hours_since_start"] / 24
).groupby("filter_type")["age_days"].max()

# ---------------------------------------------------------------------------
# Persist everything a live scoring job needs -- the 6 fitted regressors
# (primary + trend-only, each P10/median/P90), the median fill values used
# for missing features, the exact feature-column order/order each model
# expects, and the per-filter max-training-age used for the out-of-range
# warning. Nothing here is retrained at inference time; scoring only ever
# loads this bundle and calls .predict().
# ---------------------------------------------------------------------------
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
model_bundle = dict(
    final_models=final_models,
    trend_only_models=trend_only_models,
    fill_values=_fill_values,
    feature_cols=FEATURE_COLS,
    trend_only_cols=TREND_ONLY_COLS,
    max_train_age_by_filter=max_train_age_by_filter.to_dict(),
    peak_threshold=9.0,  # PSI -- must match scripts/20 and 21's PEAK_THRESHOLD
    trained_at=datetime.now(timezone.utc).isoformat(),
    n_training_rows=len(training_data),
    n_complete_cycles=len(cycle_ids),
)
joblib.dump(model_bundle, MODELS_DIR / "rul_model_bundle.joblib")
print(f"\nSaved live-scoring model bundle to {MODELS_DIR / 'rul_model_bundle.joblib'} "
      f"(6 regressors + fill values + feature columns + max training age)")

# ---------------------------------------------------------------------------
# Live inference: current open cycle for each filter
# ---------------------------------------------------------------------------
live_results = []
for filter_name in live_data["filter_type"].unique():
    sub = live_data[live_data["filter_type"] == filter_name].sort_index()
    latest = sub.iloc[[-1]]
    cycle_age_days = latest["hours_since_start"].iloc[0] / 24
    max_seen_days = max_train_age_by_filter.get(filter_name, np.nan)
    out_of_range = cycle_age_days > max_seen_days
    operational_status = ("Replaced (flag for review)" if latest["replaced"].iloc[0]
                          else "Threshold exceeded, awaiting replacement" if latest["threshold_hit"].iloc[0]
                          else "Normal - in service")

    X_live = build_matrix(latest)
    p10 = float(predict_hours(final_models["p10"], X_live)[0])
    median = float(predict_hours(final_models["median"], X_live)[0])
    p90 = float(predict_hours(final_models["p90"], X_live)[0])
    p10, median, p90 = sorted([p10, median, p90])

    X_live_trend = build_matrix_trend_only(latest)
    trend_median = float(predict_hours(trend_only_models["median"], X_live_trend)[0])

    as_of = sub.index[-1]
    current_dp = latest["dp"].iloc[0]
    live_results.append(dict(
        filter=filter_name, as_of=as_of, current_dp_psi=round(float(current_dp), 3),
        operational_status=operational_status,
        cycle_age_days=round(cycle_age_days, 1),
        max_training_cycle_age_days_seen=round(max_seen_days, 1),
        out_of_training_range=out_of_range,
        RUL_p10_days=round(p10 / 24, 1), RUL_median_days=round(median / 24, 1), RUL_p90_days=round(p90 / 24, 1),
        RUL_median_days_trend_only_model=round(trend_median / 24, 1),
        predicted_replacement_p10=as_of + pd.Timedelta(hours=p10),
        predicted_replacement_median=as_of + pd.Timedelta(hours=median),
        predicted_replacement_p90=as_of + pd.Timedelta(hours=p90),
    ))
    print(f"\n{filter_name} -- as of {as_of}, current DP {current_dp:.2f} PSI, status = {operational_status}, "
          f"cycle age {cycle_age_days:.1f} days (max seen in training: {max_seen_days:.1f} days):")
    if out_of_range:
        print(f"  *** WARNING: this cycle is already OLDER than any complete training cycle. "
              f"Treat this forecast with low confidence -- see feature importance in "
              f"rul_feature_importance.csv for how much the model leans on elapsed time. ***")
    print(f"  Primary model RUL: P10={p10/24:.1f}d  median={median/24:.1f}d  P90={p90/24:.1f}d")
    print(f"  DP-trend-only model (no time-since-start feature) median RUL: {trend_median/24:.1f}d "
          f"-- compare against the primary model as a sanity check")
    print(f"  Predicted replacement window: {as_of + pd.Timedelta(hours=p10):%Y-%m-%d} to "
          f"{as_of + pd.Timedelta(hours=p90):%Y-%m-%d} (median {as_of + pd.Timedelta(hours=median):%Y-%m-%d})")

live_results_df = pd.DataFrame(live_results)
live_results_df.to_csv(EDA_DIR / "rul_live_forecast.csv", index=False)

with open(EDA_DIR / "rul_model_metadata.json", "w") as f:
    json.dump(dict(
        feature_cols=FEATURE_COLS + ["is_carbon_filter"],
        gb_params=GB_PARAMS,
        n_complete_cycles=len(cycle_ids),
        n_training_rows=len(training_data),
        loco_mean_mae_days=round(loco_df["mae_days"].mean(), 2),
        loco_mean_coverage=round(loco_df["p10_p90_coverage"].mean(), 3),
    ), f, indent=2, default=str)

print(f"\nSaved {EDA_DIR / 'rul_loco_validation.csv'}, {EDA_DIR / 'rul_feature_importance.csv'}, "
      f"{EDA_DIR / 'rul_live_forecast.csv'}, {EDA_DIR / 'rul_model_metadata.json'}")
