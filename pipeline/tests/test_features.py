import numpy as np
import pandas as pd
import pytest

from rul_model.config import FEATURE_COLS, FEATURE_TAGS, FIT_2512_TAG, TIT_3507_TAG
from rul_model.features import engineer_features

DP_TAG = "PLANT_AMINE_P_DPIT_0510_VAL"


def _synthetic_cycle(n_hours=200, start="2026-01-01"):
    idx = pd.date_range(start, periods=n_hours, freq="1h")
    rng = np.random.default_rng(0)
    dp = np.linspace(1.5, 6.0, n_hours) + rng.normal(0, 0.05, n_hours)
    flow = 30 + rng.normal(0, 1, n_hours)
    temp = 105 + rng.normal(0, 0.5, n_hours)
    data = {DP_TAG: dp, FIT_2512_TAG: flow, TIT_3507_TAG: temp}
    # engineer_features() only emits a FEATURE_TAGS column if it's present in
    # `sub` -- a real live pull always carries all of them (tags_config.csv
    # maps every one), so the fixture must too or this test is checking
    # nothing about the tags it doesn't bother to include.
    for tag in FEATURE_TAGS:
        data.setdefault(tag, 50 + rng.normal(0, 1, n_hours))
    return pd.DataFrame(data, index=idx)


def test_engineer_features_produces_every_model_column():
    sub = _synthetic_cycle()
    feat = engineer_features(sub, DP_TAG)
    for col in FEATURE_COLS:
        assert col in feat.columns, f"missing expected feature column {col!r}"


def test_hours_since_start_begins_at_zero_and_is_cycle_relative():
    sub = _synthetic_cycle()
    feat = engineer_features(sub, DP_TAG)
    assert feat["hours_since_start"].iloc[0] == 0
    assert feat["hours_since_start"].iloc[-1] == pytest.approx(len(sub) - 1)


def test_rolling_features_do_not_leak_across_cycle_boundary():
    # a cycle only 10 hours old cannot have a real 7-day (168h) slope --
    # this is the load-bearing property documented in features.py
    sub = _synthetic_cycle(n_hours=10)
    feat = engineer_features(sub, DP_TAG)
    assert feat["dp_slope_7d"].isna().all()
    assert feat["dp_slope_24h"].isna().all()


def test_cumulative_mean_starts_at_the_first_reading():
    sub = _synthetic_cycle()
    feat = engineer_features(sub, DP_TAG)
    assert feat["flow_cummean_this_cycle"].iloc[0] == pytest.approx(sub[FIT_2512_TAG].iloc[0])
    assert feat["tit3507_cummean_this_cycle"].iloc[0] == pytest.approx(sub[TIT_3507_TAG].iloc[0])


def test_specific_resistance_is_dp_over_flow_with_a_floor():
    sub = _synthetic_cycle()
    sub.loc[sub.index[5], FIT_2512_TAG] = 0.1  # near-zero flow should be floored, not blow up the ratio
    feat = engineer_features(sub, DP_TAG)
    assert np.isfinite(feat["specific_resistance"]).all()
    expected_row0 = sub[DP_TAG].iloc[0] / max(sub[FIT_2512_TAG].iloc[0], 1.0)
    assert feat["specific_resistance"].iloc[0] == pytest.approx(expected_row0)