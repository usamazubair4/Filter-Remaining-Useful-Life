import pandas as pd
import pytest

from rul_model.cycles import current_cycle_start, detect_replacements, operational_status


def _series(values, start="2026-01-01", freq="1h"):
    idx = pd.date_range(start, periods=len(values), freq=freq)
    return pd.Series(values, index=idx)


def test_detect_replacements_finds_peak_then_fall():
    # Two peak-and-fall excursions, deliberately more than min_gap_hours (24h)
    # apart -- within 24h of each other they'd be merged into one event (see
    # test_close_excursions_within_min_gap_are_merged_into_one_event below),
    # which is the realistic case for two genuinely separate cycles.
    first = [1.5] * 5 + [3, 5, 7, 9.5, 8, 4, 1.5]           # peak at hour 8, falls by hour 10
    flat = [1.5] * 30                                        # >24h of baseline before the next excursion
    second = [2, 4, 6, 9.2, 5, 1.4]                           # peak, falls again
    s = _series(first + flat + second)
    events = detect_replacements(s, peak_threshold=9.0, baseline=1.5)
    assert len(events) == 2
    assert (events["peak_psi"] >= 9.0).all()
    assert (events["baseline_psi"] <= 1.5).all()
    # replacement is detected strictly after its own peak
    assert (events["replacement_time"] > events["peak_time"]).all()


def test_close_excursions_within_min_gap_are_merged_into_one_event():
    # Two above-threshold points only 12h apart -- inside the 24h min_gap,
    # so they're treated as one underlying event (only the higher peak and
    # the first fall after it are kept), even though DP briefly dipped
    # in between. This documents real behavior, not a bug: it exists so a
    # brief instrument wobble mid-replacement doesn't get double-counted.
    values = [1.5] * 5 + [3, 5, 7, 9.5, 8, 4, 1.5] + [1.6] * 5 + [2, 4, 6, 9.2, 5, 1.4]
    s = _series(values)
    events = detect_replacements(s, peak_threshold=9.0, baseline=1.5)
    assert len(events) == 1
    assert events["peak_psi"].iloc[0] == pytest.approx(9.5)  # the higher of the two peaks


def test_detect_replacements_no_peak_returns_empty():
    s = _series([1.0, 1.1, 1.2, 1.3])
    events = detect_replacements(s, peak_threshold=9.0, baseline=1.5)
    assert events.empty
    assert list(events.columns) == ["peak_time", "peak_psi", "replacement_time", "baseline_psi"]


def test_detect_replacements_open_peak_not_yet_fallen_is_not_an_event():
    # peaks but never comes back down within the fetched window
    values = [1.5] * 5 + [3, 5, 7, 9.5, 9.8, 9.6]
    s = _series(values)
    events = detect_replacements(s, peak_threshold=9.0, baseline=1.5)
    assert events.empty


def test_current_cycle_start_is_most_recent_replacement():
    # two excursions >24h apart (see test_detect_replacements_finds_peak_then_fall
    # for why that spacing matters) so they're detected as two distinct events
    first = [1.5] * 3 + [10, 1.4]
    flat = [1.5] * 30
    second = [10, 1.4]  # must actually reach baseline (<=1.5), unlike the flat/tail values
    tail = [2, 2.1, 2.2]
    s = _series(first + flat + second + tail)
    start = current_cycle_start(s, baseline=1.5)
    expected_idx = len(first) + len(flat) + 1  # the second replacement's timestamp
    assert start == s.index[expected_idx]


def test_current_cycle_start_falls_back_to_series_start_when_no_replacement_seen():
    s = _series([2, 2.1, 2.2, 2.3])
    start = current_cycle_start(s, baseline=1.5)
    assert start == s.index[0]


@pytest.mark.parametrize("values,expected", [
    ([1.5, 1.6, 1.7], "Normal - in service"),
    ([1.5, 5.0, 9.5, 8.0], "Threshold exceeded, awaiting replacement"),
    ([1.5, 9.5, 8.0, 1.4], "Replaced (flag for review)"),
])
def test_operational_status(values, expected):
    s = _series(values)
    assert operational_status(s, baseline=1.5) == expected