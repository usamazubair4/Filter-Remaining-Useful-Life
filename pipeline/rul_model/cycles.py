# -*- coding: utf-8 -*-
"""
Replacement-cycle detection: DP peaking at or above PEAK_THRESHOLD and then
falling back to baseline marks one filter-replacement event. Identical logic
to scripts/20_detect_replacement_events.py and the live-status block in
scripts/22_train_rul_models.py, factored out so training and live scoring
can't drift apart.
"""
import pandas as pd

from .config import PEAK_THRESHOLD


def detect_replacements(s: pd.Series, peak_threshold: float, baseline: float,
                         min_gap_hours: int = 24) -> pd.DataFrame:
    """s: a DP series indexed by ascending timestamp. Returns one row per
    detected replacement: peak_time, peak_psi, replacement_time, baseline_psi.
    Consecutive peak excursions within min_gap_hours of each other are
    treated as one event (only the first fall after each is kept)."""
    events = []
    above_peak = s >= peak_threshold
    if not above_peak.any():
        return pd.DataFrame(columns=["peak_time", "peak_psi", "replacement_time", "baseline_psi"])

    idx_above = s.index[above_peak]
    groups, cur = [], [idx_above[0]]
    for t in idx_above[1:]:
        if (t - cur[-1]) <= pd.Timedelta(hours=min_gap_hours):
            cur.append(t)
        else:
            groups.append(cur)
            cur = [t]
    groups.append(cur)

    for grp in groups:
        peak_time = s.loc[grp].idxmax()
        peak_val = s.loc[grp].max()
        after = s.loc[peak_time:]
        fall = after[after <= baseline]
        if len(fall) == 0:
            continue  # never fell back to baseline within the fetched window
        events.append(dict(peak_time=peak_time, peak_psi=round(float(peak_val), 3),
                            replacement_time=fall.index[0], baseline_psi=round(float(fall.iloc[0]), 3)))

    out = pd.DataFrame(events)
    if len(out):
        out = out.drop_duplicates(subset=["replacement_time"]).sort_values("replacement_time").reset_index(drop=True)
    return out


def current_cycle_start(dp_series: pd.Series, baseline: float,
                         peak_threshold: float = PEAK_THRESHOLD) -> pd.Timestamp:
    """The timestamp the current, still-open cycle began: the most recent
    detected replacement in dp_series, or the series' own start if the
    fetched window doesn't reach back far enough to see one. Callers should
    fetch at least LOOKBACK_DAYS of history so this is almost always a real
    replacement, not a fetch-window artifact."""
    events = detect_replacements(dp_series, peak_threshold, baseline)
    if len(events):
        return events["replacement_time"].iloc[-1]
    return dp_series.index[0]


def operational_status(dp_cycle: pd.Series, baseline: float,
                        peak_threshold: float = PEAK_THRESHOLD) -> str:
    """dp_cycle: DP series restricted to the current cycle (cycle_start
    onward). Returns one of the three states scripts/22 reports live:
    'Normal - in service', 'Threshold exceeded, awaiting replacement', or
    'Replaced (flag for review)' -- the last one should never occur for a
    genuinely open cycle; if it does, the cycle-detection pipeline missed
    closing it and that's a signal to investigate, not a real status."""
    hit_threshold = (dp_cycle >= peak_threshold).cummax()
    fell_after = (hit_threshold & (dp_cycle <= baseline)).cummax()
    if fell_after.iloc[-1]:
        return "Replaced (flag for review)"
    if hit_threshold.iloc[-1]:
        return "Threshold exceeded, awaiting replacement"
    return "Normal - in service"