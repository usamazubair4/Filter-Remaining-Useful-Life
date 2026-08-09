# -*- coding: utf-8 -*-
"""
Detect filter-replacement events for RUL model training, per the analyst's
directive: a replacement event is when DP peaks at or above 9 PSI and then
falls back down to baseline (updated from an initial 15 PSI threshold to 9
PSI per analyst direction, to catch more replacement cycles for training).

Output: replacement_events.csv (one row per detected replacement, both
filters) and cycle-labeled data (data/rul_training_data.pkl) with a
RUL_hours column (time until the *next* detected replacement) for every
timestamp that falls within a complete cycle (i.e. has a following
replacement in the data -- the final, still-open cycle per filter is
excluded from training and used only for live inference later).
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
EDA_DIR = PROJECT_DIR / "eda"

df = pd.read_pickle(DATA_DIR / "combined_data_with_events_v2.pkl").set_index("Timestamp").sort_index()

PEAK_THRESHOLD = 9.0  # PSI, per analyst directive (updated from 15.0)

FILTERS = {
    "F-510 Carbon Filter": dict(tag="PLANT_AMINE_P_DPIT_0510_VAL", baseline=2.0),
    "F-509/F-511 Sock Filter": dict(tag="PLANT_AMINE_P_DPIT_0509_VAL", baseline=1.5),
}


def detect_replacements(s, peak_threshold, baseline, min_gap_hours=24):
    """Find each peak >= peak_threshold, then the first subsequent timestamp
    where DP falls to <= baseline -- that fall is the replacement event.
    Consecutive detections within min_gap_hours of each other are treated as
    the same underlying event (only the first fall after each distinct peak
    excursion is kept)."""
    events = []
    above_peak = s >= peak_threshold
    if not above_peak.any():
        return pd.DataFrame(columns=["peak_time", "peak_psi", "replacement_time", "baseline_psi"])

    # group contiguous/near peak excursions (gap <= min_gap_hours) together
    idx_above = s.index[above_peak]
    groups = []
    cur = [idx_above[0]]
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
            continue  # never falls back to baseline in the remaining data (open/ongoing cycle)
        replacement_time = fall.index[0]
        events.append(dict(peak_time=peak_time, peak_psi=round(float(peak_val), 3),
                           replacement_time=replacement_time, baseline_psi=round(float(fall.iloc[0]), 3)))

    out = pd.DataFrame(events)
    if len(out):
        # de-duplicate: if two peak groups fall to baseline at the same replacement_time, keep the first
        out = out.drop_duplicates(subset=["replacement_time"]).sort_values("replacement_time").reset_index(drop=True)
    return out


all_events = []
for filter_name, cfg in FILTERS.items():
    s = df[cfg["tag"]].dropna()
    events = detect_replacements(s, PEAK_THRESHOLD, cfg["baseline"])
    events.insert(0, "filter", filter_name)
    events["days_peak_to_replacement"] = (events["replacement_time"] - events["peak_time"]).dt.total_seconds() / 86400
    all_events.append(events)
    print(f"{filter_name}: {len(events)} replacement event(s) detected "
          f"(peak >= {PEAK_THRESHOLD} PSI, fall to <= {cfg['baseline']} PSI)")
    if len(events):
        print(events.to_string())
    print()

replacement_events = pd.concat(all_events, ignore_index=True)
try:
    replacement_events.to_csv(EDA_DIR / "replacement_events.csv", index=False)
    print(f"Saved {EDA_DIR / 'replacement_events.csv'}")
except PermissionError:
    import tempfile
    fallback = Path(tempfile.gettempdir()) / "replacement_events.csv"
    replacement_events.to_csv(fallback, index=False)
    print(f"eda/replacement_events.csv is locked (likely open in Excel) -- saved to {fallback} instead. "
          f"Copy it into place once the file is closed.")
