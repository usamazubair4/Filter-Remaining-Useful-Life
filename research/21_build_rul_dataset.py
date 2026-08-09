# -*- coding: utf-8 -*-
"""
Build the RUL (Remaining Useful Life) training dataset from the detected
replacement events (script 20): for every hourly timestamp inside a
*complete* cycle (bounded by two detected replacements), label
RUL_hours = time until the next replacement, and engineer features.

Cycle inventory (from eda/replacement_events.csv):
  - Any cycle from the start of the dataset to the FIRST detected
    replacement is LEFT-CENSORED (true filter age at data start is unknown)
    and excluded from training.
  - Any cycle from the LAST detected replacement to the end of the dataset
    is the current, still-running cycle (right-censored) -- excluded from
    training, kept separately as the live inference case.
  - Only fully-bounded cycles (replacement -> next replacement) are used as
    training labels.

Both filters are pooled into one dataset (filter_type as a categorical
feature) because each filter alone has only 1-2 complete cycles -- pooling
is a deliberate, documented choice to get a usable-if-still-thin training
set, not a claim that the two filters behave identically.

Threshold updated to 9 PSI (from an initial 15 PSI) per analyst direction,
which increased complete cycles from 3 to 14 (7 independent time periods,
mirrored across both filters since they share the same upstream conditions).

Analyst hypotheses tested against these 7 independent F-510 cycles:
  - "Low FIT-2512 flow -> longer cycle; high flow -> shorter cycle":
    CONFIRMED directionally (Pearson r=-0.61, Spearman r=-0.75 between mean
    cycle flow and cycle duration), though not statistically significant at
    this sample size (p=0.15, n=7). Physically plausible (more throughput =
    faster contaminant loading), and included as a feature below.
  - "More TIT-1504 fluctuation -> shorter cycle; stable temp -> longer cycle":
    NOT SUPPORTED. Tested four ways (std, range, mean/max absolute hourly
    change) -- all near-zero correlation with cycle duration (r between
    -0.12 and +0.06, p between 0.79-0.98). No temperature-volatility feature
    was added on this basis; TIT-1504's raw value remains in the feature set
    as general context only.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
EDA_DIR = PROJECT_DIR / "eda"

df = pd.read_pickle(DATA_DIR / "combined_data_with_events_v2.pkl").set_index("Timestamp").sort_index()
events = pd.read_csv(EDA_DIR / "replacement_events.csv", parse_dates=["peak_time", "replacement_time"])

FILTER_CFG = {
    "F-510 Carbon Filter": dict(dp_tag="PLANT_AMINE_P_DPIT_0510_VAL", he_dp_tag="PLANT_AMINE_P_DPIT_7502_VAL", baseline=2.0),
    "F-509/F-511 Sock Filter": dict(dp_tag="PLANT_AMINE_P_DPIT_0509_VAL", he_dp_tag="PLANT_AMINE_P_DPIT_6502_VAL", baseline=1.5),
}

# Must match script 20's PEAK_THRESHOLD -- the detection rule is: DP reaching
# this level counts as "due for replacement"; it is only actually marked
# replaced once DP has *subsequently* fallen back to baseline. For a live,
# still-open cycle this same rule is evaluated below to report a current
# status (Normal / Threshold exceeded, awaiting replacement) -- a completed
# threshold-hit-then-fall within an "open" cycle can't happen by construction
# (script 20 would already have detected it as a closed cycle), so the live
# status can only ever read Normal or "awaiting replacement", never "replaced".
PEAK_THRESHOLD = 9.0

FEATURE_TAGS = [
    "PLANT_AMINE_P_DPIT_7502_VAL", "PLANT_AMINE_P_DPIT_6502_VAL",
    "PLANT_AMINE_P_FIT_2512_VAL", "PLANT_AMINE_CIRC514VFD21_RPM_FB",
    "PLANT_AMINE_CIRC515VFD22_RPM_FB", "PLANT_AMINE_P_TIT_1504_VAL",
    "PLANT_AMINE_P_PIT_6503_VAL", "PLANT_AMINE_P_TIT_3507_VAL",
]


FIT_2512_TAG = "PLANT_AMINE_P_FIT_2512_VAL"
TIT_3507_TAG = "PLANT_AMINE_P_TIT_3507_VAL"  # flash/inlet temperature


def engineer_features(sub, dp_tag):
    """sub: dataframe already resampled hourly, indexed by Timestamp, for one cycle."""
    dp = sub[dp_tag]
    feat = pd.DataFrame(index=sub.index)
    feat["dp"] = dp
    feat["dp_slope_6h"] = dp.diff(6) / 6
    feat["dp_slope_24h"] = dp.diff(24) / 24
    feat["dp_slope_7d"] = dp.diff(24 * 7) / (24 * 7)
    feat["dp_std_24h"] = dp.rolling(24, min_periods=6).std()
    feat["hours_since_start"] = (sub.index - sub.index[0]).total_seconds() / 3600
    for tag in FEATURE_TAGS:
        if tag in sub.columns:
            feat[tag] = sub[tag]
    # Cycle-cumulative mean flow: analyst-confirmed signal (r=-0.61 across 7
    # independent F-510 cycles: higher average circulation flow -> shorter
    # cycle, i.e. more amine throughput -> faster contaminant loading). A
    # running mean from cycle start captures "this cycle has been running at
    # high/low flow overall so far", which is what correlated with cycle
    # length -- the instantaneous FIT-2512 reading alone does not carry this
    # signal as directly.
    if FIT_2512_TAG in sub.columns:
        feat["flow_cummean_this_cycle"] = sub[FIT_2512_TAG].expanding(min_periods=1).mean()
        # Specific resistance (DP normalized by flow): isolates true fouling
        # state from flow-rate-driven noise in the raw DP reading -- for the
        # same fouling level, DP rises with flow, so DP alone conflates "more
        # fouled" with "flowing harder right now". DP/Flow (analogous to a
        # filtration resistance/inverse-permeability coefficient) should rise
        # monotonically with fouling regardless of flow fluctuations.
        flow_safe = sub[FIT_2512_TAG].clip(lower=1.0)  # avoid divide-by-near-zero blowups
        feat["specific_resistance"] = dp / flow_safe
        feat["specific_resistance_slope_24h"] = feat["specific_resistance"].diff(24) / 24
    # Inlet/flash temperature (TIT-3507): cycle-cumulative mean, analogous to
    # flow_cummean_this_cycle, to test whether average inlet temperature over
    # the cycle-so-far carries a fouling-rate signal (e.g. higher temp ->
    # different amine degradation/foaming byproduct load -> different cycle
    # length), plus a 24h slope for short-term trend.
    if TIT_3507_TAG in sub.columns:
        feat["tit3507_cummean_this_cycle"] = sub[TIT_3507_TAG].expanding(min_periods=1).mean()
        feat["tit3507_slope_24h"] = sub[TIT_3507_TAG].diff(24) / 24
    return feat


# Both filters show an anomalously short (~3 day) cycle in the SAME Apr 11-16
# 2026 window, versus every other cycle running 16-68 days. Two independent
# filters coincidentally fouling-and-replacing within days of each other, in
# lockstep, is much less plausible than a shared plant event (turnaround,
# upset, maintenance activity on the shared circulation loop) that reset both
# DP readings without representing genuine independent fouling-to-replacement
# behavior. These are flagged (not silently dropped) via `is_anomalous_cycle`
# so training can exclude them while the full record stays inspectable.
# Keyed on BOTH short duration and window proximity -- start-date-only would
# also catch the long, perfectly normal 68-day cycles that happen to begin
# right after the anomalous window closes.
ANOMALOUS_CYCLE_WINDOW = (pd.Timestamp("2026-04-11"), pd.Timestamp("2026-04-16"))
ANOMALOUS_MAX_DURATION = pd.Timedelta(days=5)

rows = []
cycle_id = 0
cycle_summary = []

for filter_name, cfg in FILTER_CFG.items():
    dp_tag = cfg["dp_tag"]
    f_events = events[events["filter"] == filter_name].sort_values("replacement_time").reset_index(drop=True)
    if len(f_events) < 2:
        print(f"{filter_name}: only {len(f_events)} replacement(s) detected -- no complete cycle available, skipping training data for this filter")
        continue

    for i in range(len(f_events) - 1):
        cyc_start = f_events.loc[i, "replacement_time"]
        cyc_end = f_events.loc[i + 1, "replacement_time"]
        sub = df.loc[cyc_start:cyc_end].resample("1h").last()
        sub = sub.dropna(subset=[dp_tag])
        if len(sub) < 24:
            print(f"{filter_name} cycle {cyc_start:%Y-%m-%d} -> {cyc_end:%Y-%m-%d}: too short ({len(sub)}h), skipping")
            continue

        is_anomalous = (ANOMALOUS_CYCLE_WINDOW[0] <= cyc_start <= ANOMALOUS_CYCLE_WINDOW[1]
                       and (cyc_end - cyc_start) <= ANOMALOUS_MAX_DURATION)

        feat = engineer_features(sub, dp_tag)
        feat["RUL_hours"] = (cyc_end - feat.index).total_seconds() / 3600
        feat["filter_type"] = filter_name
        feat["cycle_id"] = cycle_id
        feat["cycle_start"] = cyc_start
        feat["cycle_end"] = cyc_end
        feat["is_anomalous_cycle"] = is_anomalous
        rows.append(feat)
        cycle_summary.append(dict(cycle_id=cycle_id, filter=filter_name, start=cyc_start, end=cyc_end,
                                  duration_days=(cyc_end - cyc_start).total_seconds() / 86400, n_hourly_rows=len(sub),
                                  is_anomalous_cycle=is_anomalous))
        if is_anomalous:
            print(f"  *** {filter_name} cycle {cyc_start:%Y-%m-%d %H:%M} -> {cyc_end:%Y-%m-%d %H:%M} "
                  f"flagged as anomalous (falls in the shared Apr11-16 event window) ***")
        cycle_id += 1

training_data = pd.concat(rows, ignore_index=False) if rows else pd.DataFrame()
training_data = training_data.dropna(subset=["dp_slope_24h"])  # first 24h of each cycle can't have a 24h slope yet

print(f"\nBuilt {len(training_data)} hourly training rows across {cycle_id} complete cycles")
print("\nCycle summary:")
cycle_summary_df = pd.DataFrame(cycle_summary)
print(cycle_summary_df.to_string())

# ---------------------------------------------------------------------------
# Live (open, right-censored) cycles -- for inference, not training
# ---------------------------------------------------------------------------
live_rows = []
for filter_name, cfg in FILTER_CFG.items():
    dp_tag = cfg["dp_tag"]
    f_events = events[events["filter"] == filter_name].sort_values("replacement_time")
    if len(f_events) == 0:
        continue
    last_replacement = f_events["replacement_time"].iloc[-1]
    sub = df.loc[last_replacement:].resample("1h").last()
    sub = sub.dropna(subset=[dp_tag])
    if len(sub) < 24:
        continue
    feat = engineer_features(sub, dp_tag)
    feat["filter_type"] = filter_name
    feat["cycle_start"] = last_replacement

    # Explicit status per the analyst's rule: threshold hit AND subsequently
    # fallen to baseline = replaced; threshold hit but not yet fallen back =
    # awaiting replacement; threshold never reached = normal operation.
    baseline = cfg["baseline"]
    hit_threshold = (feat["dp"] >= PEAK_THRESHOLD).cummax()  # True from the first threshold-crossing onward
    fell_to_baseline_after = hit_threshold & (feat["dp"] <= baseline)
    feat["threshold_hit"] = hit_threshold
    feat["replaced"] = fell_to_baseline_after.cummax()  # stays True once it happens
    current_status = ("Replaced (should not occur in an open cycle -- flag for review)" if feat["replaced"].iloc[-1]
                      else "Threshold exceeded, awaiting replacement" if feat["threshold_hit"].iloc[-1]
                      else "Normal")

    feat = feat.dropna(subset=["dp_slope_24h"])
    live_rows.append(feat)
    print(f"\nLive/open cycle - {filter_name}: started {last_replacement}, "
          f"{len(feat)} hourly rows so far, current DP = {feat['dp'].iloc[-1]:.2f} PSI, "
          f"status = {current_status}")

live_data = pd.concat(live_rows, ignore_index=False) if live_rows else pd.DataFrame()

training_data.to_pickle(DATA_DIR / "rul_training_data.pkl")
live_data.to_pickle(DATA_DIR / "rul_live_data.pkl")
cycle_summary_df.to_csv(EDA_DIR / "rul_cycle_summary.csv", index=False)
print(f"\nSaved {DATA_DIR / 'rul_training_data.pkl'} ({len(training_data)} rows, {cycle_id} cycles)")
print(f"Saved {DATA_DIR / 'rul_live_data.pkl'} ({len(live_data)} rows)")
