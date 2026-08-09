# -*- coding: utf-8 -*-
"""
Shared configuration for the Rich Amine Filter RUL model.

This is the single source of truth for detection threshold, filter tags, and
the feature list -- both the research pipeline (scripts/20-22) and this live
scoring package must agree on these exactly, since a mismatch here silently
produces features the trained model was never shown. If you change the
threshold or feature list in scripts/21 or scripts/22, mirror the change
here (and retrain -- see scripts/22_train_rul_models.py).
"""

PEAK_THRESHOLD = 9.0  # PSI -- DP reaching this counts as "due for replacement"

FILTER_CFG = {
    "F-510 Carbon Filter": dict(
        dp_tag="PLANT_AMINE_P_DPIT_0510_VAL",
        he_dp_tag="PLANT_AMINE_P_DPIT_7502_VAL",
        baseline=2.0,
        is_carbon=True,
    ),
    "F-509/F-511 Sock Filter": dict(
        dp_tag="PLANT_AMINE_P_DPIT_0509_VAL",
        he_dp_tag="PLANT_AMINE_P_DPIT_6502_VAL",
        baseline=1.5,
        is_carbon=False,
    ),
}

FIT_2512_TAG = "PLANT_AMINE_P_FIT_2512_VAL"
TIT_3507_TAG = "PLANT_AMINE_P_TIT_3507_VAL"

# Every raw tag engineer_features() reads besides the two DP tags above.
FEATURE_TAGS = [
    "PLANT_AMINE_P_DPIT_7502_VAL", "PLANT_AMINE_P_DPIT_6502_VAL",
    FIT_2512_TAG, "PLANT_AMINE_CIRC514VFD21_RPM_FB",
    "PLANT_AMINE_CIRC515VFD22_RPM_FB", "PLANT_AMINE_P_TIT_1504_VAL",
    "PLANT_AMINE_P_PIT_6503_VAL", TIT_3507_TAG,
]

# Every column the trained models expect, in the order build_matrix() builds
# them (order doesn't matter to sklearn since columns are named, but keeping
# one canonical list avoids two copies drifting apart).
FEATURE_COLS = [
    "dp", "dp_slope_6h", "dp_slope_24h", "dp_slope_7d", "dp_std_24h", "hours_since_start",
    "specific_resistance", "specific_resistance_slope_24h",
    "PLANT_AMINE_P_DPIT_7502_VAL", "PLANT_AMINE_P_DPIT_6502_VAL",
    FIT_2512_TAG, "flow_cummean_this_cycle",
    "PLANT_AMINE_CIRC514VFD21_RPM_FB",
    "PLANT_AMINE_CIRC515VFD22_RPM_FB", "PLANT_AMINE_P_TIT_1504_VAL",
    "PLANT_AMINE_P_PIT_6503_VAL",
    TIT_3507_TAG, "tit3507_cummean_this_cycle", "tit3507_slope_24h",
]
TREND_ONLY_COLS = [c for c in FEATURE_COLS if c != "hours_since_start"]

# The longest rolling feature is the 7-day DP slope; scoring must fetch at
# least this much history before the current cycle's start to ever produce a
# non-null value for it. A couple of days of headroom absorbs historian gaps.
LOOKBACK_DAYS = 9

# All raw tags a live pull needs: the two DP tags plus every FEATURE_TAGS
# entry, de-duplicated, in a stable order -- this is what tags_config.csv
# maps R_Tag_Name -> Tag_Name for.
ALL_LIVE_TAGS = sorted({cfg["dp_tag"] for cfg in FILTER_CFG.values()} | set(FEATURE_TAGS))