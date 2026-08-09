# -*- coding: utf-8 -*-
"""
Feature engineering -- identical transforms to scripts/21_build_rul_dataset.py,
factored out so the research pipeline and live scoring compute every feature
exactly one way.
"""
import pandas as pd

from .config import FEATURE_TAGS, FIT_2512_TAG, TIT_3507_TAG


def engineer_features(sub: pd.DataFrame, dp_tag: str) -> pd.DataFrame:
    """sub: hourly-resampled data for ONE filter, indexed by timestamp,
    already restricted to the current cycle (index[0] == cycle start).

    This restriction matters and is not just tidiness: every rolling/
    expanding feature below (flow_cummean_this_cycle, dp_std_24h,
    dp_slope_7d, ...) is intentionally cycle-scoped -- it must not see data
    from the previous cycle, because that's how scripts/21 built the
    training data (sub was already sliced to one cycle before this function
    ran). Passing in a longer lookback window here, unsliced, would compute
    different feature values than the model was trained on for any cycle
    younger than 7 days -- a silent train/serve mismatch. Callers fetch a
    longer window only to locate the cycle boundary (see cycles.py), then
    slice to cycle_start before calling this.
    """
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
    if FIT_2512_TAG in sub.columns:
        feat["flow_cummean_this_cycle"] = sub[FIT_2512_TAG].expanding(min_periods=1).mean()
        flow_safe = sub[FIT_2512_TAG].clip(lower=1.0)  # avoid divide-by-near-zero blowups
        feat["specific_resistance"] = dp / flow_safe
        feat["specific_resistance_slope_24h"] = feat["specific_resistance"].diff(24) / 24
    if TIT_3507_TAG in sub.columns:
        feat["tit3507_cummean_this_cycle"] = sub[TIT_3507_TAG].expanding(min_periods=1).mean()
        feat["tit3507_slope_24h"] = sub[TIT_3507_TAG].diff(24) / 24
    return feat