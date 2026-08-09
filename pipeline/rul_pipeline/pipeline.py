# -*- coding: utf-8 -*-
"""
Core scoring pipeline -- no azure.functions dependency, so it runs the same
way under the Timer Trigger, from a local shell, and from tests. function_app.py
is a thin wrapper that calls run() on a schedule.
"""
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `import Configurations` resolves
import Configurations as cfg  # noqa: E402

from rul_model import (  # noqa: E402
    ALL_LIVE_TAGS,
    FILTER_CFG,
    LOOKBACK_DAYS,
    current_cycle_start,
    engineer_features,
    load_bundle,
    operational_status,
    score_latest_row,
)
from rul_pipeline.ingest import NoPlantDataError, fetch_lookback_window  # noqa: E402
from rul_pipeline.sink import write_predictions  # noqa: E402

logger = logging.getLogger("rul_pipeline.pipeline")

# Deployed Function App packages are the *contents* of azure_pipeline/ zipped
# at the root -- there's no repo root inside the package, so the model
# bundle must live inside azure_pipeline/models/, not at the repo-root
# models/ that scripts/22_train_rul_models.py writes to. The build step in
# azure-pipelines.yml copies repo-root models/ -> azure_pipeline/models/
# before zipping; do the same locally after retraining (see README.md).
# MODEL_BUNDLE_PATH overrides this for any other layout without a redeploy.
MODEL_BUNDLE_PATH = Path(os.environ.get(
    "MODEL_BUNDLE_PATH",
    str(Path(__file__).resolve().parent.parent / "models" / "rul_model_bundle.joblib"),
))


def score_one_filter(filter_name: str, filter_cfg: dict, wide: pd.DataFrame, bundle: dict, as_of: datetime) -> dict:
    """wide: hourly-resampled frame covering the last LOOKBACK_DAYS, all
    filters' tags. Locates the current cycle, restricts to it (see
    rul_model.features docstring for why that restriction is load-bearing),
    scores the latest row, and returns one result dict."""
    dp_tag = filter_cfg["dp_tag"]
    baseline = filter_cfg["baseline"]

    if dp_tag not in wide.columns:
        raise NoPlantDataError(f"{dp_tag} missing from the fetched window -- check tags_config.csv mapping")

    dp_series = wide[dp_tag].dropna()
    if dp_series.empty:
        raise NoPlantDataError(f"{dp_tag} had no readings in the last {LOOKBACK_DAYS} days")

    cycle_start = current_cycle_start(dp_series, baseline)
    sub = wide.loc[cycle_start:].dropna(subset=[dp_tag])
    if len(sub) < 24:
        raise NoPlantDataError(f"{filter_name}: only {len(sub)}h of data since detected cycle start "
                                f"{cycle_start} -- too little to score yet")

    feat = engineer_features(sub, dp_tag)
    latest = feat.tail(1)
    status = operational_status(sub[dp_tag], baseline)

    scored = score_latest_row(latest, bundle, is_carbon=filter_cfg["is_carbon"])
    cycle_age_days = float(latest["hours_since_start"].iloc[0]) / 24
    max_seen_days = bundle["max_train_age_by_filter"].get(filter_name)
    out_of_range = max_seen_days is not None and cycle_age_days > max_seen_days

    return dict(
        filter=filter_name,
        as_of=as_of.isoformat(),
        cycle_start=cycle_start.isoformat(),
        cycle_age_days=round(cycle_age_days, 1),
        current_dp_psi=round(float(latest["dp"].iloc[0]), 3),
        operational_status=status,
        max_training_cycle_age_days_seen=max_seen_days,
        out_of_training_range=bool(out_of_range),
        **scored,
        predicted_replacement_p10=(as_of + pd.Timedelta(hours=scored["rul_p10_hours"])).isoformat(),
        predicted_replacement_median=(as_of + pd.Timedelta(hours=scored["rul_median_hours"])).isoformat(),
        predicted_replacement_p90=(as_of + pd.Timedelta(hours=scored["rul_p90_hours"])).isoformat(),
    )


def run() -> list:
    """One full scoring pass for both filters. Returns the list of result
    dicts that was written to the sink (empty list if there was genuinely
    no plant data this run -- not an error, matches the reference flow's
    "exit gracefully")."""
    as_of = datetime.now(timezone.utc)
    bundle = load_bundle(MODEL_BUNDLE_PATH)

    try:
        wide = fetch_lookback_window(cfg, ALL_LIVE_TAGS, LOOKBACK_DAYS, as_of=as_of)
    except NoPlantDataError as e:
        logger.info("No plant data this run (%s) -- skipping, will retry next schedule", e)
        return []

    results = []
    for filter_name, filter_cfg in FILTER_CFG.items():
        try:
            results.append(score_one_filter(filter_name, filter_cfg, wide, bundle, as_of))
        except NoPlantDataError as e:
            logger.warning("Skipping %s this run: %s", filter_name, e)

    if results:
        write_predictions(results)
    else:
        logger.info("No filter had enough data to score this run")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = run()
    for r in out:
        print(f"{r['filter']}: {r['operational_status']} -- "
              f"P10/median/P90 = {r['rul_p10_days']}/{r['rul_median_days']}/{r['rul_p90_days']} days")