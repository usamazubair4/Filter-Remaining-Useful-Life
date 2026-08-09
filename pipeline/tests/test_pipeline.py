from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from rul_model.config import FEATURE_TAGS, FILTER_CFG
from rul_pipeline.ingest import NoPlantDataError
from rul_pipeline.pipeline import score_one_filter


def _synthetic_wide(n_hours=24 * 25, start="2026-06-23"):
    idx = pd.date_range(start, periods=n_hours, freq="1h")
    rng = np.random.default_rng(2)
    dp510 = np.linspace(1.5, 3.5, n_hours) + rng.normal(0, 0.05, n_hours)
    dp509 = np.linspace(1.0, 2.5, n_hours) + rng.normal(0, 0.05, n_hours)
    data = {
        "PLANT_AMINE_P_DPIT_0510_VAL": dp510,
        "PLANT_AMINE_P_DPIT_0509_VAL": dp509,
    }
    for tag in FEATURE_TAGS:
        data.setdefault(tag, 50 + rng.normal(0, 1, n_hours))
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def bundle():
    from rul_model import load_bundle
    from pathlib import Path
    # azure_pipeline/models/ first (deployment-correct location), falling
    # back to the repo-root models/ scripts/22 writes to directly -- so
    # tests pass right after a fresh `python scripts/22_train_rul_models.py`
    # even before the local copy-into-package step has been run.
    candidates = [
        Path(__file__).resolve().parent.parent / "models" / "rul_model_bundle.joblib",
        Path(__file__).resolve().parent.parent.parent / "models" / "rul_model_bundle.joblib",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        pytest.skip("rul_model_bundle.joblib not built yet -- run scripts/22_train_rul_models.py")
    return load_bundle(path)


def test_score_one_filter_returns_ordered_quantiles_and_window(bundle):
    wide = _synthetic_wide()
    as_of = datetime.now(timezone.utc)
    result = score_one_filter("F-510 Carbon Filter", FILTER_CFG["F-510 Carbon Filter"], wide, bundle, as_of)

    assert result["filter"] == "F-510 Carbon Filter"
    assert result["rul_p10_days"] <= result["rul_median_days"] <= result["rul_p90_days"]
    assert result["operational_status"] in (
        "Normal - in service", "Threshold exceeded, awaiting replacement", "Replaced (flag for review)",
    )
    # predicted dates should be strictly increasing with the quantiles
    assert result["predicted_replacement_p10"] <= result["predicted_replacement_median"] <= result["predicted_replacement_p90"]


def test_score_one_filter_raises_no_plant_data_when_dp_tag_missing(bundle):
    wide = _synthetic_wide().drop(columns=["PLANT_AMINE_P_DPIT_0510_VAL"])
    with pytest.raises(NoPlantDataError):
        score_one_filter("F-510 Carbon Filter", FILTER_CFG["F-510 Carbon Filter"], wide, bundle,
                          datetime.now(timezone.utc))


def test_score_one_filter_raises_no_plant_data_when_too_little_history(bundle):
    wide = _synthetic_wide(n_hours=5)
    with pytest.raises(NoPlantDataError):
        score_one_filter("F-510 Carbon Filter", FILTER_CFG["F-510 Carbon Filter"], wide, bundle,
                          datetime.now(timezone.utc))