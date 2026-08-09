from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from rul_pipeline.ingest import NoPlantDataError, fetch_tag_rows, get_bearer_token, rows_to_hourly_wide

FAKE_CFG = SimpleNamespace(
    token_endpoint="https://login.microsoftonline.com/tenant/oauth2/v2.0/token",
    client_id="id", client_secret="secret", scope="scope",
    prod_azure_link="https://omniconnect-readapi.azurewebsites.net/api/ReadAPI?code=xxx",
    credentials="dXNlcjpwYXNz", customer_name="Acme", site_name="Site-01",
)


def _resp(json_body, status=200):
    m = MagicMock()
    m.json.return_value = json_body
    m.raise_for_status = MagicMock()
    m.status_code = status
    return m


@patch("rul_pipeline.ingest.requests.post")
def test_get_bearer_token_returns_access_token(mock_post):
    mock_post.return_value = _resp({"access_token": "abc123", "expires_in": 3600})
    token = get_bearer_token(FAKE_CFG)
    assert token == "abc123"
    called_url = mock_post.call_args[0][0]
    assert called_url == FAKE_CFG.token_endpoint


@patch("rul_pipeline.ingest.requests.post")
def test_get_bearer_token_raises_if_no_access_token_in_body(mock_post):
    mock_post.return_value = _resp({"error": "invalid_client"})
    with pytest.raises(RuntimeError):
        get_bearer_token(FAKE_CFG)


@patch("rul_pipeline.ingest.requests.post")
def test_fetch_tag_rows_raises_no_plant_data_when_is_success_false(mock_post):
    mock_post.return_value = _resp({"IsSuccess": "false", "Result": {"Data": None}})
    with pytest.raises(NoPlantDataError):
        fetch_tag_rows(FAKE_CFG, ["tag1"], pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"), "token")


@patch("rul_pipeline.ingest.requests.post")
def test_fetch_tag_rows_returns_data_list_on_success(mock_post):
    rows = [{"TagName": "T1", "Timestamp": "2026-01-01T00:00:00", "Value": 1.23}]
    mock_post.return_value = _resp({"Result": {"IsSuccess": "true", "Data": rows}})
    out = fetch_tag_rows(FAKE_CFG, ["T1"], pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"), "token")
    assert out == rows

    # Authorization header matches the reference flow's "Basic {c},Bearer {t}" shape
    headers = mock_post.call_args.kwargs["headers"]
    assert headers["Authorization"] == f"Basic {FAKE_CFG.credentials},Bearer token"


def test_rows_to_hourly_wide_applies_tag_mapping_and_pivots(tmp_path):
    tags_csv = tmp_path / "tags_config.csv"
    tags_csv.write_text("R_Tag_Name,Tag_Name\nRAW.DP.510,PLANT_AMINE_P_DPIT_0510_VAL\n")

    rows = [
        {"TagName": "RAW.DP.510", "Timestamp": "2026-01-01T00:01:00", "Value": 1.5},
        {"TagName": "RAW.DP.510", "Timestamp": "2026-01-01T00:45:00", "Value": 1.7},
        {"TagName": "RAW.DP.510", "Timestamp": "2026-01-01T01:10:00", "Value": 1.9},
    ]
    wide = rows_to_hourly_wide(rows, str(tags_csv))
    assert "PLANT_AMINE_P_DPIT_0510_VAL" in wide.columns
    assert "RAW.DP.510" not in wide.columns
    # first two readings both fall in the 00:00 hourly bucket and are averaged
    assert wide["PLANT_AMINE_P_DPIT_0510_VAL"].iloc[0] == pytest.approx(1.6)


def test_rows_to_hourly_wide_drops_unmapped_tags(tmp_path):
    tags_csv = tmp_path / "tags_config.csv"
    tags_csv.write_text("R_Tag_Name,Tag_Name\nRAW.DP.510,PLANT_AMINE_P_DPIT_0510_VAL\n")
    rows = [{"TagName": "RAW.UNKNOWN.TAG", "Timestamp": "2026-01-01T00:01:00", "Value": 5.0}]
    wide = rows_to_hourly_wide(rows, str(tags_csv))
    assert wide.empty or "RAW.UNKNOWN.TAG" not in wide.columns