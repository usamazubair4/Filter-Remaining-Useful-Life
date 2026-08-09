# -*- coding: utf-8 -*-
"""
Azure ingestion -- mirrors the reference pipeline's Auth Flow (Step 2, Live
Predictions) exactly for authentication, adapted in one deliberate place for
what this model actually needs.

Reference flow:
    1. POST token_endpoint -> Bearer token (OAuth2 client credentials)
    2. POST OmniConnect API, header: Authorization: Basic {credentials},Bearer {bearer_token}
    3. Response JSON -> Result.Data (list of tag rows)
    4. IsSuccess == 'false' or Data not a list -> no plant data, exit gracefully
    5. Rename R_Tag_Name -> Tag_Name via tags_config.csv
    6. Average all rows -> single row -> 5 pickle models -> insert predictions

Where this differs, and why: step 6 assumes a stateless snapshot model --
one row in, one row out. The RUL model needs rolling history (a 7-day DP
slope, a 24h std dev, a cumulative mean since the current cycle started), so
instead of collapsing the OmniConnect response to one averaged row, this
requests a LOOKBACK_DAYS-wide time window and keeps every row, resampled to
hourly. Steps 1-2 and 4-5 are otherwise unchanged.

ASSUMPTION FLAGGED: the reference material specifies the auth flow and the
response shape (Result.Data) but not the exact request body schema for a
date-ranged pull. request_body() below is a reasonable guess (customer_name,
site_name, a tag list, start/end datetime) consistent with the fields
Configurations.py already carries -- confirm the real field names against
OmniConnect ReadAPI's documentation/Swagger before pointing this at
production, and adjust request_body() accordingly. Everything downstream
(parsing, mapping, resampling) is independent of that fix.
"""
import base64
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger("rul_pipeline.ingest")


class NoPlantDataError(Exception):
    """Raised when OmniConnect reports IsSuccess == false or returns no
    tag rows -- callers should catch this and exit the run gracefully
    (step 4 of the reference flow), not treat it as a hard failure."""


def get_bearer_token(cfg) -> str:
    """OAuth2 client credentials grant against Azure AD (step 1)."""
    resp = requests.post(
        cfg.token_endpoint,
        data={
            "grant_type": "client_credentials",
            "client_id": cfg.client_id,
            "client_secret": cfg.client_secret,
            "scope": cfg.scope,
        },
        timeout=30,
    )
    resp.raise_for_status()
    token = resp.json().get("access_token")
    if not token:
        raise RuntimeError("Token endpoint responded 200 but no access_token in body")
    return token


def request_body(cfg, tags: list, start_dt: datetime, end_dt: datetime) -> dict:
    """See the ASSUMPTION FLAGGED note above -- verify field names against
    the real OmniConnect ReadAPI contract."""
    return {
        "CustomerName": cfg.customer_name,
        "SiteName": cfg.site_name,
        "Tags": tags,
        "StartDateTime": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "EndDateTime": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
    }


def fetch_tag_rows(cfg, tags: list, start_dt: datetime, end_dt: datetime,
                    bearer_token: str) -> list:
    """Steps 2-4: call OmniConnect, validate the envelope, return the raw
    list of tag rows. Raises NoPlantDataError if there's genuinely nothing
    to score this run (matches the reference's "exit gracefully" behavior
    rather than raising on an empty-but-valid response)."""
    headers = {
        "Authorization": f"Basic {cfg.credentials},Bearer {bearer_token}",
        "Content-Type": "application/json",
    }
    resp = requests.post(cfg.prod_azure_link, json=request_body(cfg, tags, start_dt, end_dt),
                          headers=headers, timeout=60)
    resp.raise_for_status()
    body = resp.json()

    is_success = str(body.get("IsSuccess", body.get("Result", {}).get("IsSuccess", ""))).lower()
    data = body.get("Result", {}).get("Data", body.get("Data"))

    if is_success == "false" or not isinstance(data, list):
        logger.info("OmniConnect returned no plant data for this window (IsSuccess=%s) -- skipping this run", is_success)
        raise NoPlantDataError(f"IsSuccess={is_success!r}, Data type={type(data).__name__}")

    if len(data) == 0:
        raise NoPlantDataError("Data was an empty list")

    return data


def load_tag_mapping(tags_config_path: str) -> dict:
    """R_Tag_Name -> Tag_Name, from tags_config.csv (step 5)."""
    mapping_df = pd.read_csv(tags_config_path)
    return dict(zip(mapping_df["R_Tag_Name"], mapping_df["Tag_Name"]))


def rows_to_hourly_wide(rows: list, tags_config_path: str,
                         tag_col: str = "TagName", ts_col: str = "Timestamp", val_col: str = "Value") -> pd.DataFrame:
    """Long rows (one per tag reading) -> hourly-resampled wide frame
    (Timestamp index, one column per Tag_Name), matching the shape
    scripts/21's engineer_features() expects. Unlike the reference's
    "average all rows into one row," every row is kept -- see module
    docstring for why.
    """
    df = pd.DataFrame(rows)
    if tag_col not in df.columns or ts_col not in df.columns or val_col not in df.columns:
        raise ValueError(f"Unexpected OmniConnect row shape: columns={list(df.columns)} "
                          f"(expected {tag_col!r}, {ts_col!r}, {val_col!r} -- adjust to match the real response)")

    mapping = load_tag_mapping(tags_config_path)
    df["Tag_Name"] = df[tag_col].map(mapping)
    unmapped = df[df["Tag_Name"].isna()][tag_col].unique()
    if len(unmapped):
        logger.warning("%d raw tag name(s) had no entry in tags_config.csv and were dropped: %s",
                        len(unmapped), list(unmapped))
    df = df.dropna(subset=["Tag_Name"])

    df[ts_col] = pd.to_datetime(df[ts_col])
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")

    wide = df.pivot_table(index=ts_col, columns="Tag_Name", values=val_col, aggfunc="mean")
    wide = wide.sort_index().resample("1h").mean()
    return wide


def fetch_lookback_window(cfg, tags: list, lookback_days: int,
                           as_of: Optional[datetime] = None) -> pd.DataFrame:
    """End-to-end: token -> OmniConnect -> hourly wide frame covering the
    last `lookback_days` up to `as_of` (defaults to now, UTC)."""
    as_of = as_of or datetime.now(timezone.utc)
    start_dt = as_of - timedelta(days=lookback_days)

    token = get_bearer_token(cfg)
    rows = fetch_tag_rows(cfg, tags, start_dt, as_of, token)
    return rows_to_hourly_wide(rows, cfg.tags_config_path)