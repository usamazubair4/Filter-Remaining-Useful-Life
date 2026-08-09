# -*- coding: utf-8 -*-
"""
Configurations.py -- Template (secrets masked)

Mirrors the shape of the reference ingestion pipeline's Configurations.py so
this deploys the same way, with one deliberate change: every secret is read
from an environment variable first and only falls back to the masked 'xxx'
placeholder if that variable isn't set. Nothing here should ever be a real
secret committed to source control -- in Azure Function App Settings (set
via the CD stage in azure-pipelines.yml, sourced from an Azure DevOps
variable group linked to Key Vault) these environment variables are the real
values; locally, copy azure_pipeline/local.settings.json.template to
local.settings.json and fill it in for testing (see README.md).
"""
import os
from datetime import datetime


def _env(name: str, default: str = "xxx") -> str:
    return os.environ.get(name, default)


# ---------------------------------------------------------------------------
# Azure SQL Database -- long-term historian store. Used to pull the rolling
# lookback window (LOOKBACK_DAYS) needed for feature engineering at score
# time; also what scripts/01-08 would read from if the training pipeline
# moves off manual Excel exports onto this DB directly.
# ---------------------------------------------------------------------------
db_server     = _env("DB_SERVER", "xxx.database.windows.net")
db_user       = _env("DB_USER")
db_password   = _env("DB_PASSWORD")
database_name = _env("DB_NAME")

# ---------------------------------------------------------------------------
# Azure App Registration (for OmniConnect Bearer token) -- OAuth2 client
# credentials flow, identical to the reference pipeline.
# ---------------------------------------------------------------------------
client_id      = _env("AAD_CLIENT_ID")
client_secret  = _env("AAD_CLIENT_SECRET")
scope          = _env("AAD_SCOPE", "api://xxx/.default")
token_endpoint = _env("AAD_TOKEN_ENDPOINT", "https://login.microsoftonline.com/xxx/oauth2/v2.0/token")

# ---------------------------------------------------------------------------
# OmniConnect Read API -- live/near-real-time tag reads for scoring.
# ---------------------------------------------------------------------------
customer_name   = _env("OMNICONNECT_CUSTOMER", "the plant")
site_name       = _env("OMNICONNECT_SITE", "Site-01")
credentials     = _env("OMNICONNECT_CREDENTIALS")          # base64 "username:password"
prod_azure_link = _env("OMNICONNECT_URL", "https://omniconnect-readapi.azurewebsites.net/api/ReadAPI?code=xxx")

# ---------------------------------------------------------------------------
# Azure Data Explorer (ADX/KQL) -- not wired into the live scoring path yet.
# Kept here, unused, because the reference Configurations.py carries it and
# a downstream dashboard may read from ADX later; see README "Predictions
# sink" section for the current (local-file) output path.
# ---------------------------------------------------------------------------
cluster           = _env("ADX_CLUSTER", "https://xxx.southeastasia.kusto.windows.net/")
kql_client_id     = _env("ADX_CLIENT_ID")
kql_client_secret = _env("ADX_CLIENT_SECRET")
authority_id      = _env("ADX_AUTHORITY_ID")
adx_db_name       = _env("ADX_DB_NAME")

# ---------------------------------------------------------------------------
# Rich Amine Filter tags -- this model's inputs (see tags_config.csv for the
# full R_Tag_Name -> Tag_Name mapping used by rul_pipeline/ingest.py).
# ---------------------------------------------------------------------------
tags_config_path = _env("TAGS_CONFIG_PATH", str(__file__).replace("Configurations.py", "tags_config.csv"))

# ---------------------------------------------------------------------------
# Scoring window -- how far back each run pulls to compute rolling features.
# Kept here (rather than only in rul_model/config.py) so it can be widened
# via an App Setting without a redeploy if the historian ever has gaps.
# ---------------------------------------------------------------------------
lookback_days = int(_env("LOOKBACK_DAYS", "9"))

# ---------------------------------------------------------------------------
# Training date range -- unused by live scoring; kept for parity with the
# reference Configurations.py and for any future job that retrains from
# this same config file instead of scripts/21's hardcoded range.
# ---------------------------------------------------------------------------
start_datetime = datetime(2025, 9, 1, 0, 0, 0)
end_datetime   = datetime(2026, 7, 14, 23, 59, 59)