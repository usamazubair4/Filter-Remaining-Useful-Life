# Rich Amine Filter RUL -- live scoring pipeline

Turns the trained RUL model (`research/22_train_rul_models.py`) into a
scheduled Azure Function that pulls live plant data and produces a fresh
RUL forecast for both filters, on the same auth pattern as a reference
OmniConnect-style ingestion pipeline this was adapted from.

## Layout

```
pipeline/
  rul_model/          clean, Azure-agnostic model logic -- cycle detection,
                       feature engineering, scoring. Mirrors ../research/20-22
                       exactly; import this, don't reimplement it.
  rul_pipeline/        Azure-specific glue
    ingest.py            OAuth2 + OmniConnect Read API (the given auth flow)
    pipeline.py           orchestration: fetch -> detect cycle -> score -> sink
    sink.py                where predictions go (local file for now)
  function_app.py      Timer-trigger entrypoint (Azure Functions v2 model) --
                        thin wrapper around rul_pipeline.pipeline.run()
  Configurations.py    config template, secrets masked, env-var driven
  tags_config.csv       R_Tag_Name -> Tag_Name mapping for this model's 10 tags
  models/               model_bundle.joblib lives here at deploy time (gitignored --
                        populated by a build step, see "Model refresh" below)
  tests/                pytest suite, runs in CI
  azure-pipelines.yml   Azure DevOps CI/CD
```

## How live scoring differs from the reference pattern

The reference flow's last step is "average all rows -> single row -> 5
pickle models." That fits a stateless snapshot model. This model isn't one:
it needs a 7-day DP slope, a 24h rolling std, and a cumulative mean since
the *current cycle* started -- all rolling/expanding features, not a single
point-in-time reading. So `ingest.py` requests a `LOOKBACK_DAYS`-wide
window (9 days by default) and keeps every row, resampled hourly, instead
of collapsing to one average. Everything else -- the OAuth2 token flow, the
`Basic {credentials},Bearer {token}` header, the `IsSuccess`/`Data` check,
the `tags_config.csv` rename -- is unchanged.

**Flagged assumption:** the reference material specifies the auth flow and
the response envelope (`Result.Data`) but not the exact request body schema
for a date-ranged pull. `ingest.request_body()` is a reasonable guess
(`CustomerName`, `SiteName`, `Tags`, `StartDateTime`/`EndDateTime`) built
from the fields `Configurations.py` already carries -- **confirm the real
field names against OmniConnect ReadAPI's documentation/Swagger before
pointing this at production** and adjust that one function. Nothing else
in the pipeline depends on the exact request shape.

## Predictions sink

Per current direction: predictions write to `local_output/latest.json` (and
a timestamped file per run) while the solution is still being developed.
`rul_pipeline/sink.py` is the one place that changes when you're ready to
point this at Azure SQL or ADX for real -- every caller goes through
`write_predictions()`, so it's a one-function swap, not a search-and-replace.

## Local setup

```bash
cd pipeline
python -m venv .venv && .venv\Scripts\activate   # or source .venv/bin/activate
pip install -r requirements-dev.txt
copy local.settings.json.template local.settings.json   # fill in real values for local testing
pytest tests/ -v
```

Running the scoring pipeline directly (no Azure Functions runtime needed):

```bash
python -m rul_pipeline.pipeline
```

This will fail against the placeholder `xxx` config until `local.settings.json`
(or real environment variables) has real OmniConnect/Azure AD credentials --
that's expected; the test suite mocks the network calls so it doesn't need them.

To run with the Azure Functions Core Tools instead (exercises the Timer
Trigger binding itself):

```bash
func start
```

## Model refresh

`models/rul_model_bundle.joblib` is **not** committed (gitignored) --
it's a build artifact copied in from the repo-root `models/` directory that
`research/22_train_rul_models.py` writes to. After retraining:

```bash
cp ../models/rul_model_bundle.joblib models/rul_model_bundle.joblib
```

`azure-pipelines.yml`'s Deploy stage does this same copy before zipping, so
production always ships whatever the repo-root `models/` currently holds.
Re-pin `scikit-learn`/`numpy`/`pandas`/`joblib` in `requirements.txt` if the
training environment's versions change -- see the comment there for why.

## CI/CD setup (one-time, not in the repo)

`azure-pipelines.yml`'s header comment lists these; summarized:

1. An Azure Resource Manager service connection in Azure DevOps.
2. A Function App + resource group already provisioned in Azure (this
   pipeline deploys *code*, it doesn't provision infrastructure).
3. A variable group `rul-model-secrets` (ideally Key-Vault-linked) holding
   every secret `Configurations.py` reads -- `DB_PASSWORD`,
   `AAD_CLIENT_SECRET`, `OMNICONNECT_CREDENTIALS`, etc.
4. An Azure DevOps Environment named `production` with an approval check,
   so a deploy pauses for sign-off before it runs -- this redeploys a real
   scheduled job against live plant data, so it isn't a fire-and-forget CD.

## What's deliberately not decided yet

- **ADX wiring** -- `Configurations.py` carries the ADX/KQL fields for
  parity with the reference config, but nothing in the live path uses them
  yet (see "Predictions sink" above).
- **Azure SQL as the rolling-window source** -- `ingest.py` currently
  fetches history through OmniConnect's Read API. If OmniConnect can't
  actually serve a multi-day range efficiently, pulling the lookback window
  from `db_server` directly (the historian) via `pyodbc` is the fallback --
  the config fields are already in place, `pipeline.py`'s `score_one_filter()`
  doesn't care where `wide` came from.
- **F-509 vs F-511 identity** -- both physical filters share one DP tag
  (`DPIT-0509`); this pipeline scores "whichever filter is currently in
  service" as one pooled process, same as training. See
  the evidence report (`docs/evidence_report.html`), section 5.