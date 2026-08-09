# -*- coding: utf-8 -*-
"""
Predictions sink -- where the scored output goes.

Per the current decision: write locally while the solution is still being
developed; swap in a real sink (Azure SQL table, ADX) once ready to deploy
for real. write_predictions() is the one function every caller (function_app.py,
pipeline.run(), tests) goes through, so that swap is a one-function change,
not a search-and-replace.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

logger = logging.getLogger("rul_pipeline.sink")

DEFAULT_LOCAL_DIR = Path(__file__).resolve().parent.parent / "local_output"


def write_predictions(results: List[dict], output_dir: Path = DEFAULT_LOCAL_DIR) -> Path:
    """Writes one timestamped JSON file per run (append-friendly for local
    inspection) plus overwrites latest.json for "what does it say right
    now" checks.

    TODO when moving off local storage: replace the body of this function
    with an INSERT into dbo.RUL_Predictions (Azure SQL) and/or an ADX
    ingestion call, using db_server/db_user/db_password/database_name (or
    cluster/kql_client_id/...) from Configurations.py. Keep the function
    signature the same so function_app.py and pipeline.py don't change.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now(timezone.utc)
    payload = {"run_at_utc": run_ts.isoformat(), "predictions": results}

    stamped_path = output_dir / f"rul_predictions_{run_ts.strftime('%Y%m%dT%H%M%SZ')}.json"
    stamped_path.write_text(json.dumps(payload, indent=2, default=str))

    latest_path = output_dir / "latest.json"
    latest_path.write_text(json.dumps(payload, indent=2, default=str))

    logger.info("Wrote %d prediction(s) to %s", len(results), stamped_path)
    return stamped_path