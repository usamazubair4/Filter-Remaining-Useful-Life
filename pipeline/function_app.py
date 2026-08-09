# -*- coding: utf-8 -*-
"""
Azure Function App entrypoint (Python v2 programming model). Deployed by
azure-pipelines.yml's CD stage as a zip deploy to a Function App with a
Consumption or Premium plan.

This file is intentionally thin -- all real logic lives in rul_pipeline/
and rul_model/, both importable and unit-testable without the Azure
Functions runtime. Keep it that way; if you find yourself writing
pandas/sklearn code here, it belongs in pipeline.py instead.
"""
import logging

import azure.functions as func

from rul_pipeline.pipeline import run

app = func.FunctionApp()

# Every hour, at minute 5 (a few minutes past the hour so the historian has
# had time to settle that hour's readings). CRON is standard 6-field Azure
# Functions NCRONTAB: {second} {minute} {hour} {day} {month} {day-of-week}.
# Override via the Function App Setting "RulScoreScheduleCron" without a
# redeploy if the cadence needs to change.
@app.timer_trigger(schedule="%RulScoreScheduleCron%", arg_name="timer",
                    run_on_startup=False, use_monitor=True)
def rul_score(timer: func.TimerRequest) -> None:
    logging.info("RUL scoring run starting (past due: %s)", timer.past_due)
    try:
        results = run()
        logging.info("RUL scoring run complete -- %d filter(s) scored", len(results))
    except Exception:
        logging.exception("RUL scoring run failed")
        raise