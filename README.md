# Rich Amine Filter — Remaining Useful Life (RUL) Dashboard

**[Live dashboard →](https://usamazubair4.github.io/Filter-Remaining-Useful-Life/)**

An end-to-end predictive-maintenance case study for a gas-plant amine treating
unit: a rich amine filter fouls over days-to-weeks, tracked by differential
pressure (ΔP), and gets replaced when it crosses a hard threshold. This
project builds a model that forecasts *when that will happen* — a
low/typical/high estimate, not just today's reading against an alarm limit —
and a live dashboard to act on it.

> Every number on the dashboard is real model output (real ΔP history, real
> forecast accuracy, real feature correlations). The plant/operator identity
> has been generalized for this public write-up; the technical work is not.

## What's here

| | |
|---|---|
| `index.html` | The live dashboard — self-contained, no build step, no dependencies beyond two Google Fonts |
| `docs/evidence_report.html` | Full statistical evidence: correlation between every model input and actual fouling rate, leave-one-cycle-out validation results, feature importance |
| `docs/training_logic.html` | Two diagrams explaining the training/validation mechanism itself |
| `research/` | The actual research pipeline, in order: `20_detect_replacement_events.py` (label replacement cycles from raw ΔP), `21_build_rul_dataset.py` (feature engineering), `22_train_rul_models.py` (quantile regression + leave-one-cycle-out validation) |
| `pipeline/` | Production scoring service: an Azure Function that pulls live sensor data and scores the trained model on a schedule — `rul_model/` is the clean, reusable core logic; `rul_pipeline/` is the Azure-specific glue (auth, ingestion, output); `tests/` is the pytest suite that runs in CI; `azure-pipelines.yml` is the CI/CD definition |

## The problem

Two filters (a carbon filter and a sock filter) protect downstream
equipment from carryover in the rich amine stream. Operators currently
watch ΔP against a fixed threshold — useful, but reactive: it tells you
the filter is *near* end-of-life, not *how much life is left* or *when to
plan the changeout*.

## The approach

1. **Event detection** — label every historical replacement from raw ΔP:
   a peak above threshold followed by a fall back to baseline. 17 events
   found across ~10 months of history.
2. **Feature engineering** — cycle-relative rolling features (24h volatility,
   7-day trend), cumulative means since the current cycle started, and a
   flow-normalized "specific resistance" that isolates true fouling from
   flow-rate noise.
3. **Evidence before modeling** — every candidate input is checked by
   direct correlation against actual cycle length *before* it's trusted as
   a feature, not just accepted on the model's internal ranking. This is
   what caught flash/inlet temperature as the single strongest driver
   (r=0.77) — bigger than flow, which was the original hypothesis.
4. **Quantile regression** — gradient-boosted models trained on P10/median/P90
   of `log(remaining_life)`, so the forecast is a range with a stated
   confidence band, not a false-precision single number.
5. **Honest validation** — leave-one-cycle-out: every reported accuracy
   number comes from a cycle the model never trained on.

## Results

| Metric | Before adding temperature | After |
|---|---|---|
| Mean forecast error (LOCO) | 6.34 days | **4.47 days** |
| Median forecast error | 5.29 days | **3.21 days** |
| P10–P90 coverage | 46.6% | 52.7% (target 80% — reported honestly, not rounded up) |

Full breakdown, correlation tables, and per-cycle validation results in
[`docs/evidence_report.html`](docs/evidence_report.html).

## Stack

- **Modeling**: Python, pandas, scikit-learn (gradient-boosted quantile
  regression), joblib
- **Validation**: leave-one-cycle-out cross-validation (not random split —
  adjacent hourly rows are autocorrelated, a random split would be
  optimistic)
- **Live scoring pipeline**: Azure Functions (Timer Trigger), OAuth2 client-
  credentials auth, Azure DevOps CI/CD
- **Dashboard**: hand-built HTML/SVG (no charting library) — every chart
  (DP trend + forecast band, daily ΔP & fouling-rate comparison) is inline
  SVG driven by real embedded data
- **Companion semantic model**: Power BI (TMDL), 11 tables / 39 DAX measures,
  built for a live operational deployment of the same model

## Honesty notes (kept intentionally, not scrubbed for polish)

- **Confidence bands are not yet well-calibrated.** 52.7% coverage against
  an 80% target, with only 12 validated cycles. The dashboard shows this
  number rather than hiding it.
- **Two of the three filter tags share one sensor.** The model pools them
  rather than pretending it can distinguish what isn't instrumented.
- **A confounding check is documented, not glossed over**: the strongest
  driver (temperature) also drifted with season across the data window —
  the report states plainly that more data across a second summer/winter
  is needed to separate a causal effect from a seasonal proxy.

---

*Portfolio build — plant/company identity generalized; methodology,
code, and all numeric results are real.*
