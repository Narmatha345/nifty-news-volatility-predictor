# API reference

Base URL: `http://127.0.0.1:8000/api` (interactive docs at `/docs`). All timestamps are UTC ISO-8601.

| Method | Path | Purpose |
|---|---|---|
| GET | `/news?ticker=&start=&end=&limit=` | Stored relevant news with per-entity analysis (event type, `price_report`), `timestamp_precision`, `availability_status` |
| POST | `/news/collect` `{start?, end?}` | Collect from all enabled providers (+ analyze). With start/end: historical backfill |
| POST | `/news/analyze` | No body: analyze pending articles. `{title, summary?}`: ad-hoc analysis (not stored) |
| GET | `/news/fetch-log` | Provider call log (status, errors) |
| GET | `/news/stats` | Article / analysis counts, availability-status and timestamp-precision counts |
| GET | `/sentiment` | Latest System 1 output, all horizons; company rows include `relative_sentiment`, `sentiment_zscore`, `new_news_count` |
| GET | `/sentiment/history?entity=NIFTY&horizon_type=today` | Daily score series (entity = ticker, MACRO, NIFTY, NIFTY_COMPANY) |
| POST | `/sentiment/aggregate?as_of=` | Compute and store System 1 scores now (or at `as_of`) |
| POST | `/predictions` `{as_of?, validate_with_llm?}` | Run System 1 + 2 and store a prediction run |
| GET | `/predictions?ticker=&horizon_type=&target_date=&run_id=` | Predictions (latest live run by default) |
| GET | `/predictions/runs` | Live prediction runs |
| POST | `/predictions/validate` `{run_id?, tickers?, provider?}` | LLM review (openai / anthropic / ollama) of a stored run; stored separately in `llm_validations`, never alters predictions. Statuses: supported, weakly_supported, contradicted, indeterminate, not_performed, validation_error, api_error |
| POST | `/backtest` `{start, end, tickers?, horizon_types?, step?, tune?, n_trials?, objective?, param_version?, availability_mode?, train_end?, valid_end?}` | Start a backtest / tuning run (background). `availability_mode`: `provider_timestamp` (default) or `verified` (only news fetched before each cut-off). Tuning splits train ≤ `train_end` < validation ≤ `valid_end` < untouched test (default 60/20/20) |
| GET | `/backtest/results` | Runs and status |
| GET | `/backtest/results/{id}` | AI-readable JSON report |
| GET | `/backtest/results/{id}/report.md` | Human-readable report |
| POST | `/backtest/score-live` | Attach actuals to matured live predictions; live track record |
| GET | `/parameters` | Parameter sets (active / candidate / archived) |
| GET | `/parameters/changes` | Proposed/applied changes: name, old, new, reason, backtest result, timestamp |
| POST | `/parameters/{version}/activate` | Explicitly promote a parameter set |
| GET | `/companies?on=&catalog=` / PUT `/companies/{ticker}` | Effective universe on a date with the weights effective then (`catalog=true`: every catalogued instrument) |
| GET | `/universe?on=` | Active Top-10 version (members, weights, effective dates, source), versions requiring confirmation with the changes they would make, full history |
| GET | `/news/audit?start=&end=` | Availability audit: overall, by provider, by company, by trading date |
| GET | `/market-data?ticker=HDFCBANK.NS` | Stored closes |
| GET | `/market-data/coverage` | First/last date per symbol |
| POST | `/market-data/refresh?start=&end=` | Download prices for the universe + NIFTY |
| GET | `/config` | Display timezone, active providers, calendar source, today's horizons |

### Example prediction item

```json
{
  "company": "HDFC Bank", "ticker": "HDFCBANK",
  "horizon_type": "today", "target_date": "2026-10-01",
  "previous_close": 951.4, "previous_close_date": "2026-09-30",
  "sentiment_score": 12.4,
  "predicted_movement_percent": 0.087, "predicted_direction": "NEUTRAL",
  "estimated_price": 952.23, "estimated_price_note": "model-derived estimate, not a quote",
  "uncertainty_1sigma_pct": 1.21,
  "contributions_pct": {"company_sentiment": 0.075, "macro_sentiment": -0.01, "nifty_sentiment": 0.02, "drift": 0.002},
  "evidence_news_ids": [812, 799],
  "features": {"model_inputs": {"relative_sentiment": 4.2, "sentiment_zscore": 0.11, "...": "..."},
               "news_flow": {"status": "ok", "relative_sentiment": 4.2, "baseline_mean": 8.2, "news_count": 3}},
  "model_version": "factor-v2@default-v3",
  "validations": [{"validation_status": "CONSISTENT", "confidence": 0.7, "reasoning": "..."}]
}
```
(Values illustrative of the shape only. These are model estimates, not guaranteed outcomes.)
