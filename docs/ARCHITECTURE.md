# Architecture

```text
 News providers (adapters)        Market data (adapter)
 google_news_rss | newsapi |      yfinance (.NS, ^NSEI)
 gnews | marketaux | <yours>              │
          │                               │
          ▼                               ▼
┌───────────────────────────┐   ┌──────────────────┐
│ SYSTEM 1  system1_news/   │   │ data/market_data │
│ collector → dedup         │   │ validate → store │
│ analyzer: relevance,      │   └────────┬─────────┘
│   keywords, sentiment     │            │
│ aggregation (pure)        │            │
│ pipeline: horizons        │            │
└────────────┬──────────────┘            │
             │ company / macro / NIFTY scores per horizon
             ▼                           ▼
┌──────────────────────────────────────────────┐
│ SYSTEM 2  system2_prediction/                │
│ features (prior closes only) → model (pure)  │
│ predictor: rows + contributions + evidence   │
│ validator: LLM review (stored separately)    │
└────────────┬─────────────────────────────────┘
             ▼
┌──────────────────────────────────────────────┐
│ SYSTEM 3  system3_backtesting/               │
│ backtester (point-in-time replay of 1 + 2)   │
│ metrics → tuner (train/valid/test) → reports │
│ candidate parameter sets (never auto-active) │
└──────────────────────────────────────────────┘
```

## Layout

```text
backend/
  config/              settings (env), YAML reference data, loader
  database/            SQLAlchemy models + session management
  data/                market data adapters + storage
  services/            http (retry/rate-limit), calendar, horizons, parameters, universe,
                       llm adapters, orchestrator (live workflows), time utilities
  system1_news/        providers/, collector, dedup, relevance, keywords, sentiment,
                       analyzer, aggregation, pipeline
  system2_prediction/  features, model, predictor, validator
  system3_backtesting/ backtester, metrics, tuner, experiments, reports, runner
  api/                 FastAPI app + routers
  cli.py               command-line entry points
frontend/              dashboard (static HTML/JS, Chart.js)
tests/                 unit + integration (+ opt-in live) tests
docs/                  this documentation
```

## Design rules
* **Pure cores, thin orchestration.** `aggregation.aggregate`, `pipeline.score_horizons`,
  `predictor.build_predictions`, `model.predict` and `backtester.run_backtest` take plain data and
  touch no database. The live path and the backtest call the *same* functions.
* **Adapters everywhere.** News (`NewsProvider`), market data (`MarketDataProvider`), sentiment
  (`SentimentEngine`) and LLM (`LLMProvider`) are interfaces; register a new implementation without
  touching System 1/2/3.
* **Traceability.** prediction → `evidence_news_ids` → `news` + `news_analysis` (engine, matched
  terms) → `aggregated_scores` (contributors) → `features`/`contributions` → `actual_results`.
* **Versioning.** Parameter sets are immutable and versioned; model version = `<model>@<params>` (`factor-v1` raw sentiment, `factor-v2` relative sentiment, `ridge-v1` fitted linear model);
  sentiment analyses are keyed by engine version.
* **UTC internally.** Naive UTC datetimes in the DB; the dashboard converts to `DISPLAY_TIMEZONE`.
* **Missing is explicit.** Missing prices → `MISSING_DATA` predictions with a reason; missing
  sentiment → API status `MISSING`; nothing is interpolated or invented.

## Database tables
`companies`, `news`, `news_analysis`, `aggregated_scores`, `market_data`, `prediction_runs`,
`predictions`, `actual_results`, `llm_validations`, `parameter_sets`, `parameters` (flattened),
`parameter_changes`, `backtest_runs`, `fetch_log` — see [models.py](../backend/database/models.py).
SQLite by default; set `DATABASE_URL` for PostgreSQL.

## Reliability
* HTTP: exponential backoff with jitter on network errors and 408/425/429/5xx, honours
  `Retry-After`, per-provider minimum call spacing, no retry on other 4xx.
* Every provider call is logged in `fetch_log` (status, item counts, error).
* One failing provider/query never aborts a collection run.
* Provider responses are validated; invalid items are counted and skipped.
