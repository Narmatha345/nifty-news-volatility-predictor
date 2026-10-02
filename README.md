# NIFTY News to Volatility Predictor

A research system that turns financial and macro-economic news into quantitative sentiment scores
(−100 … +100) and uses them to estimate the percentage movement of the Top-10 NIFTY 50 stocks —
for today, the next 3 month-ends and the next 6 quarter-ends — then measures, on history, how good
those estimates actually are.

> **Research output only.** Predicted movements and estimated prices are model estimates, not
> guaranteed outcomes. Historical accuracy is reported exactly as measured, including when the
> model does not beat a no-change forecast. Historical performance does not guarantee future performance.

> **Model status: EXPERIMENTAL.** On the May–Sep 2026 replay no model version beats the no-change
> forecast out-of-sample, and none of that period's news was demonstrably collected before its
> prediction cut-off. See [Findings](docs/METHODOLOGY.md#findings-may-sep-2026), the out-of-sample
> study [reports/exp-102ab0c4.md](reports/exp-102ab0c4.md), the backtests by news-availability mode
> ([provider timestamps](reports/bt-d8fc63d143.md), [exact timestamps only](reports/bt-8219b40a9c.md),
> [verified only](reports/bt-e82b802213.md)) and the news audit [reports/audit-2026-10-02.md](reports/audit-2026-10-02.md).
> Predictive skill remains experimental because the verified pre-market dataset is not yet large enough.

| System | Does | Code |
|---|---|---|
| 1. News collector & sentiment scorer | Collects news via API adapters, de-duplicates, detects companies/macro topics, extracts keywords, scores sentiment, aggregates company / macro / NIFTY scores per horizon | [backend/system1_news](backend/system1_news) |
| 2. Deviation predictor | Combines *relative* sentiment (news vs each company's usual tone) with previous close, volatility and beta into a predicted % move + estimated price (factor-v2, or a System 3-fitted ridge model); LLM reviews the result | [backend/system2_prediction](backend/system2_prediction) |
| 3. Backtester & fine-tuner | Replays 1 + 2 on history with no look-ahead, compares with four transparent baselines, analyses features, tunes on train / validation / untouched test, proposes (never activates) parameter versions, writes human + AI reports | [backend/system3_backtesting](backend/system3_backtesting) |

Docs: [Architecture](docs/ARCHITECTURE.md) · [Methodology & formulas](docs/METHODOLOGY.md) ·
[API](docs/API.md) · [Data sources](docs/DATA_SOURCES.md)

## Deploy to Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Narmatha345/nifty-news-volatility-predictor)

`render.yaml` defines one free web service. When you create it Render asks for **`OPENAI_API_KEY`**
(enter your key there - it is never stored in the repository).
The deployed app is fully open: anyone with the link can view it and use every action (collect news,
predict, validate with OpenAI, backtests, parameter activation). OpenAI validations are billed to your
key, so set a monthly spending limit in the OpenAI dashboard.

Free-plan behaviour: the disk is ephemeral and the service sleeps after ~15 minutes without traffic
(the first request after that takes ~1 minute). On every start the bundled snapshot
`deploy/nifty_nvp_seed.sqlite.gz` (news, prices, predictions and reference backtests up to 2026-10-02) is
restored, then prices, news and a fresh prediction are refreshed in the background. Nothing written on
Render survives a restart; the continuous verified-news collection runs on the local scheduled jobs.

## Setup

Requires Python 3.11+.

```powershell
cd D:\nifty-news-volatility-predictor
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env      # then fill in keys you have
```

`.env` settings (all optional except where noted):

| Variable | Purpose |
|---|---|
| `NEWSAPI_KEY`, `GNEWS_API_KEY`, `MARKETAUX_API_KEY` | Extra news providers. Without keys, Google News RSS is used alone. |
| `LLM_PROVIDER`, `OPENAI_API_KEY`, `OPENAI_MODEL` | LLM **validation** (review only, never changes predictions). `openai` (default in `.env.example`) needs both key and model; otherwise validation is stored as "not performed". `anthropic` (`ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`) and `ollama` remain available. |
| `CA_BUNDLE` | CA bundle path if HTTPS is intercepted by antivirus/proxy (this machine needs it). |
| `DATABASE_URL` | Defaults to SQLite at `data/nifty_nvp.db`. |
| `DISPLAY_TIMEZONE` | Dashboard timezone (default `Asia/Kolkata`). Storage is always UTC. |
| `QUARTER_CONVENTION` | `indian_fy` (default) or `calendar`. |
| `SENTIMENT_ENGINE` | `lexicon` (default) or `finbert` (`pip install -r requirements-finbert.txt`). |

## Usage

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m backend.cli init                                  # seed Top-10 universe + default parameters
& $py -m backend.cli market-data --start 2024-06-01        # daily prices for Top-10 + NIFTY 50
& $py -m backend.cli collect                               # live news -> analysis
& $py -m backend.cli predict --validate                    # System 1 + 2 (+ LLM review)
& $py -m backend.cli backfill --start 2026-04-01 --end 2026-09-30   # historical news
& $py -m backend.cli analyze                               # (re)analyze stored news with the current engine
& $py -m backend.cli audit-news                            # availability audit (overall/provider/company/date)
& $py -m backend.cli backtest --start 2026-05-01 --end 2026-09-30 --param-version default-v3 `
      --tune --trials 30 --train-end 2026-07-31 --valid-end 2026-08-31   # train May-Jul / valid Aug / test Sep
& $py -m backend.cli experiment --start 2026-05-01 --end 2026-09-30 `
      --train-end 2026-07-31 --valid-end 2026-08-31        # before/after + out-of-sample model study
& $py -m backend.cli daily --validate                      # live workflow (below)
& $py -m backend.cli score-live                            # score matured live predictions
& $py -m backend.cli seed-universe --update                # catalog + (new) universe versions into the DB
& $py -m backend.cli universe                              # active + historical Top-10, pending versions
& $py -m backend.cli universe --confirm nse-factsheet-2026-09-30   # ONLY after the owner confirms it
& $py -m backend.cli outcomes                              # after 15:30 IST: prices -> actual outcomes
& $py -m backend.cli serve                                 # dashboard: http://127.0.0.1:8000
```

### Daily live workflow (schedule before 09:15 IST)
`daily` runs collect → timestamp validation and availability audit → de-duplication → scoring →
price refresh → prediction (→ optional LLM review). It predicts the first NSE session whose 09:15 IST
open is still ahead: run after the open, it targets the **next** session; weekends and exchange
holidays are skipped. Every article it stores before the open is `verified_pre_market`, which is the
only kind of news that supports a true live-news backtest. Live runs use only news this system held before the cut-off and store the
cut-off, universe version, news snapshot and a VERIFIED / UNCERTAIN / NO_NEW_NEWS flag per prediction.
After the close, `outcomes` stores the actual return, direction and errors.

Register both jobs (08:30 and 16:30 IST on weekdays, current user, no stored credentials):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\schedule_windows.ps1          # create / update
powershell -ExecutionPolicy Bypass -File scripts\schedule_windows.ps1 -Remove  # delete
```

### Parameter sets and universe (current state, 2026-10-02)
* **Active parameters: `default-v3`** (factor-v2: relative sentiment, price reports excluded), activated
  explicitly on the owner's instruction. `default-v2` (the raw-sentiment model with the UP bias) is
  archived but kept for exact replay. Tuned sets are never activated automatically.
* **Universe:** NSE factsheet of 2026-09-30 (KOTAKBANK and M&M replace ITC and TCS from 2026-09-30);
  earlier dates keep the original membership. `python -m backend.cli universe`.
* **News providers:** Google News RSS + publisher RSS (ET, Mint, Business Standard, CNBC-TV18), both
  keyless. NewsAPI / GNews / Marketaux activate automatically when their keys are added to `.env`.
* **Scheduled (Windows Task Scheduler, current user):** `NiftyNews-PreMarket` 08:30 IST weekdays
  (`daily --validate`) and `NiftyNews-AfterClose` 16:30 IST weekdays (`outcomes`); logs in
  `data/daily.log` / `data/outcomes.log`. Remove with `scripts\schedule_windows.ps1 -Remove`. Runs only
  while this user is logged on and the PC is awake.
* **LLM validation:** OpenAI; recorded as "not performed" until `OPENAI_API_KEY` and `OPENAI_MODEL` are set.
  Test with `.\.venv\Scripts\python.exe -m backend.cli validate --tickers HDFCBANK`.

### Web app (`python -m backend.cli serve` → http://127.0.0.1:8000)
Separate pages with a sidebar; every page has its own URL.

| Page | URL | What it does |
|---|---|---|
| Overview | `/` | Status of System 1/2/3, scheduled jobs (next/last run, log), configuration |
| System 1 · News & Sentiment | `/system-1` | Collect news, recompute sentiment; NIFTY / macro / company scores for all 10 horizons, charts, news data quality (VERIFIED / UNCERTAIN / POST-EVENT), latest news with filters |
| System 2 · Predictions | `/system-2` | Refresh prices, run predictions, validate with OpenAI (all or one company), score outcomes; run selector, cut-off and data quality, model predictions table, a **separate** LLM-validation section, live track record |
| System 3 · Backtesting | `/system-3` | Start backtests (news mode, tuning), results vs four baselines, confusion matrix, company / month tables, fine-tuning, out-of-sample experiments with significance tests, reports library (Markdown + JSON) |
| Settings | `/settings` | Top-10 universe (active + history), parameter sets (explicit Activate), news providers, LLM configuration (key never shown), scheduled jobs |

## Tests

```powershell
& $py -m pytest                         # offline unit + integration tests
$env:RUN_LIVE_TESTS="1"; & $py -m pytest tests/test_live.py   # real Google News + Yahoo calls
```

Coverage: news parsing, timestamp normalisation, duplicate detection, company identification,
keyword extraction, sentiment scoring (incl. real-headline regressions), aggregation, horizons,
feature generation, prediction & estimated-price maths, no-look-ahead guarantees, predicted-vs-actual
alignment, metrics (hand-computed), tuning (never auto-activates), report sections, API, contextual
sentiment, price-report detection, relative sentiment (baseline, insufficient history, no future news),
UP/DOWN/NEUTRAL predictions, holidays / after-open targeting, availability classification, verified-news
mode, point-in-time baselines, chronological splits, tuner and experiment blindness to the test period,
candidate versioning, and a full
News → System 1 → System 2 → LLM validation → System 3 integration test.

## Traceability
Every prediction row links to the news IDs behind it; each news analysis stores the engine version
and matched terms; aggregated snapshots store their top contributors; predictions store features
and per-term contributions; actual outcomes and errors are stored once known:

```
news → timestamp → company → sentiment (+ evidence) → aggregated score → features
     → prediction (+ contributions, model version) → actual outcome → error
```
