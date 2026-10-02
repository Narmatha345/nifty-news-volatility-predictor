# Methodology

Everything below is implemented in code; file references point to the implementation.
All parameters named in `monospace` live in [backend/config/model_params.yaml](../backend/config/model_params.yaml)
and are versioned in the `parameter_sets` table.

> **Status (Oct 2026): experimental.** On the May–Sep 2026 replay neither the previous model
> (`factor-v1`) nor the revised one (`factor-v2`, relative sentiment) nor any fitted model beats a
> no-change forecast out-of-sample, and none of the 2026 news was demonstrably available before its
> cut-off. See [Findings](#findings-may-sep-2026) and the reports in `reports/`.

---

## System 1 — News collection & sentiment scoring

### 1. Collection ([collector.py](../backend/system1_news/collector.py))
* One query per active company (its primary alias, quoted) plus the `macro_queries` in
  [taxonomy.yaml](../backend/config/taxonomy.yaml), sent to every enabled provider adapter.
* **Timestamp validation**: an article needs a provider publish timestamp (never stamped "now")
  and is rejected if that timestamp is more than 15 minutes in the future.
* **Exact duplicates**: SHA-256 of the canonical URL is a unique key. **Near duplicates**:
  Jaccard ≥ 0.8 between title token sets within ±48 h, linked through `duplicate_of`.
* Every stored article gets `timestamp_precision` and `availability_status` (next section).

### 2. News availability integrity ([availability.py](../backend/system1_news/availability.py))
Every article stores: provider, source, URL, title, description, article text (only if the provider
returns it — none of the current ones do), stable provider id, the timestamp **as received**
(`published_raw`), its timezone and `timezone_status` (explicit / provider_documented / unknown),
`published_at` (UTC), `published_at_ist`, `fetched_at` / `fetched_at_ist`, `timestamp_precision`,
`availability_status` and `first_usable_session`. Entity, event type, relevance, sentiment and engine
version are stored per entity in `news_analysis`.

**Precision** — `exact` (seconds), `minute`, `hour`, `date_only`, `unknown`. Never upgraded:
Google News RSS returns older items as `07:00:00 GMT`, which is classified `date_only`, not 07:00.
A timestamp without an offset is normalised only when the provider documents its timezone; otherwise
`timezone_status = unknown`.

**Cut-off rule (09:15 IST) — per prediction** (`availability.eligibility`):

| Eligibility | Condition | Live | Research replay |
|---|---|---|---|
| `verified` | `fetched_at` ≤ cut-off (this system held it) | used | used |
| `timestamp_before` | exact/minute/hour stamp with known timezone before the cut-off (hour stamps count from the END of the hour), fetched later | **never** | `exact_timestamp`, `provider_timestamp` modes |
| `post_cutoff` | stamp at/after the cut-off | never | never (belongs to a later session) |
| `uncertain` | date-only / unknown precision / unknown timezone, fetched after the cut-off | **never** | `provider_timestamp` mode only, labelled UNCERTAIN |

Future timestamps (> now + 15 min) are rejected at collection; missing timestamps are rejected; both
are counted in the fetch log. Effective availability time = the earlier of `fetched_at` and what the
provider stamp allows (exact/minute: the stamp; hour: +1 h; date-only/unknown tz: end of that UTC date).
Check on the 2026 data: date-stamped "shares move X%" headlines correlate 0.48 with the return of the
*same* session and 0.03 with the next one, so the date is the event date and the end-of-date rule does
not leak.

**Article status**, relative to *S* = its first usable session:

| Status | Meaning |
|---|---|
| `verified_pre_market` | `fetched_at ≤ open(S)`: demonstrably held before the session it is used for |
| `published_before_cutoff` | precise stamp, fetched during session *S* |
| `collected_after_event` | precise stamp, fetched after *S* closed (retrospective collection) |
| `unknown` | date-only / unknown precision or timezone, not verified |

`python -m backend.cli audit-news` reports these overall, by provider (with precision shares,
duplicates, rejections and each adapter's documented timestamp quality), by company and by trading
date (with whether a live prediction was generated); written to `reports/audit-<date>.md/.json`.

### 3. Relevance and event type ([relevance.py](../backend/system1_news/relevance.py))
* Company detection: word-boundary alias matching; short tickers case-sensitive; subsidiaries
  stripped via `exclude_patterns`. Relevance 0.7 (title) / 0.4 (summary) + mentions; divided by
  √(n−1) when more than two companies are named.
* **Event types** (taxonomy.yaml): company `earnings, profit, revenue, guidance, contract, order,
  regulatory, management, m_and_a, corporate_action, debt, credit, analyst_rating, price_action,
  other`; macro `interest_rate, inflation, gdp, currency, fund_flows, commodity, government_policy,
  geopolitical, market_wrap`. Highest-strength category wins, title matches first.
* **Price reports.** A headline that only reports a price move ("HDFC Bank shares fall 3%",
  "Infosys slips 7% to 52-week low", "Sensex jumps 500 points") is flagged `price_report`. A move
  word whose subject is a fundamental quantity ("profit rises 12%") is not. *Rationale:* by the time
  such an article can be used, the move it describes is already in the previous close; its wording
  is positive or negative *about the past*. `price_report_weight` (default 0) removes these from the
  news signal; their information is represented by the `previous_return` feature instead.

### 4. Article sentiment ([sentiment.py](../backend/system1_news/sentiment.py))
Default engine `lexicon-v3` (deterministic, explainable; stored terms explain every score):
* Financial polarity lexicon; title ×2, summary ×1; `score = 100 · tanh(raw / 3)`.
* Inverted objects (`inflation reaches 16-month high` −, `rate cut` +, `losses narrow` +).
* **Context (new in v3)** — positive language is not the same as positive impact:
  * expectation-relative phrases override the direction word: `better/worse than expected`,
    `above/below estimates`, `falls less than expected` (+), `rises less than expected` (−);
  * events: `regulatory action`, `show-cause notice`, `tax demand` (−); `wins/bags/secures … contract/order/deal` (+);
  * contrast: terms before `but`/`however` ×0.5, after ×1.5 (`profit rises but misses estimates` −);
    concession: terms after `despite`/`although`/`though` ×0.5.
* Negation, intensifiers as before. Analyses are keyed by engine version: `lexicon-v2` rows are
  kept so the previous model can be replayed exactly.

### 5. Aggregation ([aggregation.py](../backend/system1_news/aggregation.py))
**Raw score** (unchanged formula, legacy input):
```
usable  = available <= as_of, relevance >= min_relevance, age <= max_lookback_days
w       = reliability · relevance · strength · decay · (1 + corroboration_bonus · ln(cluster_size)) · (price_report_weight if price report)
score   = (Σ w·s / Σ w) · W / (W + shrinkage_k)
```
With a 120-day lookback and hundreds of articles, `W ≫ shrinkage_k`, so this score is essentially
the company's **habitual headline tone**. On the 2026 data the mean article sentiment is positive
for every company (+4 TCS … +25 L&T) and negative for macro (−3.4).

**Relative sentiment (news flow)** — the input of the revised model:
```
window_start       = 09:15 IST open of the previous session (the previous prediction cut-off)
new news  N        = clusters available in (window_start, as_of]
baseline  B        = clusters available in (window_start − baseline_days, window_start]
current            = weighted mean sentiment of N              (None if no new news)
relative_sentiment = current − mean(B)                         (0.0 if no new news)
sentiment_zscore   = relative_sentiment / std(B)               (None if std(B) = 0)
```
* Only information available at `as_of` is used; the baseline ends where the new window starts.
* `< min_baseline_n` baseline stories → the pooled baseline of all universe companies is used
  (`ok_pooled_baseline`, it still removes the general positive skew); if that is also thin the
  value is `None` (`insufficient_history`) — never guessed.
* NIFTY relative = weight-averaged company relative sentiments; macro relative uses the MACRO entity.

### 6. Horizons ([horizons.py](../backend/services/horizons.py))
Unchanged: next session, next 3 month-ends, next 6 quarter-ends (last trading session on or before
the nominal date); `exchange_calendars` XBOM; projected dates flagged.

---

## System 2 — Deviation predictor

### Target (next-session horizon)
```
actual_return_% = (close[S] − close[previous session]) / close[previous session] × 100
```
*S* = the predicted NSE session (first session whose 09:15 IST open is after the run time;
weekends/holidays skipped via the exchange calendar live and via the stored sessions in backtests).
The previous close is the last stored close strictly before *S*; a stale previous close makes the
prediction `MISSING_DATA`. Closes are Yahoo Finance split/dividend-adjusted. Timezone: everything is
stored as naive UTC; cut-offs are 09:15 IST = 03:45 UTC. Verified in the backtester
(`test_target_previous_close_and_baselines_are_point_in_time`).

### Features ([features.py](../backend/system2_prediction/features.py)) — all point-in-time
`raw_sentiment, relative_sentiment, sentiment_zscore, news_count (log1p), news_relevance,
news_recency (exp(−hours/24)), macro_sentiment, macro_relative, nifty_sentiment, nifty_relative,
historical_volatility, beta, previous_return, rolling_return_5d, rolling_return_20d`.
Every prediction stores the full vector (`features.model_inputs`) and the news-flow details.

### Models ([model.py](../backend/system2_prediction/model.py), [linear.py](../backend/system2_prediction/linear.py))
Selected by the parameter set:

* **factor-v1** (`sentiment_input: raw`, the May–Sep 2026 model): news term
  `sensitivity · (w_c·S_c + w_m·m·S_m + w_n·β·S_nifty)/100 · σ_h` on raw scores.
* **factor-v2** (`sentiment_input: relative`, default-v3): the same weights on **relative**
  sentiment, scaled by the *daily* σ (a news surprise is a one-off price adjustment; it does not
  grow with the horizon, while the uncertainty band σ_h still does). No new news ⇒ 0.
* **ridge-v1** (`model: linear`): a ridge regression on standardised features, fitted by System 3
  on a training window and stored in the parameter set. h = 1 uses the fitted value; h > 1 uses
  only the feature-driven shock (the intercept is a trailing mean return and is not extrapolated).

Direction: UP / DOWN beyond ±`neutral_threshold_sigma`·σ_h, else NEUTRAL. Estimated price =
previous close · (1 + predicted%/100) — a model estimate, never a quote.

**Live session.** A run predicts the first session whose 09:15 IST open is still ahead; run after
the open it targets the next session. `python -m backend.cli daily` = collect → timestamp validation
→ dedup → relevance / company-macro classification → sentiment → relative sentiment → prediction →
storage. Live runs use **only `verified` news** (fetched before the information time, which is itself
before the cut-off) and only closes before the target session. Each run stores `prediction_cutoff`
(e.g. 2026-10-05 09:15:00+05:30), `target_session`, `universe_version`, `sentiment_engine`,
`availability_policy` and a data-quality summary; each prediction stores its full feature vector,
the snapshot of every news item behind it (`news_inputs`: ids, timestamps, precision, eligibility,
event type, sentiment, engine) and a flag: **VERIFIED** (all new news held before the cut-off),
**UNCERTAIN** (some relied on provider timestamps — research replays only) or **NO_NEW_NEWS**.
Stored predictions are never modified.

**After the close.** `python -m backend.cli outcomes` (after 15:30 IST) refreshes prices and writes
`actual_results`: actual close and return, actual direction, error, absolute and squared error, and
the previous-close date. Only sessions that have closed are scored; the target is always an exchange
session (holidays are skipped by the calendar, never scored as zero returns); a target close missing
from the price data while later sessions exist is reported as `missing_target_price`, not guessed.
`scripts/schedule_windows.ps1` registers both jobs (08:30 and 16:30 IST, weekdays) for the current
user without storing credentials.

### LLM validation ([validator.py](../backend/system2_prediction/validator.py), [llm.py](../backend/services/llm.py))
PREDICTION → the LLM reviews the prediction and its evidence → a structured validation stored in
`llm_validations`. The numeric prediction (return, direction, sentiment, previous close, features) is
authoritative and never modified; nothing the LLM returns is copied into it.

* **Provider** = `LLM_PROVIDER` only (`openai` | `anthropic` | `ollama` | `none`); no silent fallback.
  `openai` uses the official SDK (Chat Completions, strict JSON-schema output) with `OPENAI_API_KEY`
  and `OPENAI_MODEL` from the environment. Without them the result is stored as `not_performed`
  ("LLM validation not performed: OpenAI API key is not configured.").
* **Sent:** company, target session, cut-off, previous close, predicted return / direction /
  uncertainty / contributions per horizon, relative sentiment, stored model features, event types,
  data-quality flag and only the news the prediction used whose eligibility the run's availability
  policy allows (headline, timestamps, eligibility, event type, sentiment). **Never sent:** actual
  closes, returns or errors (even after the session closed), news available or fetched after the
  cut-off, or date-only/uncertain news under the live verified-only policy. A leakage check runs
  before every call; a failing payload is not sent (`leakage_blocked`).
* **Returned schema:** `validation_status` (supported | weakly_supported | contradicted |
  indeterminate), `reason`, `key_supporting_factors`, `key_contradicting_factors`,
  `data_quality_issues`, `cutoff_check` (passed | failed | uncertain), `missing_information`. Every
  response is checked against it (exact fields, types, enums; any extra field such as a replacement
  prediction is rejected) → otherwise `validation_error`. API failures/timeouts → `api_error`.
* **Pipeline safety:** the prediction is committed before any LLM call; validation failures never
  lose or alter it. Error text is redacted (keys, `Bearer` / `Authorization` values) before it is
  logged, stored or returned; keys are excluded from settings `repr`.
* Command: `python -m backend.cli validate [--run-id R] [--tickers T ...]` (or `daily --validate`).

---

## System 3 — Backtesting, evaluation & tuning

### No look-ahead ([backtester.py](../backend/system3_backtesting/backtester.py))
Per evaluation session *d*: cut-off 09:15 IST on *d*; System 1 sees only items available by then
(optionally only verified ones); relative-sentiment baselines end at the previous cut-off; System 2
uses closes `< d`; the outcome is the close on the target session.

### Baselines (closes strictly before *d*, × horizon sessions *h*)
| Baseline | Forecast |
|---|---|
| No change | 0 |
| Historical mean | mean daily return of all universe stocks over all stored history before *d* |
| Company historical mean | the company's mean daily return before *d* |
| Rolling mean (20 sessions) | the company's mean daily return over the last 20 sessions |

Every comparison uses identical samples.

### Metrics ([metrics.py](../backend/system3_backtesting/metrics.py))
Directional / sign accuracy (binomial p vs 50 %), MAE, RMSE, MAPE, correlation, **bias**
(mean predicted − actual), prediction distribution, **3×3 confusion matrix**, baseline comparison,
per-component contribution summary, and **feature analysis** (count, missing, mean, std, Pearson
and rank correlation with p-value, sign agreement, top-minus-bottom-quintile return).

**Is an improvement real?** `paired_improvement_test`: Diebold–Mariano-style one-sided t-test on
the loss differential |model error| − |baseline error|, averaged per date first (the 10 stocks of a
day share market moves). A lower MAE without p < 0.05 is not counted as skill.

### Out-of-sample experiment ([experiments.py](../backend/system3_backtesting/experiments.py))
`python -m backend.cli experiment --start 2026-05-01 --end 2026-09-30 --train-end 2026-07-31 --valid-end 2026-08-31`
* Chronological split, never shuffled: **train May–Jul, validation Aug, test Sep** (or 60/20/20).
* Incremental feature sets: intercept only → + raw sentiment → + relative sentiment → + news
  features → recommended initial set → market only → news + market → full.
* Ridge penalty and the selected set are chosen on **validation**; **test** is scored afterwards.
  L2 logistic regression for P(up) vs the majority class.
* Walk-forward: expanding monthly windows, penalty chosen on the preceding month.
* Before/after: factor-v1 (`default-v2`, `lexicon-v2` analyses) vs factor-v2 (`default-v3`) on
  identical rows. A candidate is stored only if the selected model **significantly** beats every
  baseline on validation **and** test; it is never activated.

### Tuning ([tuner.py](../backend/system3_backtesting/tuner.py))
Seeded random search, **train / validation / untouched test**: trials are scored on train only; the
best is gated on validation (must beat the active set and the no-change MAE); only then is the test
period evaluated, once, and it can only veto (must significantly beat no-change). Tests prove that
scrambling test-period prices does not change the selected parameters. Activation stays explicit:
`POST /api/parameters/{version}/activate`. A new default parameter version is seeded as a
*candidate* whenever another set is active.

### Reports ([reports.py](../backend/system3_backtesting/reports.py))
Data quality (availability statuses, coarse timestamps, duplicate rate, price-report share, missing
prices), overall results, prediction distribution + confusion matrix + contribution summary, baseline
comparison, results by horizon, feature analysis, company-wise table (predictions, MAE, RMSE,
directional accuracy, bias, UP/DOWN/NEUTRAL distribution), monthly results, tuning (train / validation
/ test), failure cases and rule-based observations. Markdown + JSON in `reports/`.

---

## Findings (May–Sep 2026)

See `reports/` for the full numbers (experiment and backtest IDs are listed in the README).

* **UP bias — cause.** The previous model fed *raw* scores that are long decayed averages of
  headline tone. Company scores were positive on 88 % of next-session predictions, the NIFTY
  company-weighted term (× beta) was positive on 100 %, macro negative on 100 %; drift and intercept
  were 0. Result (replayed on the stored `lexicon-v2` analyses, 1 090 next-session predictions):
  339 UP / 0 DOWN / 751 NEUTRAL against 429 actual UP / 505 DOWN / 156 NEUTRAL. The bias came from the
  sentiment *level* (and its use in the NIFTY term), not from the formula, beta, volatility,
  thresholds or drift.
* **Fix.** Relative sentiment removes the habitual tone; price reports are excluded; context rules
  fix wording-vs-impact errors. The revised model makes 204 UP / 250 DOWN / 636 NEUTRAL calls; mean
  bias (predicted − actual) falls from 0.180 to 0.057 pct-pts. Its MAE is slightly *higher*
  (1.071 vs 1.062; no-change 1.039): it now makes more directional calls, and they are no better
  than chance (sign accuracy 48.5 %, p = 0.53). Removing the bias was necessary, not sufficient.
* **Skill.** Neither version beats the no-change forecast; no fitted model's improvement over any
  baseline is statistically significant on validation or test. Relative sentiment's correlation
  with next-session returns is ≈ −0.04 (not significant); only `nifty_relative` and
  `rolling_return_5d` reach p < 0.05 univariately, both *negative* (reversal-like), across 15 tests
  without multiple-comparison correction.
* **Data limitation.** 0 of the ~16 k articles in the period were collected before their cut-off;
  ~92 % have date-only timestamps. The historical test is therefore weaker than a live-news test and
  the result is best described as *does not beat the baseline on the available, mostly
  retrospectively collected news*.

### Universe (date-aware)
`universe_members` stores immutable versions (ticker, company, weight, effective_from/to, source,
source_date, status). For a date *d* the universe is the **confirmed** version with the latest
effective_from ≤ *d*; backtests switch versions on their effective dates, so historical periods use
their own membership and weights. Versions: `legacy-approx-2025-06-30` (the project's original
configuration, approximate unverified weights; in effect until 2026-09-29) and
`nse-factsheet-2026-09-30` (official NSE factsheet; **confirmed by the project owner on 2026-10-02**;
replaces ITC and TCS with KOTAKBANK and M&M from 2026-09-30). News is analysed against every company
of every confirmed version, and prices are refreshed for all of them, so former members keep their
history and pending outcomes. `python -m backend.cli universe [--confirm VERSION]` shows / confirms.
*Correction:* the Phase 2 final backtest (`bt-b402c7db01`) used the 2026-09-30 factsheet weights for
May–Sep; Phase 3 re-runs use the weights effective then.

### Known limitations
* The lexicon engine remains shallow (numbers, sarcasm, multi-sentence context).
* Google News RSS gives headlines only, with date-only timestamps for older items.
* Universe weights are reference data from the 2026-09-30 factsheet; ITC and TCS are no longer in
  the official top 10 (stored with an explicit upper-bound weight).
