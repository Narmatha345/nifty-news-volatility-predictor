# Data sources

| Data | Adapter | Key | History | Notes |
|---|---|---|---|---|
| News (publisher feeds) | `publisher_rss` | none | no (recent items only) | Official RSS of The Economic Times, Mint, Business Standard, CNBC-TV18 (`config/news_feeds.yaml`). Exact timestamps with +0530 offset, headline + description. |
| News headlines | `google_news_rss` | none | yes (`after:`/`before:` operators) | Headline, publisher, link, pubDate. No body. ≤100 items per query, so backfill in weekly windows. |
| News | `newsapi` ([newsapi.org](https://newsapi.org)) | `NEWSAPI_KEY` | ~1 month on free tier | Adds description/summary. 100 requests/day free. |
| News | `gnews` ([gnews.io](https://gnews.io)) | `GNEWS_API_KEY` | plan-dependent | Indian sources via `country=in`. |
| News | `marketaux` ([marketaux.com](https://www.marketaux.com)) | `MARKETAUX_API_KEY` | yes | Entity tagging; its own sentiment is kept in `raw` for comparison only. |
| Daily prices | `yfinance` | none | years | NSE tickers `XXX.NS`, NIFTY 50 `^NSEI`. Adjusted closes. Unofficial API — verify critical numbers. |
| Trading calendar | `exchange_calendars` XBOM | none | — | BSE calendar (NSE shares trading holidays). |
| LLM validation | Anthropic Claude / Ollama | `ANTHROPIC_API_KEY` | — | Review only; never produces the prediction. |
| NIFTY membership & weights | `backend/config/universe.yaml` → `universe_members` | — | versioned | Date-aware versions with source and source date: `legacy-approx-2025-06-30` (approximate, until 2026-09-29) and `nse-factsheet-2026-09-30` (NSE Indices factsheet, confirmed 2026-10-02). No free official API; reference data, not live values. |

Only official APIs/feeds are used; no article pages are scraped.

## Timestamp quality by provider

| Provider | Timestamp | Precision observed | Timezone | Text |
|---|---|---|---|---|
| google_news_rss | RFC-822 `pubDate` | recent items to the second; older/backfilled items date-only (`07:00:00 GMT`) | explicit GMT | headline only |
| publisher_rss | RFC-822 `pubDate` | seconds (some minute) | explicit +05:30 | headline + description |
| newsapi | `publishedAt` ISO-8601 `Z` | seconds | explicit UTC | description |
| gnews | `publishedAt` ISO-8601 `Z` | seconds | explicit UTC | description |
| marketaux | `published_at` ISO-8601 with offset | microseconds | explicit | snippet |

Active without keys: Google News RSS and the publisher feeds (two independent providers; the same
story arriving through both is linked as a near-duplicate and counted once). None of the free tiers provides full
article text; `article_text` stays empty rather than being scraped or invented.

## Timestamp quality and availability
Google News RSS returns most older items with a **date-only** pubDate (`07:00:00 GMT`). Such
timestamps are stored with `timestamp_precision = coarse` and treated as available only from the end
of that date. Every article also records `fetched_at`, and `availability_status`
(`verified_pre_market`, `published_before_cutoff`, `collected_after_event`, `unknown`) says whether it
was demonstrably held before the cut-off it is used for. As of 2026-10-02 all stored news was fetched on
2026-10-01/02, so **none** of the May–Sep 2026 news is verified pre-market: historical backtests on it
rely on provider timestamps. Run `python -m backend.cli daily` before 09:15 IST every trading day to
build a verified history.

## Adding a news provider
```python
from backend.system1_news.providers import register
from backend.system1_news.providers.base import NewsProvider, RawArticle

class MyProvider(NewsProvider):
    name = "myprovider"
    supports_history = True
    def is_configured(self): return bool(os.getenv("MY_KEY"))
    def search(self, query, start=None, end=None, limit=50) -> list[RawArticle]:
        ...  # call the API via backend.services.http.request_with_retry, map fields

register(MyProvider)
```
Then add `myprovider` to `NEWS_PROVIDERS`. Use the provider's own publish timestamp; drop items
without one.
