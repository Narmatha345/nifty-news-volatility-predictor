"""Key-based JSON news APIs: NewsAPI.org, GNews.io, Marketaux.

Each adapter maps the provider's documented response fields to RawArticle with the timestamp's
precision / timezone as received. Missing timestamps are passed on as None so the collector rejects
and counts them (never defaulted to "now"). None of these free tiers returns full article text, so
article_text stays None; description/snippet is stored as the summary.
"""
from __future__ import annotations

import logging

from backend.config.settings import get_settings
from backend.services.http import request_with_retry
from backend.services.timeutil import parse_timestamp_info
from backend.system1_news.providers.base import NewsProvider

log = logging.getLogger(__name__)


def _iso(dt):
    return dt.replace(microsecond=0).isoformat() + "Z" if dt else None


class NewsAPIProvider(NewsProvider):
    """https://newsapi.org/docs/endpoints/everything — free tier: ~1 month history, 100 req/day."""
    name = "newsapi"
    supports_history = True
    min_interval_s = 1.0
    timestamp_timezone = "UTC"
    timestamp_quality = "ISO-8601 UTC ('Z') to the second (publishedAt). Description, no full text."

    def is_configured(self):
        return bool(get_settings().newsapi_key)

    def search(self, query, start=None, end=None, limit=50):
        params = {"q": query, "language": "en", "sortBy": "publishedAt", "pageSize": min(limit, 100)}
        if start:
            params["from"] = _iso(start)
        if end:
            params["to"] = _iso(end)
        resp = request_with_retry("GET", "https://newsapi.org/v2/everything", params=params,
                                  headers={"X-Api-Key": get_settings().newsapi_key},
                                  rate_key=self.name, min_interval_s=self.min_interval_s)
        data = resp.json()
        if data.get("status") != "ok":
            raise ValueError(f"newsapi error: {data.get('code')} {data.get('message')}")
        return [self.make_article((a.get("source") or {}).get("name") or "unknown", a.get("title") or "",
                                  a.get("url") or "", parse_timestamp_info(a.get("publishedAt"), self.timestamp_timezone),
                                  a.get("description"), query, {"author": a.get("author")}, a.get("url"))
                for a in data.get("articles", [])]


class GNewsProvider(NewsProvider):
    """https://gnews.io/docs/v4#search-endpoint"""
    name = "gnews"
    supports_history = True
    min_interval_s = 1.1
    timestamp_timezone = "UTC"
    timestamp_quality = "ISO-8601 UTC ('Z') to the second (publishedAt). Description, no full text on free tier."

    def is_configured(self):
        return bool(get_settings().gnews_api_key)

    def search(self, query, start=None, end=None, limit=50):
        params = {"q": query, "lang": "en", "country": "in", "max": min(limit, 100),
                  "apikey": get_settings().gnews_api_key, "sortby": "publishedAt"}
        if start:
            params["from"] = _iso(start)
        if end:
            params["to"] = _iso(end)
        resp = request_with_retry("GET", "https://gnews.io/api/v4/search", params=params,
                                  rate_key=self.name, min_interval_s=self.min_interval_s)
        return [self.make_article((a.get("source") or {}).get("name") or "unknown", a.get("title") or "",
                                  a.get("url") or "", parse_timestamp_info(a.get("publishedAt"), self.timestamp_timezone),
                                  a.get("description"), query, {}, a.get("id") or a.get("url"))
                for a in resp.json().get("articles", [])]


class MarketauxProvider(NewsProvider):
    """https://www.marketaux.com/documentation — supports Indian exchanges and date windows."""
    name = "marketaux"
    supports_history = True
    min_interval_s = 1.0
    timestamp_timezone = "UTC"
    timestamp_quality = "ISO-8601 with microseconds and explicit offset (published_at). Snippet, no full text."

    def is_configured(self):
        return bool(get_settings().marketaux_api_key)

    def search(self, query, start=None, end=None, limit=50):
        params = {"search": query, "language": "en", "countries": "in", "limit": min(limit, 50),
                  "api_token": get_settings().marketaux_api_key}
        if start:
            params["published_after"] = start.strftime("%Y-%m-%dT%H:%M:%S")
        if end:
            params["published_before"] = end.strftime("%Y-%m-%dT%H:%M:%S")
        resp = request_with_retry("GET", "https://api.marketaux.com/v1/news/all", params=params,
                                  rate_key=self.name, min_interval_s=self.min_interval_s)
        # Provider-side sentiment is kept in `raw` for comparison only; our own engine scores.
        return [self.make_article(a.get("source") or "unknown", a.get("title") or "", a.get("url") or "",
                                  parse_timestamp_info(a.get("published_at"), self.timestamp_timezone),
                                  a.get("description") or a.get("snippet"), query,
                                  {"entities": a.get("entities")}, a.get("uuid"))
                for a in resp.json().get("data", [])]
