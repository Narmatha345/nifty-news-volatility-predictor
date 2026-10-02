"""News provider adapter interface. Add a provider = subclass NewsProvider + register it.

Every adapter returns normalised RawArticle records that carry not only a UTC timestamp but what is
actually KNOWN about it (precision, timezone, raw value). Adapters must never invent a time: a
date-only value stays date_only, a missing value stays missing (the collector rejects and counts it).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from backend.services.timeutil import TimestampInfo


@dataclass
class RawArticle:
    """Provider-normalised article, validated before it reaches the database."""
    provider: str
    source: str
    title: str
    url: str
    published_at: datetime | None   # naive UTC from the provider (never invented); None = missing
    summary: str | None = None      # provider description / snippet
    query: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    article_text: str | None = None          # full text only if the provider returns it
    provider_article_id: str | None = None   # stable identifier (RSS guid, API uuid)
    published_raw: str | None = None         # timestamp exactly as received
    published_timezone: str | None = None
    timezone_status: str = "unknown"         # explicit | provider_documented | unknown
    timestamp_precision: str = "unknown"     # exact | minute | hour | date_only | unknown


class ValidationError(ValueError):
    reason = "invalid"


class MissingTimestamp(ValidationError):
    reason = "missing_timestamp"


class FutureTimestamp(ValidationError):
    reason = "future_timestamp"


MAX_FUTURE_SKEW = timedelta(minutes=15)


def validate_article(a: RawArticle, now: datetime | None = None) -> RawArticle:
    if not a.title or not a.title.strip():
        raise ValidationError("missing title")
    if not a.url or not a.url.startswith(("http://", "https://")):
        raise ValidationError(f"invalid url: {a.url!r}")
    if a.published_at is None:
        raise MissingTimestamp("missing published timestamp")
    if now is not None and a.published_at > now + MAX_FUTURE_SKEW:
        # A timestamp in the future is wrong (bad timezone / provider bug) and could leak into a
        # prediction made before the real publication time - reject rather than guess.
        raise FutureTimestamp(f"published timestamp {a.published_at} is in the future (now {now})")
    a.title = a.title.strip()
    a.source = (a.source or "unknown").strip()
    return a


class NewsProvider(ABC):
    name: str = "base"
    requires_key: bool = True
    supports_history: bool = False   # can it return articles for a past date range?
    min_interval_s: float = 1.0      # polite spacing between calls
    # Timezone the provider DOCUMENTS for timestamps without an explicit offset (None = undocumented).
    timestamp_timezone: str | None = None
    # Plain-language label shown in audits: how far can this provider's timestamps be trusted?
    timestamp_quality: str = "unknown"
    provides_article_text: bool = False

    def is_configured(self) -> bool:
        return True

    def classify_precision(self, info: TimestampInfo) -> str:
        """Provider-specific precision rules (override when the provider has known quirks)."""
        return info.precision

    def make_article(self, source: str, title: str, url: str, info: TimestampInfo, summary: str | None,
                     query: str | None, raw: dict | None = None, article_id: str | None = None,
                     article_text: str | None = None) -> RawArticle:
        return RawArticle(provider=self.name, source=source, title=title, url=url, published_at=info.utc,
                          summary=summary, query=query, raw=raw or {}, article_text=article_text,
                          provider_article_id=article_id, published_raw=info.raw,
                          published_timezone=info.timezone, timezone_status=info.tz_status,
                          timestamp_precision=self.classify_precision(info) if info.utc else "unknown")

    @abstractmethod
    def search(self, query: str, start: datetime | None = None, end: datetime | None = None,
               limit: int = 50) -> list[RawArticle]:
        """Return articles matching `query` published in [start, end] (UTC) where supported."""
