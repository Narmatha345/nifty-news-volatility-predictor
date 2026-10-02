"""News collection: query every enabled provider, validate, de-duplicate, store raw articles."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.config.loader import taxonomy_config
from backend.database.models import FetchLog, NewsArticle
from backend.services.calendar import TradingCalendar, get_calendar
from backend.services.timeutil import utcnow
from backend.services.universe import active_companies
from backend.system1_news.availability import annotate, timestamp_precision
from backend.system1_news.dedup import find_near_duplicate, title_fingerprint, url_hash
from backend.system1_news.providers import NewsProvider, RawArticle, enabled_providers
from backend.system1_news.providers.base import (
    FutureTimestamp, MissingTimestamp, ValidationError, validate_article,
)

log = logging.getLogger(__name__)


@dataclass
class CollectStats:
    providers: list[str] = field(default_factory=list)
    skipped_providers: list[str] = field(default_factory=list)
    fetched: int = 0
    new: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    invalid: int = 0
    rejected_future: int = 0
    rejected_missing_timestamp: int = 0
    errors: list[str] = field(default_factory=list)


def build_queries(session: Session) -> list[str]:
    queries = [f'"{c.aliases[0] if c.aliases else c.name}"' for c in active_companies(session)]
    return queries + list(taxonomy_config().get("macro_queries", []))


def store_articles(session: Session, articles: list[RawArticle], stats: CollectStats,
                   backfilled: bool = False, cal: TradingCalendar | None = None) -> list[NewsArticle]:
    stored: list[NewsArticle] = []
    cal = cal or get_calendar()
    for raw in articles:
        stats.fetched += 1
        now = utcnow()
        try:
            a = validate_article(raw, now)
        except FutureTimestamp as exc:
            stats.rejected_future += 1
            log.info("rejected future-dated article from %s: %s", raw.provider, exc)
            continue
        except MissingTimestamp:
            stats.rejected_missing_timestamp += 1
            continue
        except ValidationError as exc:
            stats.invalid += 1
            log.debug("invalid article from %s: %s", raw.provider, exc)
            continue
        h = url_hash(a.url)
        if session.scalar(select(NewsArticle.id).where(NewsArticle.url_hash == h)) is not None:
            stats.exact_duplicates += 1
            continue
        window = timedelta(hours=48)
        candidates = session.execute(
            select(NewsArticle.id, NewsArticle.title_fingerprint, NewsArticle.published_at)
            .where(NewsArticle.published_at.between(a.published_at - window, a.published_at + window))
            .where(NewsArticle.duplicate_of.is_(None))
        ).all()
        dup = find_near_duplicate(a.title, a.published_at, [tuple(r) for r in candidates])
        if dup is not None:
            stats.near_duplicates += 1
        # Adapters report precision; records from adapters that do not (tests, custom providers)
        # fall back to the generic rule. Never upgraded beyond what the provider gave.
        precision = a.timestamp_precision if a.timestamp_precision != "unknown" \
            else timestamp_precision(a.published_at, a.provider)
        row = NewsArticle(source=a.source, provider=a.provider, title=a.title, url=a.url, url_hash=h,
                          title_fingerprint=title_fingerprint(a.title), published_at=a.published_at,
                          fetched_at=now, summary=a.summary, query=a.query, backfilled=backfilled,
                          article_text=a.article_text, provider_article_id=a.provider_article_id,
                          published_raw=a.published_raw, published_timezone=a.published_timezone,
                          timezone_status=a.timezone_status, timestamp_precision=precision,
                          duplicate_of=dup, raw=a.raw)
        annotate(row, cal)
        session.add(row)
        session.flush()
        stored.append(row)
        stats.new += 1
    return stored


def collect(session: Session, start: datetime | None = None, end: datetime | None = None,
            providers: list[NewsProvider] | None = None, queries: list[str] | None = None,
            limit_per_query: int = 50) -> CollectStats:
    """Live collection (start/end None) or historical backfill (start/end set, UTC)."""
    stats = CollectStats()
    if providers is None:
        providers, skipped = enabled_providers()
        stats.skipped_providers = skipped
    backfilled = start is not None
    if backfilled:
        providers = [p for p in providers if p.supports_history]
    stats.providers = [p.name for p in providers]
    queries = queries or build_queries(session)
    for p in providers:
        for q in queries:
            log_row = FetchLog(provider=p.name, query=q, started_at=utcnow(), status="ok")
            try:
                arts = p.search(q, start=start, end=end, limit=limit_per_query)
                before = (stats.new, stats.exact_duplicates + stats.near_duplicates, stats.rejected_future,
                          stats.rejected_missing_timestamp, stats.invalid)
                store_articles(session, arts, stats, backfilled=backfilled)
                log_row.items, log_row.new_items = len(arts), stats.new - before[0]
                log_row.duplicates = stats.exact_duplicates + stats.near_duplicates - before[1]
                log_row.rejected_future = stats.rejected_future - before[2]
                log_row.rejected_missing_timestamp = stats.rejected_missing_timestamp - before[3]
                log_row.rejected_invalid = stats.invalid - before[4]
            except Exception as exc:  # one failing provider/query must not stop collection
                log_row.status, log_row.error = "error", str(exc)[:1000]
                stats.errors.append(f"{p.name} [{q}]: {exc}")
                log.warning("provider %s failed for %r: %s", p.name, q, exc)
            session.add(log_row)
            session.commit()
    return stats
