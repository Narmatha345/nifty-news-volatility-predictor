"""Article-level analysis: relevance -> keywords -> sentiment -> stored NewsAnalysis rows."""
from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.models import NewsAnalysis, NewsArticle
from backend.services.timeutil import utcnow
from backend.system1_news.aggregation import AnalysisItem
from backend.system1_news.availability import effective_available_at, timestamp_precision
from backend.system1_news.keywords import extract_keywords
from backend.system1_news.relevance import CompanyRef, detect_entities
from backend.system1_news.sentiment import SentimentEngine, label_for

# Analyses are stored down to this floor; aggregation applies the (tunable) min_relevance.
STORAGE_RELEVANCE_FLOOR = 0.10
IRRELEVANT = "NONE"


def source_reliability(source: str, table: dict[str, float]) -> float:
    s = (source or "").lower()
    best = None
    for name, val in table.items():
        if name != "default" and name in s and (best is None or len(name) > best[0]):
            best = (len(name), val)
    return best[1] if best else table.get("default", 0.6)


def event_strength(base_strength: float, intensity: float) -> float:
    """Category base strength scaled by how forcefully the article is worded (0.5x .. 1.0x)."""
    return round(base_strength * (0.5 + 0.5 * min(1.0, intensity / 3.0)), 4)


def analyze_article(article: NewsArticle, companies: list[CompanyRef], engine: SentimentEngine,
                    p1: dict) -> list[NewsAnalysis]:
    matches = detect_entities(article.title, article.summary, companies, STORAGE_RELEVANCE_FLOOR)
    now = utcnow()
    if not matches:
        return [NewsAnalysis(news_id=article.id, entity=IRRELEVANT, is_macro=False, category="unrelated",
                             keywords=[], relevance_score=0.0, sentiment_score=0.0, sentiment_label="NEUTRAL",
                             strength=0.0, source_reliability=0.0, engine=engine.name,
                             evidence={"reason": "no Top-10 company or macro topic detected"}, analyzed_at=now)]
    sent = engine.score(article.title, article.summary)
    keywords = extract_keywords(article.title, article.summary)
    rel_src = source_reliability(article.source, p1["source_reliability"])
    rows = []
    for m in matches:
        rows.append(NewsAnalysis(
            news_id=article.id, entity=m.entity, is_macro=m.is_macro, category=m.category, keywords=keywords,
            relevance_score=m.relevance, sentiment_score=sent.score,
            sentiment_label=label_for(sent.score, p1["neutral_band"]),
            strength=event_strength(m.strength, sent.intensity), source_reliability=rel_src,
            engine=engine.name, analyzed_at=now,
            evidence={**sent.evidence, "category_terms": m.matched, "half_life_days": m.half_life_days,
                      "raw": sent.raw, "price_report": m.price_report}))
    return rows


def analyze_pending(session: Session, companies: list[CompanyRef], engine: SentimentEngine, p1: dict,
                    limit: int | None = None) -> int:
    done = select(NewsAnalysis.news_id).where(NewsAnalysis.engine == engine.name)
    q = select(NewsArticle).where(NewsArticle.id.not_in(done)).order_by(NewsArticle.published_at)
    if limit:
        q = q.limit(limit)
    n = 0
    for art in session.scalars(q):
        for row in analyze_article(art, companies, engine, p1):
            session.add(row)
        n += 1
    session.flush()
    return n


def load_items(session: Session, engine_name: str, until: datetime | None = None,
               since: datetime | None = None) -> list[AnalysisItem]:
    """Analysis items with point-in-time availability.

    `published_at` on the returned items is the EFFECTIVE availability time
    (availability.effective_available_at: the earlier of what the provider timestamp allows - hour
    stamps from the end of the hour, date-only / unknown-timezone stamps from the end of the date - and
    fetched_at). `until` / `since` filter on it. The raw facts (provider stamp, precision, timezone
    status, fetched_at) are kept so availability.eligibility() can be applied per cut-off.
    """
    q = (select(NewsAnalysis, NewsArticle).join(NewsArticle, NewsAnalysis.news_id == NewsArticle.id)
         .where(NewsAnalysis.engine == engine_name).where(NewsAnalysis.entity != IRRELEVANT))
    if until is not None:   # coarse stamps shift forward by < 1 day: filter after computing below
        q = q.where(NewsArticle.published_at <= until)
    if since is not None:
        q = q.where(NewsArticle.published_at >= since - timedelta(days=1))
    items = []
    for an, art in session.execute(q):
        precision = art.timestamp_precision if art.timestamp_precision not in (None, "coarse")             else timestamp_precision(art.published_at, art.provider)
        tz_status = art.timezone_status or ("explicit" if art.provider == "google_news_rss" else "unknown")
        available = effective_available_at(art.published_at, precision, art.fetched_at, tz_status)
        if (until is not None and available > until) or (since is not None and available < since):
            continue
        ev = an.evidence or {}
        items.append(AnalysisItem(
            news_id=art.id, cluster_id=art.duplicate_of or art.id, entity=an.entity, is_macro=an.is_macro,
            category=an.category, relevance=an.relevance_score, sentiment=an.sentiment_score,
            strength=an.strength, reliability=an.source_reliability,
            half_life_days=float(ev.get("half_life_days", 5)),
            published_at=available, source=art.source, title=art.title,
            fetched_at=art.fetched_at, price_report=bool(ev.get("price_report", False)),
            availability=art.availability_status or "unknown", provider_published_at=art.published_at,
            timestamp_precision=precision, timezone_status=tz_status, engine=an.engine,
            event_type=an.category))
    return items
