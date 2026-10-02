"""News availability integrity: WHEN could an article have been used for a prediction?

Two timestamps exist per article:
  published_at  what the provider says, with a PRECISION and a TIMEZONE STATUS
  fetched_at    when THIS system stored it (hard evidence we had it)

Timestamp precision (stored on every article; never upgraded):
  exact      seconds present
  minute     hh:mm:00
  hour       hh:00:00 (treated as "some time within that hour")
  date_only  a date without time (Google News RSS renders these as 07:00:00 GMT)
  unknown    unparseable / no timestamp
Timezone status: explicit (offset in the value) | provider_documented | unknown.

Effective availability time (the earliest moment the article may be used):
  * fetched_at, if earlier than anything the provider timestamp allows - we provably held it then;
  * exact / minute with a known timezone: published_at;
  * hour: published_at + 1 h (end of the stated hour, conservative);
  * date_only, or unknown timezone: end of the stated UTC date (next 00:00 UTC = 05:30 IST).
    This is only an ordering convention for research replays: such items are never VERIFIED.

PER-PREDICTION ELIGIBILITY for a session with cut-off C (09:15 IST):
  verified           fetched_at <= C                                   -> usable live and in backtests
  timestamp_before   exact/minute/hour stamp with known tz, available <= C, fetched later
                                                                       -> usable only in research replays
  post_cutoff        provider stamp at/after C (it belongs to a later session)
  uncertain          date_only / unknown precision / unknown timezone, fetched after C
Future timestamps are rejected at collection; missing timestamps are rejected and counted.

Article-level status, relative to S = the first session whose cut-off is at/after the article's
effective availability time (the first session it can be used for):
  verified_pre_market      fetched_at <= open(S)
  published_before_cutoff  precise stamp, fetched during session S (near-real-time, not provable pre-open)
  collected_after_event    precise stamp, fetched after S closed (retrospective collection)
  unknown                  date_only / unknown precision or timezone, not verified
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database.models import FetchLog, NewsAnalysis, NewsArticle, PredictionRun
from backend.services.calendar import TradingCalendar, get_calendar, prediction_session
from backend.services.timeutil import market_close_utc, market_open_utc, to_ist_naive

STATUSES = ("verified_pre_market", "published_before_cutoff", "collected_after_event", "unknown")
PRECISIONS = ("exact", "minute", "hour", "date_only", "unknown")
PRECISE = ("exact", "minute", "hour")
ELIGIBILITY = ("verified", "timestamp_before", "post_cutoff", "uncertain")
# Which eligibility classes each availability policy may use.
POLICIES = {
    "verified": ("verified",),                                  # live predictions, true live-news test
    "exact_timestamp": ("verified", "timestamp_before"),        # research: precise provider stamps
    "provider_timestamp": ("verified", "timestamp_before", "uncertain"),  # research: everything (Phase 2)
}


def timestamp_precision(published_at: datetime, provider: str | None = None) -> str:
    """Precision inferred from a STORED timestamp (used for rows collected before Phase 3, whose raw
    value was not kept). Google News RSS 07:00:00 stamps are date-only (see google_news_rss.py)."""
    if published_at is None:
        return "unknown"
    if provider == "google_news_rss" and (published_at.hour, published_at.minute, published_at.second,
                                          published_at.microsecond) == (7, 0, 0, 0):
        return "date_only"
    if published_at.second or published_at.microsecond:
        return "exact"
    return "minute" if published_at.minute else "hour"


def provider_available_at(published_at: datetime, precision: str | None, tz_status: str | None = "explicit"
                          ) -> datetime:
    """Earliest use time implied by the provider timestamp ALONE (conservative)."""
    precision = precision or timestamp_precision(published_at)
    if precision in ("date_only", "unknown") or tz_status == "unknown":
        return datetime.combine(published_at.date() + timedelta(days=1), datetime.min.time())
    if precision == "hour":
        return published_at + timedelta(hours=1)
    return published_at


def effective_available_at(published_at: datetime, precision: str | None = None,
                           fetched_at: datetime | None = None, tz_status: str | None = "explicit") -> datetime:
    t = provider_available_at(published_at, precision, tz_status)
    return min(t, fetched_at) if fetched_at is not None else t


def is_precise(precision: str | None, tz_status: str | None) -> bool:
    return precision in PRECISE and tz_status != "unknown"


def eligibility(published_at: datetime, precision: str | None, tz_status: str | None,
                fetched_at: datetime | None, cutoff: datetime) -> str:
    """Can this article inform a prediction whose cut-off is `cutoff`? (see module docstring)"""
    if fetched_at is not None and fetched_at <= cutoff:
        return "verified"
    if is_precise(precision, tz_status):
        return "timestamp_before" if provider_available_at(published_at, precision, tz_status) <= cutoff \
            else "post_cutoff"
    if provider_available_at(published_at, precision, tz_status) > cutoff:
        return "post_cutoff"          # even the conservative reading puts it after the cut-off
    return "uncertain"


def first_usable_session(published_at: datetime, fetched_at: datetime | None, cal: TradingCalendar,
                         precision: str | None = None, tz_status: str | None = "explicit") -> date | None:
    try:
        return prediction_session(effective_available_at(published_at, precision, fetched_at, tz_status), cal)
    except ValueError:
        return None


def classify(published_at: datetime, fetched_at: datetime | None, cal: TradingCalendar,
             precision: str | None = None, tz_status: str | None = "explicit") -> str:
    precision = precision or timestamp_precision(published_at)
    if fetched_at is None:
        return "unknown"
    s = first_usable_session(published_at, fetched_at, cal, precision, tz_status)
    if s is None:
        return "unknown"
    if fetched_at <= market_open_utc(s):
        return "verified_pre_market"
    if not is_precise(precision, tz_status):
        return "unknown"
    if fetched_at <= market_close_utc(s):
        return "published_before_cutoff"
    return "collected_after_event"


def annotate(a: NewsArticle, cal: TradingCalendar) -> None:
    """(Re)derive every availability field of one article from its stored facts."""
    if not a.timestamp_precision or a.timestamp_precision == "coarse":     # legacy value
        a.timestamp_precision = timestamp_precision(a.published_at, a.provider)
    if a.timezone_status is None:
        # Rows stored before Phase 3: every adapter in use (Google News RSS) states GMT explicitly.
        a.timezone_status = "explicit" if a.provider == "google_news_rss" else "unknown"
        a.published_timezone = a.published_timezone or ("GMT" if a.provider == "google_news_rss" else None)
    a.availability_status = classify(a.published_at, a.fetched_at, cal, a.timestamp_precision, a.timezone_status)
    a.first_usable_session = first_usable_session(a.published_at, a.fetched_at, cal, a.timestamp_precision,
                                                  a.timezone_status)
    a.published_at_ist = to_ist_naive(a.published_at)
    a.fetched_at_ist = to_ist_naive(a.fetched_at)


def audit(session: Session, cal: TradingCalendar | None = None, recompute: bool = False) -> dict:
    """Fill availability fields (only rows missing them unless `recompute`)."""
    cal = cal or get_calendar()
    q = select(NewsArticle)
    if not recompute:
        q = q.where((NewsArticle.availability_status.is_(None)) | (NewsArticle.first_usable_session.is_(None))
                    | (NewsArticle.timestamp_precision == "coarse"))
    n = 0
    for a in session.scalars(q):
        annotate(a, cal)
        n += 1
    session.flush()
    counts = Counter(s for (s,) in session.execute(select(NewsArticle.availability_status)))
    return {"updated": n, "by_status": {k: counts.get(k, 0) for k in STATUSES}}


# ------------------------------------------------------------------------------- audit report
def _share(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def _quality_block(rows: list) -> dict:
    n = len(rows)
    prec = Counter(r.timestamp_precision or "unknown" for r in rows)
    st = Counter(r.availability_status or "unknown" for r in rows)
    return {
        "articles": n,
        "duplicates": sum(r.duplicate_of is not None for r in rows),
        "duplicate_rate": _share(sum(r.duplicate_of is not None for r in rows), n),
        "precision": {p: prec.get(p, 0) for p in PRECISIONS},
        "precision_share": {p: _share(prec.get(p, 0), n) for p in PRECISIONS},
        "timezone_unknown": sum(r.timezone_status == "unknown" for r in rows),
        "status": {s: st.get(s, 0) for s in STATUSES},
        "verified_pre_market_share": _share(st.get("verified_pre_market", 0), n),
        "unknown_share": _share(st.get("unknown", 0), n),
        "post_event_share": _share(st.get("collected_after_event", 0), n),
    }


def audit_report(session: Session, start: date | None = None, end: date | None = None,
                 cal: TradingCalendar | None = None) -> dict:
    """Overall / provider / company / trading-date availability report (articles by first usable session)."""
    from backend.system1_news.providers import REGISTRY
    from backend.system1_news.sentiment import get_engine
    from backend.config.settings import get_settings

    cal = cal or get_calendar()
    audit(session, cal)
    q = select(NewsArticle)
    if start:
        q = q.where(NewsArticle.first_usable_session >= start)
    if end:
        q = q.where(NewsArticle.first_usable_session <= end)
    arts = list(session.scalars(q))
    ids = {a.id for a in arts}
    by_provider: dict[str, list] = defaultdict(list)
    for a in arts:
        by_provider[a.provider].append(a)
    logs = session.execute(select(FetchLog.provider, FetchLog.rejected_future, FetchLog.rejected_missing_timestamp,
                                  FetchLog.rejected_invalid, FetchLog.duplicates, FetchLog.items)).all()
    rej: dict[str, Counter] = defaultdict(Counter)
    for prov, fut, miss, inv, dup, items in logs:
        rej[prov].update({"future": fut or 0, "missing_timestamp": miss or 0, "invalid": inv or 0,
                          "duplicates_at_collection": dup or 0, "fetched": items or 0})
    providers = {}
    for prov in sorted(set(by_provider) | set(rej)):
        cls = REGISTRY.get(prov)
        providers[prov] = {**_quality_block(by_provider.get(prov, [])),
                           "rejected_at_collection": dict(rej.get(prov, {})),
                           "timestamp_quality": getattr(cls, "timestamp_quality", "unknown provider"),
                           "documented_timezone": getattr(cls, "timestamp_timezone", None)}
    engine = get_engine(get_settings().sentiment_engine).name
    by_company: dict[str, Counter] = defaultdict(Counter)
    if ids:
        for ent, st in session.execute(select(NewsAnalysis.entity, NewsArticle.availability_status)
                                       .join(NewsArticle, NewsAnalysis.news_id == NewsArticle.id)
                                       .where(NewsAnalysis.engine == engine, NewsAnalysis.entity != "NONE",
                                              NewsArticle.id.in_(ids))):
            by_company[ent][st or "unknown"] += 1
            by_company[ent]["total"] += 1
    runs = {r for (r,) in session.execute(select(PredictionRun.target_session)
                                          .where(PredictionRun.is_backtest.is_(False),
                                                 PredictionRun.target_session.is_not(None)))}
    # Runs from before Phase 3 have no target_session column value: derive it (read-only) from their
    # next-session predictions instead of rewriting the stored run.
    from backend.database.models import Prediction
    runs |= {t for (t,) in session.execute(select(Prediction.target_date).join(PredictionRun)
                                           .where(PredictionRun.is_backtest.is_(False),
                                                  PredictionRun.target_session.is_(None),
                                                  Prediction.horizon_type == "today").distinct())}
    by_date: dict[date, Counter] = defaultdict(Counter)
    for a in arts:
        if a.first_usable_session:
            by_date[a.first_usable_session][a.availability_status or "unknown"] += 1
    sessions = sorted(by_date)
    return {
        "generated_for": {"start": str(start) if start else None, "end": str(end) if end else None,
                          "engine": engine},
        "overall": {**_quality_block(arts),
                    "rejected_future": sum(c["future"] for c in rej.values()),
                    "rejected_missing_timestamp": sum(c["missing_timestamp"] for c in rej.values()),
                    "rejected_note": "rejections are counted from the fetch log (Phase 3 onwards; earlier "
                                     "collections did not record them)"},
        "by_provider": providers,
        "by_company": {e: {"total": c["total"], **{s: c.get(s, 0) for s in STATUSES}}
                       for e, c in sorted(by_company.items())},
        "by_trading_date": [{"session": str(d), **{s: by_date[d].get(s, 0) for s in STATUSES},
                             "live_prediction_generated": d in runs} for d in sessions],
    }
