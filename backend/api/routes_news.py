"""System 1 endpoints: /news, /news/analyze, /sentiment, /sentiment/aggregate."""
from __future__ import annotations

from datetime import date, datetime

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.api.schemas import parse_dt
from backend.config.settings import get_settings
from backend.database.db import get_db
from backend.database.models import FetchLog, NewsAnalysis, NewsArticle, SentimentSnapshot
from backend.services import orchestrator, parameters as params_svc
from backend.services.timeutil import iso_z
from backend.services.universe import active_companies, to_ref
from backend.system1_news.analyzer import IRRELEVANT
from backend.system1_news.keywords import extract_keywords
from backend.system1_news.relevance import detect_entities
from backend.system1_news.sentiment import get_engine, label_for

router = APIRouter()


class CollectRequest(BaseModel):
    start: str | None = None   # ISO datetime/date (UTC) for historical backfill
    end: str | None = None


class AnalyzeTextRequest(BaseModel):
    title: str
    summary: str | None = None


@router.get("/news")
def list_news(ticker: str | None = None, start: str | None = None, end: str | None = None,
              include_irrelevant: bool = False, limit: int = Query(100, le=1000), db: Session = Depends(get_db)):
    engine = get_engine(get_settings().sentiment_engine).name
    q = select(NewsArticle).order_by(NewsArticle.published_at.desc())
    current = select(NewsAnalysis.news_id).where(NewsAnalysis.engine == engine)
    if start:
        q = q.where(NewsArticle.published_at >= parse_dt(start))
    if end:
        q = q.where(NewsArticle.published_at <= parse_dt(end))
    if ticker:
        q = q.where(NewsArticle.id.in_(current.where(NewsAnalysis.entity == ticker)))
    elif not include_irrelevant:
        q = q.where(NewsArticle.id.in_(current.where(NewsAnalysis.entity != IRRELEVANT)))
    out = []
    for a in db.scalars(q.limit(limit)):
        out.append({
            "id": a.id, "source": a.source, "provider": a.provider, "title": a.title, "url": a.url,
            "published_at": iso_z(a.published_at), "fetched_at": iso_z(a.fetched_at), "summary": a.summary,
            "backfilled": a.backfilled, "duplicate_of": a.duplicate_of,
            "timestamp_precision": a.timestamp_precision, "availability_status": a.availability_status,
            "timezone_status": a.timezone_status, "published_timezone": a.published_timezone,
            "published_raw": a.published_raw, "published_at_ist": a.published_at_ist.isoformat()
            if a.published_at_ist else None, "fetched_at_ist": a.fetched_at_ist.isoformat() if a.fetched_at_ist
            else None, "first_usable_session": str(a.first_usable_session) if a.first_usable_session else None,
            "description": a.summary, "article_text": a.article_text,
            "provider_article_id": a.provider_article_id,
            "analysis": [{"entity": an.entity, "is_macro": an.is_macro, "category": an.category,
                          "keywords": an.keywords, "relevance_score": an.relevance_score,
                          "sentiment_score": an.sentiment_score, "sentiment_label": an.sentiment_label,
                          "strength": an.strength, "source_reliability": an.source_reliability,
                          "engine": an.engine, "evidence": an.evidence}
                         for an in a.analyses if an.engine == engine],
        })
    return {"count": len(out), "items": out}


@router.post("/news/collect")
def collect_news(req: CollectRequest, db: Session = Depends(get_db)):
    return orchestrator.collect_and_analyze(db, parse_dt(req.start) if req.start else None,
                                            parse_dt(req.end) if req.end else None)


@router.get("/news/fetch-log")
def fetch_log(limit: int = 100, db: Session = Depends(get_db)):
    rows = db.scalars(select(FetchLog).order_by(FetchLog.started_at.desc()).limit(limit))
    return [{"provider": r.provider, "query": r.query, "started_at": iso_z(r.started_at), "status": r.status,
             "items": r.items, "new_items": r.new_items, "error": r.error} for r in rows]


@router.post("/news/analyze")
def analyze(req: AnalyzeTextRequest | None = None, db: Session = Depends(get_db)):
    """Without a body: analyze all stored, not-yet-analyzed articles.
    With {title, summary}: analyze ad-hoc text (not stored) and return the breakdown."""
    if req is None:
        return {"analyzed": orchestrator.analyze_news(db)}
    _, params = params_svc.get_active(db)
    refs = [to_ref(c) for c in active_companies(db)]
    sent = get_engine(get_settings().sentiment_engine).score(req.title, req.summary)
    ents = detect_entities(req.title, req.summary, refs, params["system1"]["min_relevance"])
    return {"sentiment_score": sent.score, "sentiment_label": label_for(sent.score, params["system1"]["neutral_band"]),
            "engine": sent.engine, "evidence": sent.evidence,
            "keywords": extract_keywords(req.title, req.summary),
            "entities": [{"entity": e.entity, "is_macro": e.is_macro, "relevance": e.relevance,
                          "category": e.category, "strength": e.strength, "matched": e.matched,
                          "price_report": e.price_report} for e in ents],
            "relevant": bool(ents)}


@router.get("/sentiment")
def latest_sentiment(db: Session = Depends(get_db)):
    """Latest stored System 1 output (all horizons)."""
    last = db.scalar(select(SentimentSnapshot.run_id).where(SentimentSnapshot.is_backtest.is_(False))
                     .order_by(SentimentSnapshot.as_of.desc(), SentimentSnapshot.id.desc()).limit(1))
    if last is None:
        return {"status": "MISSING", "message": "No sentiment computed yet. POST /sentiment/aggregate."}
    rows = db.scalars(select(SentimentSnapshot).where(SentimentSnapshot.run_id == last)).all()
    horizons: dict[tuple, dict] = {}
    for r in rows:
        h = horizons.setdefault((r.target_date, r.horizon_type), {
            "horizon_type": r.horizon_type, "label": r.horizon_label, "target_date": r.target_date.isoformat(),
            "company_scores": [], "macro_score": None, "nifty_company_weighted_score": None, "nifty_score": None})
        if r.entity == "MACRO":
            h["macro_score"], h["macro_news_count"], h["macro_details"] = r.score, r.news_count, r.details
            h["macro_relative_sentiment"] = (r.details.get("news_flow") or {}).get("relative_sentiment")
        elif r.entity == "NIFTY":
            h["nifty_score"], h["nifty_weight_coverage"] = r.score, r.details.get("weight_coverage")
        elif r.entity == "NIFTY_COMPANY":
            h["nifty_company_weighted_score"] = r.score
        else:
            flow = r.details.get("news_flow") or {}
            h["company_scores"].append({"ticker": r.entity, "sentiment_score": r.score, "news_count": r.news_count,
                                        "evidence_weight": r.evidence_weight, "raw_mean": r.raw_mean,
                                        "relative_sentiment": flow.get("relative_sentiment"),
                                        "sentiment_zscore": flow.get("sentiment_zscore"),
                                        "new_news_count": flow.get("news_count"),
                                        "news_flow_status": flow.get("status"),
                                        "top_contributors": r.details.get("contributors", [])})
    order = {"today": 0, "month_end": 1, "quarter_end": 2}
    return {"run_id": last, "timestamp": iso_z(rows[0].as_of), "param_version": rows[0].param_version,
            "horizons": sorted(horizons.values(), key=lambda h: (order[h["horizon_type"]], h["target_date"]))}


@router.get("/sentiment/history")
def sentiment_history(entity: str = "NIFTY", horizon_type: str = "today", days: int = 90,
                      db: Session = Depends(get_db)):
    """One point per day (the latest live run that day) for charting sentiment over time."""
    rows = db.execute(select(SentimentSnapshot.as_of, SentimentSnapshot.score)
                      .where(SentimentSnapshot.entity == entity, SentimentSnapshot.horizon_type == horizon_type,
                             SentimentSnapshot.is_backtest.is_(False))
                      .order_by(SentimentSnapshot.as_of)).all()
    by_day: dict[date, tuple[datetime, float]] = {}
    for as_of, score in rows:
        by_day[as_of.date()] = (as_of, score)
    pts = [{"date": d.isoformat(), "as_of": iso_z(v[0]), "score": v[1]} for d, v in sorted(by_day.items())]
    return {"entity": entity, "horizon_type": horizon_type, "points": pts[-days:]}


@router.post("/sentiment/aggregate")
def aggregate_now(as_of: str | None = None, db: Session = Depends(get_db)):
    out, *_ = orchestrator.sentiment(db, parse_dt(as_of) if as_of else None)
    return out


_AUDIT_CACHE: dict[tuple, dict] = {}


def _audit_fingerprint(db: Session) -> tuple:
    """Cheap aggregate that changes whenever anything the audit reads changes (articles, their availability
    fields, fetch logs, live runs, sentiment engine)."""
    from backend.database.models import PredictionRun
    arts = db.execute(select(func.count(NewsArticle.id), func.max(NewsArticle.id), func.max(NewsArticle.fetched_at),
                             func.count(NewsArticle.availability_status), func.count(NewsArticle.first_usable_session),
                             func.count(NewsArticle.duplicate_of))).one()
    logs = db.execute(select(func.count(FetchLog.id), func.max(FetchLog.id))).one()
    runs = db.scalar(select(func.count(PredictionRun.id)).where(PredictionRun.is_backtest.is_(False)))
    return (*arts, *logs, runs, get_settings().sentiment_engine)


@router.get("/news/audit")
def news_audit(start: str | None = None, end: str | None = None, db: Session = Depends(get_db)):
    """Availability audit: overall / by provider / by company / by trading date.
    The full report reads every article (~10 s on a small server), so it is cached until the data changes."""
    from backend.api.schemas import parse_date
    from backend.system1_news.availability import audit_report
    key = (start, end, _audit_fingerprint(db))
    if key not in _AUDIT_CACHE:
        report = audit_report(db, parse_date(start) if start else None, parse_date(end) if end else None)
        db.commit()   # audit_report fills missing availability fields; persist them before fingerprinting again
        key = (start, end, _audit_fingerprint(db))
        _AUDIT_CACHE.clear()   # keep only the latest result per process
        _AUDIT_CACHE[key] = report
    return _AUDIT_CACHE[key]


@router.get("/news/stats")
def news_stats(db: Session = Depends(get_db)):
    total = db.scalar(select(func.count(NewsArticle.id)))
    by_entity = dict(db.execute(select(NewsAnalysis.entity, func.count(NewsAnalysis.id))
                                .group_by(NewsAnalysis.entity)).all())
    availability = dict(db.execute(select(NewsArticle.availability_status, func.count(NewsArticle.id))
                                   .group_by(NewsArticle.availability_status)).all())
    precision = dict(db.execute(select(NewsArticle.timestamp_precision, func.count(NewsArticle.id))
                                .group_by(NewsArticle.timestamp_precision)).all())
    return {"articles": total, "analyses_by_entity": by_entity,
            "availability_status": {str(k): v for k, v in availability.items()},
            "timestamp_precision": {str(k): v for k, v in precision.items()}}
