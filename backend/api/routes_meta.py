"""/companies, /parameters, /market-data, /config, /health."""
from __future__ import annotations

import json
from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.api.schemas import parse_date
from backend.config.settings import get_settings
from backend.data.market_data import load_closes
from backend.database.db import get_db
from backend.database.models import Company, MarketData, ParameterChange, ParameterSet
from backend.services import orchestrator, parameters as params_svc
from backend.services.calendar import get_calendar
from backend.services.horizons import generate_horizons, horizons_as_dicts
from backend.services.llm import provider_status
from backend.services.universe import universe_on, universe_report
from backend.services.timeutil import iso_z
from backend.system1_news.providers import enabled_providers

router = APIRouter()


class CompanyIn(BaseModel):
    name: str | None = None
    yahoo_symbol: str | None = None
    sector: str | None = None
    nifty_weight: float | None = None
    weight_as_of: str | None = None
    weight_source: str | None = None
    active_from: str | None = None
    active_to: str | None = None
    aliases: list[str] | None = None
    case_sensitive_aliases: list[str] | None = None
    exclude_patterns: list[str] | None = None


def _company(c: Company) -> dict:
    return {"ticker": c.ticker, "name": c.name, "yahoo_symbol": c.yahoo_symbol, "sector": c.sector,
            "nifty_weight": c.nifty_weight, "weight_as_of": str(c.weight_as_of) if c.weight_as_of else None,
            "weight_source": c.weight_source, "weight_note": "reference data, not a live index weight",
            "active_from": str(c.active_from), "active_to": str(c.active_to) if c.active_to else None,
            "aliases": c.aliases, "case_sensitive_aliases": c.case_sensitive_aliases,
            "exclude_patterns": c.exclude_patterns}


@router.get("/companies")
def companies(catalog: bool = False, on: str | None = None, db: Session = Depends(get_db)):
    """Effective universe on `on` (default today) with the weights effective then; `catalog=true`
    lists every catalogued instrument (membership is NOT implied)."""
    snap = universe_on(db, parse_date(on) if on else None)
    members = {m.ticker: m for m in snap.members}
    rows = db.scalars(select(Company)).all()
    out = []
    for c in rows:
        m = members.get(c.ticker)
        if m is None and not catalog:
            continue
        out.append({**_company(c), "in_universe": m is not None, "universe_version": snap.version,
                    "nifty_weight": m.weight if m else None,
                    "weight_source": m.source if m else None,
                    "weight_as_of": str(m.source_date) if m and m.source_date else None})
    return sorted(out, key=lambda x: -(x["nifty_weight"] or 0))


@router.get("/universe")
def universe(on: str | None = None, db: Session = Depends(get_db)):
    """ACTIVE Top-10 (version, members, weights, effective dates, source), unconfirmed versions with
    the membership changes they would make, and the full version history."""
    return universe_report(db, parse_date(on) if on else None)


@router.put("/companies/{ticker}")
def upsert_company(ticker: str, body: CompanyIn, db: Session = Depends(get_db)):
    c = db.scalar(select(Company).where(Company.ticker == ticker))
    if c is None:
        missing = [f for f in ("name", "yahoo_symbol", "sector", "nifty_weight") if getattr(body, f) is None]
        if missing:
            raise HTTPException(422, f"new company requires {missing}")
        c = Company(ticker=ticker, active_from=date.today(), aliases=[body.name])
        db.add(c)
    for field, value in body.model_dump(exclude_none=True).items():
        if field in ("weight_as_of", "active_from", "active_to"):
            value = parse_date(value)
        setattr(c, field, value)
    db.flush()
    return _company(c)


@router.get("/parameters")
def parameters(db: Session = Depends(get_db)):
    params_svc.ensure_default(db)
    rows = db.scalars(select(ParameterSet).order_by(ParameterSet.created_at.desc()))
    return [{"version": p.version, "status": p.status, "created_at": iso_z(p.created_at),
             "created_by": p.created_by, "parent_version": p.parent_version, "notes": p.notes,
             "activated_at": iso_z(p.activated_at), "params": p.params} for p in rows]


@router.get("/parameters/changes")
def parameter_changes(db: Session = Depends(get_db)):
    rows = db.scalars(select(ParameterChange).order_by(ParameterChange.created_at.desc()))
    return [{"from_version": c.from_version, "to_version": c.to_version, "parameter_name": c.parameter_name,
             "old_value": json.loads(c.old_value), "new_value": json.loads(c.new_value), "reason": c.reason,
             "backtest_run_id": c.backtest_run_id, "backtest_result": c.backtest_result, "applied": c.applied,
             "timestamp": iso_z(c.created_at)} for c in rows]


@router.post("/parameters/{version}/activate")
def activate(version: str, db: Session = Depends(get_db)):
    try:
        ps = params_svc.activate(db, version)
    except KeyError:
        raise HTTPException(404, f"unknown parameter version {version}")
    return {"activated": ps.version, "activated_at": iso_z(ps.activated_at)}


@router.get("/market-data")
def market_data(ticker: str, start: str | None = None, end: str | None = None, db: Session = Depends(get_db)):
    s = load_closes(db, ticker, parse_date(start) if start else None, parse_date(end) if end else None)
    if s.empty:
        return {"ticker": ticker, "status": "MISSING", "bars": []}
    return {"ticker": ticker, "status": "OK", "bars": [{"date": str(d), "close": v} for d, v in s.items()]}


@router.get("/market-data/coverage")
def coverage(db: Session = Depends(get_db)):
    rows = db.execute(select(MarketData.ticker, func.min(MarketData.date), func.max(MarketData.date),
                             func.count(MarketData.id)).group_by(MarketData.ticker)).all()
    return [{"ticker": t, "first": str(a), "last": str(b), "bars": n} for t, a, b, n in rows]


@router.post("/market-data/refresh")
def refresh_market(start: str | None = None, end: str | None = None, db: Session = Depends(get_db)):
    return orchestrator.refresh_market_data(db, parse_date(start) if start else None,
                                            parse_date(end) if end else None)


@router.get("/config")
def config():
    s = get_settings()
    active, skipped = enabled_providers()
    cal = get_calendar()
    from backend.database.db import session_scope
    with session_scope() as db:
        snap = universe_on(db)
        uni = {"universe_version": snap.version, "universe_pending_confirmation": snap.pending}
    return {**uni, "display_timezone": s.display_timezone, "quarter_convention": s.quarter_convention,
            "sentiment_engine": s.sentiment_engine, **provider_status(),
            "news_providers_active": [p.name for p in active], "news_providers_skipped": skipped,
            "calendar_source": cal.source, "calendar_reliable_until": str(cal.reliable_until),
            "horizons_today": horizons_as_dicts(generate_horizons(date.today(), cal, s.quarter_convention))}


@router.get("/health")
def health():
    return {"status": "ok"}
