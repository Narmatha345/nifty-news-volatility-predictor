"""System 1 entry point: aggregated company / macro / NIFTY scores for every forecast horizon."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from backend.config.settings import get_settings
from backend.database.models import SentimentSnapshot
from backend.services.calendar import TradingCalendar, get_calendar, prediction_session
from backend.services.horizons import Horizon, generate_horizons
from backend.services.timeutil import ist_date, iso_z, market_close_utc, market_open_utc, utcnow
from backend.system1_news.aggregation import (
    AggregateScore, AnalysisItem, NewsFlow, aggregate, baseline_sentiments, news_flow, nifty_relative,
    nifty_scores,
)


@dataclass
class CompanyInfo:
    ticker: str
    name: str
    sector: str
    nifty_weight: float


@dataclass
class HorizonScores:
    horizon: Horizon
    companies: dict[str, AggregateScore]
    macro: AggregateScore
    nifty: dict[str, float]


def group_items(items: list[AnalysisItem]) -> dict[str, list[AnalysisItem]]:
    grouped: dict[str, list[AnalysisItem]] = {}
    for it in items:
        grouped.setdefault(it.entity, []).append(it)
    return grouped


def score_horizons(items_by_entity: dict[str, list[AnalysisItem]], companies: list[CompanyInfo],
                   as_of: datetime, horizons: list[Horizon], p1: dict) -> list[HorizonScores]:
    """Pure: identical inputs give identical outputs (used live and in backtests)."""
    weights = {c.ticker: c.nifty_weight for c in companies}
    out = []
    for h in horizons:
        target = max(as_of, market_close_utc(h.target_date))
        comp = {c.ticker: aggregate(c.ticker, items_by_entity.get(c.ticker, []), as_of, target, p1)
                for c in companies}
        macro = aggregate("MACRO", items_by_entity.get("MACRO", []), as_of, target, p1)
        out.append(HorizonScores(h, comp, macro, nifty_scores(comp, weights, macro, p1)))
    return out


def score_news_flow(items_by_entity: dict[str, list[AnalysisItem]], companies: list[CompanyInfo],
                    window_start: datetime, as_of: datetime, p1: dict) -> dict[str, NewsFlow]:
    """Pure. Relative sentiment of what arrived in (window_start, as_of] for every company + MACRO.
    `window_start` is the previous session's cut-off (its 09:15 IST open)."""
    pooled = [x for c in companies for x in baseline_sentiments(items_by_entity.get(c.ticker, []), window_start, p1)]
    flows = {c.ticker: news_flow(c.ticker, items_by_entity.get(c.ticker, []), window_start, as_of, p1, pooled)
             for c in companies}
    flows["MACRO"] = news_flow("MACRO", items_by_entity.get("MACRO", []), window_start, as_of, p1)
    return flows


def news_window_start(target_session, cal: TradingCalendar) -> datetime:
    """New-news window for a prediction of `target_session` starts at the previous session's open."""
    return market_open_utc(cal.previous_session(target_session))


def to_output(as_of: datetime, scores: list[HorizonScores], companies: list[CompanyInfo],
              param_version: str, run_id: str, calendar_source: str,
              flows: dict[str, NewsFlow] | None = None) -> dict:
    names = {c.ticker: c.name for c in companies}
    weights = {c.ticker: c.nifty_weight for c in companies}
    return {
        "run_id": run_id,
        "timestamp": iso_z(as_of),
        "param_version": param_version,
        "calendar_source": calendar_source,
        "news_flow": None if flows is None else {
            "companies": {t: f.to_dict() for t, f in flows.items() if t != "MACRO"},
            "macro": flows["MACRO"].to_dict(),
            "nifty_relative_sentiment": nifty_relative(flows, weights),
            "definition": "relative_sentiment = mean sentiment of news since the previous session's "
                          "09:15 IST cut-off minus that entity's baseline mean over the preceding window",
        },
        "horizons": [{
            **hs.horizon.to_dict(),
            "company_scores": [{"ticker": t, "company": names[t], "sentiment_score": s.score,
                                "raw_mean": s.raw_mean, "evidence_weight": s.evidence_weight,
                                "news_count": s.news_count, "article_count": s.article_count,
                                "top_contributors": s.contributors} for t, s in hs.companies.items()],
            "macro_score": hs.macro.score,
            "macro_news_count": hs.macro.news_count,
            "macro_by_category": hs.macro.by_category,
            "nifty_company_weighted_score": hs.nifty["company_weighted"],
            "nifty_score": hs.nifty["overall"],
            "nifty_weight_coverage": hs.nifty["weight_coverage"],
        } for hs in scores],
    }


def persist(session: Session, run_id: str, as_of: datetime, scores: list[HorizonScores],
            param_version: str, is_backtest: bool = False, flows: dict[str, NewsFlow] | None = None) -> None:
    for hs in scores:
        h = hs.horizon
        base = dict(run_id=run_id, as_of=as_of, horizon_type=h.horizon_type, horizon_label=h.label,
                    target_date=h.target_date, param_version=param_version, is_backtest=is_backtest)
        for t, s in list(hs.companies.items()) + [("MACRO", hs.macro)]:
            details = s.to_details()
            if flows and t in flows and h.horizon_type == "today":
                details["news_flow"] = flows[t].to_dict()
            session.add(SentimentSnapshot(**base, entity=t, score=s.score, raw_mean=s.raw_mean,
                                          evidence_weight=s.evidence_weight, news_count=s.news_count,
                                          details=details))
        for key, entity in (("company_weighted", "NIFTY_COMPANY"), ("overall", "NIFTY")):
            session.add(SentimentSnapshot(**base, entity=entity, score=hs.nifty[key], raw_mean=None,
                                          evidence_weight=0.0, news_count=0,
                                          details={"weight_coverage": hs.nifty["weight_coverage"]}))
    session.flush()


def run_system1(session: Session, companies: list[CompanyInfo], items: list[AnalysisItem], p1: dict,
                param_version: str, as_of: datetime | None = None, cal: TradingCalendar | None = None,
                save: bool = True) -> tuple[dict, list[HorizonScores], dict[str, NewsFlow]]:
    as_of = as_of or utcnow()
    cal = cal or get_calendar()
    target_session = prediction_session(as_of, cal)
    horizons = generate_horizons(target_session, cal, get_settings().quarter_convention,
                                 calendar_today=ist_date(as_of))
    lookback = as_of - timedelta(days=max(p1["max_lookback_days"], p1.get("baseline_days", 60) + 7))
    items = [i for i in items if lookback <= i.published_at <= as_of]
    grouped = group_items(items)
    scores = score_horizons(grouped, companies, as_of, horizons, p1)
    flows = score_news_flow(grouped, companies, news_window_start(target_session, cal), as_of, p1)
    run_id = f"s1-{uuid.uuid4().hex[:12]}"
    if save:
        persist(session, run_id, as_of, scores, param_version, flows=flows)
    return to_output(as_of, scores, companies, param_version, run_id, cal.source, flows), scores, flows
