"""System 2 entry point: per-company, per-horizon predicted % movement with full traceability."""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd
from sqlalchemy.orm import Session

from backend.database.models import Prediction, PredictionRun
from backend.services.timeutil import iso_z, utcnow
from backend.system1_news.aggregation import NewsFlow, nifty_relative
from backend.system1_news.pipeline import CompanyInfo, HorizonScores
from backend.system2_prediction.features import (
    MissingData, macro_sensitivity, market_features, prediction_features,
)
from backend.system2_prediction.linear import LinearModel
from backend.system2_prediction.model import direction_for, model_version, predict


@dataclass
class PredictionRow:
    ticker: str
    company: str
    horizon_type: str
    horizon_label: str
    target_date: date
    trading_days_ahead: int
    status: str
    previous_close: float | None = None
    previous_close_date: date | None = None
    sentiment_score: float | None = None
    predicted_movement: float | None = None
    predicted_price: float | None = None
    uncertainty_1sigma_pct: float | None = None
    predicted_direction: str = "MISSING"
    missing_reason: str | None = None
    features: dict = field(default_factory=dict)
    news_inputs: list = field(default_factory=list)   # snapshot of the news behind the prediction
    data_quality_flag: str | None = None              # VERIFIED | UNCERTAIN | NO_NEW_NEWS
    contributions: dict = field(default_factory=dict)
    evidence_news_ids: list = field(default_factory=list)


def build_predictions(scores: list[HorizonScores], companies: list[CompanyInfo],
                      closes: dict[str, pd.Series], index_closes: pd.Series | None,
                      base_date: date, p2: dict, feature_cache: dict | None = None,
                      expected_previous_date: date | None = None,
                      flows: dict[str, NewsFlow] | None = None) -> list[PredictionRow]:
    """Pure. `closes` keyed by ticker. `base_date` = the session being predicted for 'today'
    (its previous close is the base price for all horizons). `feature_cache` lets the backtester
    reuse market features across tuning trials (they do not depend on tuned parameters).
    `flows` = System 1 relative-sentiment news flow (required by factor-v2 and the linear model;
    without it relative inputs are treated as missing)."""
    rows: list[PredictionRow] = []
    feats: dict[str, object] = {}
    for c in companies:
        key = (c.ticker, base_date)
        if feature_cache is not None and key in feature_cache:
            feats[c.ticker] = feature_cache[key]
            continue
        try:
            feats[c.ticker] = market_features(closes.get(c.ticker, pd.Series(dtype=float)), index_closes,
                                              base_date, p2, expected_previous_date)
        except MissingData as exc:
            feats[c.ticker] = exc
        if feature_cache is not None:
            feature_cache[key] = feats[c.ticker]
    flows = flows or {}
    weights = {c.ticker: c.nifty_weight for c in companies}
    nifty_rel = nifty_relative(flows, weights) if flows else None
    relative = p2.get("sentiment_input", "raw") == "relative"
    linear = LinearModel.from_dict(p2["linear"]) if p2.get("model") == "linear" else None
    today = next((hs for hs in scores if hs.horizon.horizon_type == "today"), scores[0] if scores else None)
    for hs in scores:
        h = hs.horizon
        for c in companies:
            agg = hs.companies[c.ticker]
            f = feats[c.ticker]
            fc, fm = flows.get(c.ticker), flows.get("MACRO")
            row = PredictionRow(c.ticker, c.name, h.horizon_type, h.label, h.target_date,
                                h.trading_days_ahead, "OK", sentiment_score=agg.score,
                                evidence_news_ids=[x["news_id"] for x in agg.contributors]
                                + [x["news_id"] for x in hs.macro.contributors])
            if isinstance(f, MissingData):
                row.status, row.missing_reason = "MISSING_DATA", str(f)
                rows.append(row)
                continue
            m_sens = macro_sensitivity(c.sector, hs.macro)
            # The model feature vector is anchored on the next-session ("today") scores so the same
            # vector is used for every horizon (and matches what the linear model was fitted on).
            fv = prediction_features(today.companies[c.ticker].score, fc, fm, nifty_rel, today.macro.score,
                                     today.nifty["company_weighted"], m_sens, f)
            sigma_h = f.daily_vol * math.sqrt(max(1, h.trading_days_ahead)) * 100
            if linear is not None:
                one_day = linear.predict(fv)
                # h = 1: fitted next-session return. h > 1: the one-off news/feature shock only - the
                # intercept is a trailing mean return and extrapolating it is noise (cf. drift_weight).
                pct = one_day if h.trading_days_ahead <= 1 else one_day - linear.intercept
                cap = p2["max_sigma_clip"] * sigma_h
                pct = max(-cap, min(cap, pct))
                row.predicted_movement = round(pct, 4)
                row.predicted_price = round(f.previous_close * (1 + pct / 100), 2)
                row.uncertainty_1sigma_pct = round(sigma_h, 4)
                row.predicted_direction = direction_for(pct, sigma_h, p2)
                row.contributions = dict(linear.contributions(fv))
                row.contributions["intercept"] = round(linear.intercept if h.trading_days_ahead <= 1 else 0.0, 5)
            else:
                if relative:
                    s_c = fc.relative_sentiment if fc and fc.relative_sentiment is not None else 0.0
                    s_m = fm.relative_sentiment if fm and fm.relative_sentiment is not None else 0.0
                    s_n = nifty_rel or 0.0
                else:
                    s_c, s_m, s_n = agg.score, hs.macro.score, hs.nifty["company_weighted"]
                out = predict(f.previous_close, f.daily_vol, f.daily_drift, f.beta, h.trading_days_ahead,
                              s_c, s_m, m_sens, s_n, p2)
                row.predicted_movement, row.predicted_price = out.predicted_movement_pct, out.estimated_price
                row.uncertainty_1sigma_pct, row.predicted_direction = out.uncertainty_1sigma_pct, out.direction
                row.contributions = out.contributions_pct
            if fc is not None:
                row.evidence_news_ids = list(dict.fromkeys(fc.news_ids[:10] + row.evidence_news_ids))
            row.previous_close, row.previous_close_date = f.previous_close, f.previous_close_date
            row.features = {**f.to_dict(), "company_score": agg.score, "company_news_count": agg.news_count,
                            "company_evidence_weight": agg.evidence_weight, "macro_score": hs.macro.score,
                            "macro_sector_sensitivity": round(m_sens, 4),
                            "nifty_company_weighted_score": hs.nifty["company_weighted"],
                            "nifty_overall_score": hs.nifty["overall"],
                            "horizon_trading_days": h.trading_days_ahead,
                            "projected_calendar": h.projected_calendar,
                            "model_inputs": {k: None if v is None else round(float(v), 5) for k, v in fv.items()},
                            "news_flow": None if fc is None else fc.to_dict(),
                            "macro_news_flow": None if fm is None else fm.to_dict(),
                            "sentiment_input": "linear" if linear else ("relative" if relative else "raw")}
            rows.append(row)
    return rows


def persist_predictions(session: Session, rows: list[PredictionRow], as_of: datetime, param_version: str,
                        sentiment_run_id: str, is_backtest: bool = False,
                        backtest_run_id: str | None = None, p2: dict | None = None,
                        run_meta: dict | None = None) -> tuple[PredictionRun, list[Prediction]]:
    """Writes new, immutable rows. Outcomes (actual_results) and LLM reviews (llm_validations) are
    stored in their own tables; a stored prediction is never modified afterwards."""
    run = PredictionRun(id=f"p-{uuid.uuid4().hex[:12]}", created_at=utcnow(), as_of=as_of,
                        param_version=param_version, model_version=model_version(param_version, p2),
                        sentiment_run_id=sentiment_run_id, is_backtest=is_backtest,
                        backtest_run_id=backtest_run_id, **(run_meta or {}))
    session.add(run)
    objs = []
    for r in rows:
        p = Prediction(run_id=run.id, ticker=r.ticker, company=r.company, prediction_timestamp=as_of,
                       horizon_type=r.horizon_type, horizon_label=r.horizon_label, target_date=r.target_date,
                       trading_days_ahead=r.trading_days_ahead, previous_close_date=r.previous_close_date,
                       previous_close=r.previous_close, sentiment_score=r.sentiment_score,
                       predicted_movement=r.predicted_movement, predicted_price=r.predicted_price,
                       uncertainty_1sigma_pct=r.uncertainty_1sigma_pct, predicted_direction=r.predicted_direction,
                       status=r.status, missing_reason=r.missing_reason, features=r.features,
                       contributions=r.contributions, evidence_news_ids=r.evidence_news_ids,
                       model_version=run.model_version, param_version=param_version,
                       news_inputs=r.news_inputs or None, data_quality_flag=r.data_quality_flag)
        session.add(p)
        objs.append(p)
    session.flush()
    return run, objs


def prediction_to_dict(p: Prediction, include_inputs: bool = True) -> dict:
    """`include_inputs=False` omits the (large) per-prediction news snapshot; it stays stored."""
    return {
        "prediction_id": p.id, "run_id": p.run_id, "ticker": p.ticker, "company": p.company,
        "prediction_timestamp": iso_z(p.prediction_timestamp), "horizon_type": p.horizon_type,
        "horizon_label": p.horizon_label, "target_date": p.target_date.isoformat(),
        "trading_days_ahead": p.trading_days_ahead,
        "previous_close_date": p.previous_close_date.isoformat() if p.previous_close_date else None,
        "previous_close": None if p.previous_close is None else round(p.previous_close, 2),
        "sentiment_score": p.sentiment_score,
        "predicted_movement_percent": p.predicted_movement, "predicted_direction": p.predicted_direction,
        "estimated_price": p.predicted_price, "estimated_price_note": "model-derived estimate, not a quote",
        "uncertainty_1sigma_pct": p.uncertainty_1sigma_pct, "status": p.status,
        "missing_reason": p.missing_reason, "features": p.features, "contributions_pct": p.contributions,
        "evidence_news_ids": p.evidence_news_ids, "model_version": p.model_version,
        "data_quality_flag": p.data_quality_flag,
        "news_inputs": p.news_inputs if include_inputs else None,
        "news_inputs_count": len(p.news_inputs or []),
        "prediction_cutoff": iso_z(p.run.prediction_cutoff) if p.run else None,
        "universe_version": p.run.universe_version if p.run else None,
        "availability_policy": p.run.availability_policy if p.run else None,
        "param_version": p.param_version,
        "actual": None if p.actual is None else {
            "actual_close": p.actual.actual_close, "actual_close_date": p.actual.actual_close_date.isoformat(),
            "actual_movement_percent": p.actual.actual_movement, "error": p.actual.error,
            "actual_direction": p.actual.actual_direction, "abs_error": p.actual.abs_error,
            "squared_error": p.actual.squared_error, "direction_correct": p.actual.direction_correct},
        # LLM reviews are separate records ABOUT the prediction - never part of it.
        "validations": [{
            "validation_status": v.validation_status, "confidence": v.confidence,
            "inconsistencies": v.inconsistencies, "missing_information": v.missing_information,
            "reasoning": v.reasoning, "provider": v.provider, "model": v.model,
            "supporting_factors": v.supporting_factors, "contradicting_factors": v.contradicting_factors,
            "data_quality_issues": v.data_quality_issues, "cutoff_check": v.cutoff_check,
            "request_status": v.request_status, "error_message": v.error_message,
            "created_at": iso_z(v.created_at)} for v in p.validations],
    }
