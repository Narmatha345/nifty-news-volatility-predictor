"""High-level live operations used by both the API and the CLI."""
from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, timedelta

from collections import Counter

from sqlalchemy.orm import Session

from backend.config.settings import get_settings
from backend.data.market_data import load_closes, refresh
from backend.services import parameters as params_svc
from backend.services.calendar import get_calendar, prediction_session
from backend.system1_news import availability
from backend.services.timeutil import IST, ist_date, iso_z, market_open_utc, utcnow
from backend.services.universe import (
    active_companies, companies_ever_confirmed, company_infos, index_symbol, seed_universe, to_ref,
)
from backend.system1_news.analyzer import analyze_pending, load_items
from backend.system1_news.collector import collect
from backend.system1_news.pipeline import CompanyInfo, run_system1
from backend.system1_news.sentiment import get_engine
from backend.system2_prediction.predictor import build_predictions, persist_predictions, prediction_to_dict


def cutoff_ist(cutoff_utc: datetime) -> str:
    """e.g. '2026-10-05 09:15:00+05:30'."""
    from datetime import timezone
    return cutoff_utc.replace(tzinfo=timezone.utc).astimezone(IST).isoformat(sep=" ")


def bootstrap(session: Session) -> None:
    seed_universe(session)
    params_svc.ensure_default(session)
    availability.audit(session)   # classifies only articles that have no status yet


LIVE_POLICY = "verified"   # live predictions use ONLY news this system held before the cut-off


def _infos(session: Session, on: date | None = None) -> list[CompanyInfo]:
    return company_infos(session, on)[1]


def collect_and_analyze(session: Session, start: datetime | None = None, end: datetime | None = None,
                        queries: list[str] | None = None) -> dict:
    stats = collect(session, start, end, queries=queries)
    analyzed = analyze_news(session)
    return {**asdict(stats), "analyzed": analyzed}


def analyze_news(session: Session) -> int:
    _, params = params_svc.get_active(session)
    engine = get_engine(get_settings().sentiment_engine)
    # every company of every confirmed universe version (not only today's), see universe.py
    refs = [to_ref(c) for c in companies_ever_confirmed(session)]
    return analyze_pending(session, refs, engine, params["system1"])


def refresh_market_data(session: Session, start: date | None = None, end: date | None = None) -> dict:
    start = start or (date.today() - timedelta(days=450))
    # all confirmed members (past ones too): outcomes of earlier predictions still need their prices
    symbols = [c.yahoo_symbol for c in companies_ever_confirmed(session)] + [index_symbol()]
    return refresh(session, symbols, start, end)


def live_inputs(session: Session, as_of: datetime, policy: str = LIVE_POLICY) -> dict:
    """Everything a live System 1 run may see at `as_of`, filtered by the availability policy.

    The target session is the first one whose 09:15 IST open is still ahead; its cut-off is that
    open. Eligibility is judged at `as_of` (<= cut-off), so a replay at a past `as_of` cannot use
    news fetched after that moment either."""
    cal = get_calendar()
    target = prediction_session(as_of, cal)
    cutoff = market_open_utc(target)
    engine = get_engine(get_settings().sentiment_engine)
    loaded = load_items(session, engine.name, until=as_of)
    allowed = availability.POLICIES[policy]
    labels = {id(i): i.eligibility(as_of) for i in loaded}
    items = [i for i in loaded if labels[id(i)] in allowed]
    uversion, infos = company_infos(session, target)
    return {"target": target, "cutoff": cutoff, "items": items, "infos": infos, "universe_version": uversion,
            "engine": engine.name, "policy": policy,
            "summary": {"policy": policy, "loaded_items": len(loaded), "used_items": len(items),
                        "excluded_items": len(loaded) - len(items),
                        "eligibility_of_loaded": dict(Counter(labels.values())),
                        "cutoff_utc": iso_z(cutoff), "cutoff_ist": cutoff_ist(cutoff), "as_of_utc": iso_z(as_of)}}


def sentiment(session: Session, as_of: datetime | None = None, save: bool = True, inputs: dict | None = None):
    as_of = as_of or utcnow()
    version, params = params_svc.get_active(session)
    inputs = inputs or live_inputs(session, as_of)
    return run_system1(session, inputs["infos"], inputs["items"], params["system1"], version, as_of, save=save)


def predict(session: Session, as_of: datetime | None = None) -> dict:
    as_of = as_of or utcnow()
    version, params = params_svc.get_active(session)
    inputs = live_inputs(session, as_of)
    s1_out, scores, flows = sentiment(session, as_of, inputs=inputs)
    infos = inputs["infos"]
    symbols = {c.ticker: c.yahoo_symbol for c in active_companies(session, inputs["target"])}
    # Closes strictly before the target session only - a price from the predicted session (or later)
    # can never enter the features.
    closes = {c.ticker: load_closes(session, symbols[c.ticker], end=inputs["target"] - timedelta(days=1))
              for c in infos}
    idx = load_closes(session, index_symbol(), end=inputs["target"] - timedelta(days=1))
    cal = get_calendar()
    base = inputs["target"]   # same session System 1 scored as "today"
    # The base price must be the close of the session right before `base`; a stale close is MISSING.
    rows = build_predictions(scores, infos, closes, idx if not idx.empty else None, base, params["system2"],
                             expected_previous_date=cal.previous_session(base), flows=flows)
    attach_news_inputs(rows, inputs["items"], flows, as_of)
    quality = {**inputs["summary"], "flags": dict(Counter(r.data_quality_flag for r in rows if r.status == "OK"))}
    run, objs = persist_predictions(session, rows, as_of, version, s1_out["run_id"], p2=params["system2"],
                                    run_meta={"prediction_cutoff": inputs["cutoff"], "target_session": base,
                                              "universe_version": inputs["universe_version"],
                                              "sentiment_engine": inputs["engine"],
                                              "availability_policy": inputs["policy"], "data_quality": quality})
    return {"run_id": run.id, "model_version": run.model_version, "target_session": base.isoformat(),
            "prediction_cutoff": cutoff_ist(inputs["cutoff"]), "universe_version": inputs["universe_version"],
            "data_quality": quality, "sentiment": s1_out, "predictions": [prediction_to_dict(p) for p in objs],
            "disclaimer": "These are model estimates, not guaranteed outcomes. Research output only."}


def attach_news_inputs(rows, items, flows, cutoff_or_as_of: datetime) -> None:
    """Snapshot of every news item behind each prediction (new-news window + raw-score evidence),
    with its availability relative to the cut-off, and a per-prediction data-quality flag:
      VERIFIED     all new company/macro news was held before the cut-off
      UNCERTAIN    some of it relies on provider timestamps only (research replays)
      NO_NEW_NEWS  no new company or macro news since the previous cut-off"""
    by_id: dict[int, list] = {}
    for it in items:
        by_id.setdefault(it.news_id, []).append(it)

    def snap(nid, entity, role):
        cands = by_id.get(nid, [])
        it = next((x for x in cands if x.entity == entity), cands[0] if cands else None)
        if it is None:
            return {"news_id": nid, "entity": entity, "role": role, "eligibility": "not_loaded"}
        return {"news_id": nid, "entity": it.entity, "role": role, "title": it.title, "source": it.source,
                "provider_published_at": iso_z(it.provider_published_at), "available_at": iso_z(it.published_at),
                "fetched_at": iso_z(it.fetched_at), "timestamp_precision": it.timestamp_precision,
                "timezone_status": it.timezone_status, "eligibility": it.eligibility(cutoff_or_as_of),
                "event_type": it.event_type, "price_report": it.price_report, "sentiment": it.sentiment,
                "engine": it.engine}

    for r in rows:
        if r.status != "OK":
            continue
        fc, fm = flows.get(r.ticker), flows.get("MACRO")
        new = [snap(n, r.ticker, "new_company_news") for n in (fc.news_ids if fc else [])] + \
              [snap(n, "MACRO", "new_macro_news") for n in (fm.news_ids if fm else [])]
        new_ids = {x["news_id"] for x in new}
        evidence = [snap(n, r.ticker, "raw_score_evidence") for n in r.evidence_news_ids if n not in new_ids]
        r.news_inputs = new + evidence
        r.data_quality_flag = "NO_NEW_NEWS" if not new else \
            ("VERIFIED" if all(x["eligibility"] == "verified" for x in new) else "UNCERTAIN")


def daily_run(session: Session, validate: bool = False, now: datetime | None = None) -> dict:
    """Production workflow, meant to be scheduled before 09:15 IST on trading days:

      collect -> validate timestamps (future stamps rejected, precision + availability classified at
      storage) -> de-duplicate (exact URL + near-duplicate titles) -> score (System 1) -> refresh prices
      -> predict (System 2) [-> LLM review, stored separately].

    The predicted session is the first NSE session whose 09:15 IST open is still ahead of `now`
    (weekends and exchange holidays skipped). Run after the open, it targets the NEXT session -
    a session that has already opened is never predicted.
    """
    now = now or utcnow()
    cal = get_calendar()
    target = prediction_session(now, cal)
    steps: dict = {"started_at": now.isoformat() + "Z", "target_session": target.isoformat(),
                   "today_ist": ist_date(now).isoformat(),
                   "today_is_trading_day": cal.is_trading_day(ist_date(now)),
                   "market_open_passed_today": cal.is_trading_day(ist_date(now)) and target != ist_date(now)}
    steps["collect"] = collect_and_analyze(session)
    steps["timestamp_audit"] = availability.audit(session)
    steps["market_data"] = {k: v.get("status") for k, v in refresh_market_data(session).items()}
    session.commit()
    # Predict with the information time AFTER collection (still before the cut-off), so news fetched in
    # this run is usable; if collection ran past the open, the target moves to the next session.
    as_of = utcnow()
    if prediction_session(as_of, cal) != target:
        steps["warning"] = (f"collection finished after the {target} cut-off; predicting "
                            f"{prediction_session(as_of, cal)} instead")
    out = predict(session, as_of)
    steps["prediction_as_of"] = as_of.isoformat() + "Z"
    steps["prediction_run"] = out["run_id"]
    steps["predictions"] = len(out["predictions"])
    steps["missing"] = sum(p["status"] != "OK" for p in out["predictions"])
    if validate:
        steps["llm_validation"] = validate_safely(session, out["run_id"])
    return steps


def validate_safely(session: Session, run_id: str, provider: str | None = None,
                    tickers: list[str] | None = None) -> dict | str:
    """LLM review AFTER the prediction is committed: whatever happens during validation (provider not
    configured, API error, timeout, invalid JSON, unexpected exception), the stored numeric prediction
    is never lost or modified."""
    from backend.services.llm import get_llm, provider_status, redact
    from backend.system2_prediction.validator import validate_run
    session.commit()                                   # the prediction is durable before any LLM call
    llm = get_llm(provider)
    if llm is None:
        return "LLM validation not performed: LLM_PROVIDER=none"
    try:
        res = validate_run(session, run_id, llm, tickers)
        session.commit()
        return res
    except Exception as exc:                           # pragma: no cover - validate_run already guards
        session.rollback()
        return {"error": redact(f"LLM validation failed: {type(exc).__name__}: {exc}"), **provider_status()}
