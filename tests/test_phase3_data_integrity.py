"""Phase 3 regression tests: timestamp quality, cut-off eligibility, verified-only live inputs,
reproducible prediction snapshots, after-close outcomes and the date-aware universe.
All data here is SYNTHETIC test data."""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import select

from backend.data.market_data import store_bars
from backend.database.models import ActualResult, LLMValidation, NewsArticle, Prediction, PredictionRun
from backend.services import orchestrator
from backend.services import universe as uni
from backend.services.llm import LLMProvider, UnavailableLLM
from backend.services.calendar import get_calendar
from backend.services.timeutil import market_close_utc, market_open_utc, parse_timestamp_info
from backend.system1_news.aggregation import AnalysisItem
from backend.system1_news.availability import eligibility
from backend.system1_news.collector import CollectStats, store_articles
from backend.system2_prediction.predictor import PredictionRow
from backend.system2_prediction.validator import validate_run
from backend.system3_backtesting import runner
from backend.system3_backtesting.backtester import BacktestData, run_backtest
from tests.conftest import COMPANIES, raw, synthetic_market, weekdays

CUTOFF = datetime(2026, 9, 29, 3, 45)          # 09:15 IST on Tue 29 Sep 2026


# ---------------------------------------------------------------------------- cut-off eligibility
def test_exact_timestamp_before_cutoff_is_usable_after_is_not():
    before = CUTOFF - timedelta(minutes=30, seconds=7)
    assert eligibility(before, "exact", "explicit", before + timedelta(minutes=2), CUTOFF) == "verified"
    assert eligibility(before, "exact", "explicit", CUTOFF + timedelta(days=3), CUTOFF) == "timestamp_before"
    at = CUTOFF
    assert eligibility(at + timedelta(seconds=1), "exact", "explicit", CUTOFF + timedelta(days=3), CUTOFF) \
        == "post_cutoff"
    # hour precision counts from the END of the stated hour
    assert eligibility(datetime(2026, 9, 29, 3, 0), "hour", "explicit", None, CUTOFF) == "post_cutoff"
    assert eligibility(datetime(2026, 9, 29, 2, 0), "hour", "explicit", None, CUTOFF) == "timestamp_before"


def test_date_only_and_unknown_timezone_are_never_verified_by_timestamp():
    stamp = datetime(2026, 9, 28)                                     # date only
    assert eligibility(stamp, "date_only", "explicit", CUTOFF + timedelta(days=1), CUTOFF) == "uncertain"
    assert eligibility(datetime(2026, 9, 29), "date_only", "explicit", None, CUTOFF) == "post_cutoff"
    precise_but_no_tz = datetime(2026, 9, 28, 20, 13, 5)
    assert eligibility(precise_but_no_tz, "exact", "unknown", CUTOFF + timedelta(days=1), CUTOFF) == "uncertain"
    # ... but anything this system demonstrably held before the cut-off IS verified
    assert eligibility(stamp, "date_only", "unknown", CUTOFF - timedelta(hours=1), CUTOFF) == "verified"


@pytest.mark.parametrize("value,tz,utc,precision,label,status", [
    ("2026-10-01T08:07:00+05:30", None, datetime(2026, 10, 1, 2, 37), "minute", "+05:30", "explicit"),
    ("2026-10-01T08:00:00+05:30", None, datetime(2026, 10, 1, 2, 30), "hour", "+05:30", "explicit"),
    ("2026-10-01T02:30:15Z", None, datetime(2026, 10, 1, 2, 30, 15), "exact", "UTC", "explicit"),
    ("Thu, 01 Oct 2026 02:30:15 GMT", None, datetime(2026, 10, 1, 2, 30, 15), "exact", "GMT", "explicit"),
    ("2026-10-01T02:30:15", None, datetime(2026, 10, 1, 2, 30, 15), "exact", None, "unknown"),
    ("2026-10-01T02:30:15", "UTC", datetime(2026, 10, 1, 2, 30, 15), "exact", "UTC", "provider_documented"),
    ("2026-10-01", None, datetime(2026, 10, 1), "date_only", None, "unknown"),
    ("garbage", None, None, "unknown", None, "unknown"),
])
def test_timezone_normalisation_and_precision(value, tz, utc, precision, label, status):
    info = parse_timestamp_info(value, tz)
    assert (info.utc, info.precision, info.timezone, info.tz_status, info.raw) == (utc, precision, label, status, value)


def test_future_and_missing_timestamps_are_rejected_and_counted(session):
    now_ref = datetime(2026, 9, 29, 2, 0)
    stats = CollectStats()
    arts = [raw("HDFC Bank wins order", now_ref - timedelta(hours=1)),
            raw("From the future", datetime(2099, 1, 1)),
            dataclass_replace(raw("No timestamp", now_ref), published_at=None)]
    store_articles(session, arts, stats)
    assert stats.new == 1 and stats.rejected_future == 1 and stats.rejected_missing_timestamp == 1
    stored = session.scalars(select(NewsArticle)).all()
    assert len(stored) == 1 and stored[0].timezone_status == "explicit"
    assert stored[0].published_at_ist == stored[0].published_at + timedelta(hours=5, minutes=30)


def dataclass_replace(obj, **kw):
    import dataclasses
    return dataclasses.replace(obj, **kw)


def test_duplicates_are_counted_and_linked(session):
    stats = CollectStats()
    t = datetime(2026, 9, 28, 10, 0, 5)
    store_articles(session, [raw("Infosys wins big deal from US bank", t, url="https://a.example/1"),
                             raw("Infosys wins big deal from US bank", t, url="https://a.example/1"),
                             raw("Infosys wins big deal from a US bank", t + timedelta(minutes=9),
                                 url="https://b.example/2")], stats)
    assert stats.new == 2 and stats.exact_duplicates == 1 and stats.near_duplicates == 1
    rows = session.scalars(select(NewsArticle).order_by(NewsArticle.id)).all()
    assert rows[1].duplicate_of == rows[0].id


# ------------------------------------------------------------------------------- live pipeline
def _seed_prices(session, last_day: date = date(2026, 10, 9)):
    symbols = [c.yahoo_symbol for c in uni.companies_ever_confirmed(session)] + [uni.index_symbol()]
    days = [d for d in weekdays(date(2025, 6, 2), 400) if d <= last_day]
    closes, idx, days = synthetic_market(symbols[:-1], start=days[0], n=len(days))
    closes[symbols[-1]] = idx
    cal = get_calendar()                     # real NSE sessions: no prices on weekends / exchange holidays
    for sym, s in closes.items():
        s = s[[cal.is_trading_day(d) for d in s.index]]
        df = pd.DataFrame({"open": s, "high": s, "low": s, "close": s, "volume": 1e6})
        store_bars(session, sym, df, "synthetic")
    return closes


def _article(session, title, published, fetched, precision="exact", tz="explicit"):
    a = NewsArticle(source="Reuters", provider="fake", title=title, url=f"https://x.example/{hash(title)}",
                    url_hash=str(abs(hash(title))), title_fingerprint=title.lower(), published_at=published,
                    fetched_at=fetched, timestamp_precision=precision, timezone_status=tz)
    session.add(a)
    session.flush()
    return a


@pytest.fixture
def live_world(session):
    _seed_prices(session)
    verified = _article(session, "HDFC Bank wins large contract", datetime(2026, 9, 28, 10, 0, 5),
                        datetime(2026, 9, 28, 10, 5, 0))
    late = _article(session, "HDFC Bank faces regulatory action", datetime(2026, 9, 28, 11, 0, 7),
                    datetime(2026, 10, 1, 9, 0, 0))                      # stamp before cut-off, fetched after
    dated = _article(session, "ICICI Bank shares in focus", datetime(2026, 9, 28, 7, 0), datetime(2026, 10, 1),
                     precision="date_only")
    orchestrator.analyze_news(session)
    return {"verified": verified.id, "late": late.id, "dated": dated.id}


def test_live_prediction_uses_only_verified_news_and_records_the_cutoff(session, live_world):
    as_of = datetime(2026, 9, 29, 2, 0)                                  # 07:30 IST
    out = orchestrator.predict(session, as_of)
    run = session.get(PredictionRun, out["run_id"])
    assert run.prediction_cutoff == market_open_utc(date(2026, 9, 29)) == CUTOFF
    assert out["prediction_cutoff"] == "2026-09-29 09:15:00+05:30"
    assert run.target_session == date(2026, 9, 29) and run.availability_policy == "verified"
    assert run.universe_version == "legacy-approx-2025-06-30" and run.sentiment_engine.startswith("lexicon")
    preds = session.scalars(select(Prediction).where(Prediction.run_id == run.id)).all()
    used = {x["news_id"] for p in preds for x in (p.news_inputs or [])}
    assert live_world["verified"] in used
    assert live_world["late"] not in used and live_world["dated"] not in used     # never as verified input
    hdfc = next(p for p in preds if p.ticker == "HDFCBANK" and p.horizon_type == "today")
    assert hdfc.data_quality_flag == "VERIFIED"
    assert all(x["eligibility"] == "verified" for x in hdfc.news_inputs)
    other = next(p for p in preds if p.ticker == "TCS" and p.horizon_type == "today")
    assert other.data_quality_flag == "NO_NEW_NEWS"
    assert run.data_quality["excluded_items"] >= 2


def test_prediction_is_reproducible_and_ignores_future_prices_and_news(session, live_world):
    as_of = datetime(2026, 9, 29, 2, 0)
    a = orchestrator.predict(session, as_of)
    # later events: a close on/after the target session and news fetched after as_of
    for sym in ("HDFCBANK.NS", "ICICIBANK.NS"):
        s = pd.Series([1e5, 1e5], index=[date(2026, 9, 29), date(2026, 9, 30)])
        store_bars(session, sym, pd.DataFrame({"open": s, "high": s, "low": s, "close": s, "volume": 1.0}), "x")
    _article(session, "HDFC Bank crashes on fraud probe", datetime(2026, 9, 28, 22, 0, 1), datetime(2026, 9, 29, 2, 30))
    orchestrator.analyze_news(session)
    b = orchestrator.predict(session, as_of)
    key = lambda o: {(p["ticker"], p["horizon_label"]): (p["predicted_movement_percent"], p["previous_close"])
                     for p in o["predictions"]}
    assert key(a) == key(b)
    assert {p["previous_close_date"] for p in a["predictions"] if p["status"] == "OK"} == {"2026-09-28"}


def test_holiday_and_next_valid_session(session):
    _seed_prices(session)
    # 2026-10-02 (Gandhi Jayanti) is an NSE holiday: an evening run on 1 Oct targets Mon 5 Oct
    out = orchestrator.predict(session, datetime(2026, 10, 1, 14, 0))
    assert out["target_session"] == "2026-10-05"
    run = session.get(PredictionRun, out["run_id"])
    assert run.prediction_cutoff == market_open_utc(date(2026, 10, 5))
    today = [p for p in out["predictions"] if p["horizon_type"] == "today"]
    assert today and all(p["status"] == "OK" for p in today)
    assert all(p["previous_close_date"] == "2026-10-01" for p in today if p["status"] == "OK")


def test_after_close_outcomes_never_score_a_holiday_or_unclosed_session(session):
    closes = _seed_prices(session)
    orchestrator.predict(session, datetime(2026, 10, 1, 14, 0))           # targets Mon 5 Oct
    early = runner.score_live_predictions(session, now=market_close_utc(date(2026, 10, 5)) - timedelta(minutes=5))
    assert early["scored"] == 0 and early["not_matured"] > 0
    res = runner.score_live_predictions(session, now=market_close_utc(date(2026, 10, 5)) + timedelta(hours=1))
    assert res["scored"] > 0
    ar = session.scalars(select(ActualResult)).first()
    p = session.get(Prediction, ar.prediction_id)
    assert ar.actual_close_date == date(2026, 10, 5) and ar.previous_close_date == date(2026, 10, 1)
    assert ar.abs_error == pytest.approx(abs(p.predicted_movement - ar.actual_movement), abs=1e-3)
    assert ar.squared_error == pytest.approx(ar.abs_error ** 2, rel=1e-3)
    assert ar.actual_direction in ("UP", "DOWN", "NEUTRAL")


def test_missing_target_price_is_reported_not_scored(session):
    _seed_prices(session)
    orchestrator.predict(session, datetime(2026, 10, 1, 14, 0))
    from backend.database.models import MarketData
    for bar in session.scalars(select(MarketData).where(MarketData.date == date(2026, 10, 5))):
        session.delete(bar)                                               # data gap on the target session
    session.flush()
    res = runner.score_live_predictions(session, now=datetime(2026, 10, 9, 12, 0))
    assert res["scored"] == 0 and len(res["missing_target_price"]) > 0
    assert session.scalars(select(ActualResult)).first() is None


# ------------------------------------------------------------------------------------- LLM review
class OverreachingLLM(LLMProvider):
    name, model = "fake-llm", "fake-1"

    def complete_json(self, system, user, schema):
        return {"validation_status": "INCONSISTENT", "confidence": 0.9, "inconsistencies": ["x"],
                "missing_information": [], "reasoning": "r", "predicted_movement_percent": 99.0}, "{}"


def test_llm_cannot_overwrite_prediction_and_unavailable_llm_is_labelled(session, live_world):
    out = orchestrator.predict(session, datetime(2026, 9, 29, 2, 0))
    before = {p.id: (p.predicted_movement, p.predicted_direction)
              for p in session.scalars(select(Prediction).where(Prediction.run_id == out["run_id"]))}
    res = validate_run(session, out["run_id"], OverreachingLLM(), tickers=["HDFCBANK"])
    after = {p.id: (p.predicted_movement, p.predicted_direction)
             for p in session.scalars(select(Prediction).where(Prediction.run_id == out["run_id"]))}
    assert before == after
    # a response carrying a replacement number fails schema validation and is NOT stored as a review
    assert res["HDFCBANK"]["validation_status"] == "validation_error"
    assert res["HDFCBANK"]["validation_performed"] is False
    res2 = validate_run(session, out["run_id"], UnavailableLLM("anthropic", "claude-opus-5-5", "no API key"),
                        tickers=["TCS"])
    assert res2["TCS"]["validation_status"] == "not_performed"
    assert res2["TCS"]["validation_performed"] is False and "no API key" in res2["TCS"]["reason"]
    v = session.scalars(select(LLMValidation).where(LLMValidation.provider == "anthropic")).first()
    assert v.validation_status == "not_performed" and v.request_status == "not_configured"


# ------------------------------------------------------------------------------------ uncertainty
def test_uncertain_inputs_are_labelled():
    cutoff = CUTOFF
    it_v = AnalysisItem(1, 1, "HDFCBANK", False, "other", 1.0, 40, 1.0, 1.0, 5, cutoff - timedelta(hours=3),
                        fetched_at=cutoff - timedelta(hours=2), provider_published_at=cutoff - timedelta(hours=3))
    it_u = AnalysisItem(2, 2, "HDFCBANK", False, "other", 1.0, 40, 1.0, 1.0, 5, cutoff - timedelta(hours=2),
                        fetched_at=cutoff + timedelta(days=5), provider_published_at=datetime(2026, 9, 28),
                        timestamp_precision="date_only")

    class F:
        def __init__(self, ids):
            self.news_ids = ids

    def row():
        return PredictionRow("HDFCBANK", "HDFC Bank", "today", "Today", date(2026, 9, 29), 1, "OK")
    r1, r2, r3 = row(), row(), row()
    orchestrator.attach_news_inputs([r1], [it_v, it_u], {"HDFCBANK": F([1]), "MACRO": F([])}, cutoff)
    orchestrator.attach_news_inputs([r2], [it_v, it_u], {"HDFCBANK": F([1, 2]), "MACRO": F([])}, cutoff)
    orchestrator.attach_news_inputs([r3], [], {"HDFCBANK": F([]), "MACRO": F([])}, cutoff)
    assert (r1.data_quality_flag, r2.data_quality_flag, r3.data_quality_flag) == ("VERIFIED", "UNCERTAIN", "NO_NEW_NEWS")
    assert {x["eligibility"] for x in r2.news_inputs} == {"verified", "uncertain"}


# ---------------------------------------------------------------------------------------- universe
def test_universe_is_date_aware_and_unconfirmed_versions_are_not_used(session):
    # seeded config: legacy version until 2026-09-29, NSE factsheet version (confirmed) from 2026-09-30
    past = uni.universe_on(session, date(2026, 6, 1))
    assert past.version == "legacy-approx-2025-06-30" and past.weights["HDFCBANK"] == 13.0   # no retro weights
    assert {"ITC", "TCS"} <= set(past.weights) and len(past.members) == 10
    now = uni.universe_on(session, date(2026, 10, 1))
    assert now.version == "nse-factsheet-2026-09-30" and now.weights["HDFCBANK"] == 10.38
    assert {"KOTAKBANK", "M&M"} <= set(now.weights) and not {"ITC", "TCS"} & set(now.weights)
    periods = uni.universe_periods(session, date(2026, 9, 1), date(2026, 10, 15))
    assert [(str(a), str(b), s.version) for a, b, s in periods] == [
        ("2026-09-01", "2026-09-29", "legacy-approx-2025-06-30"),
        ("2026-09-30", "2026-10-15", "nse-factsheet-2026-09-30")]
    # an UNCONFIRMED later version is reported but never used until confirmed
    from backend.database.models import UniverseMember
    from backend.services.timeutil import utcnow
    for t, w in (("HDFCBANK", 11.0), ("ITC", 5.0)):
        session.add(UniverseMember(version="test-pending-2027", ticker=t, company=t, weight=w,
                                   effective_from=date(2027, 1, 1), source="test", status="requires_confirmation",
                                   created_at=utcnow()))
    session.flush()
    snap = uni.universe_on(session, date(2027, 2, 1))
    assert snap.version == "nse-factsheet-2026-09-30" and snap.pending == ["test-pending-2027"]
    pend = uni.universe_report(session, date(2027, 2, 1))["requires_confirmation"][0]
    assert pend["would_add"] == ["ITC"] and "KOTAKBANK" in pend["would_remove"]
    uni.confirm_version(session, "test-pending-2027")
    assert uni.universe_on(session, date(2027, 2, 1)).version == "test-pending-2027"
    assert uni.universe_on(session, date(2026, 10, 1)).version == "nse-factsheet-2026-09-30"
    # versions are immutable: re-seeding never rewrites stored members
    uni.seed_versions(session)
    assert uni.universe_on(session, date(2026, 6, 1)).weights["HDFCBANK"] == 13.0
    # news is analysed against every confirmed member, past and present
    assert {c.ticker for c in uni.companies_ever_confirmed(session)} >= {"ITC", "TCS", "KOTAKBANK", "M&M"}


def test_backtest_uses_the_universe_effective_on_each_date(params):
    from backend.system1_news.pipeline import CompanyInfo
    tickers = [c.ticker for c in COMPANIES]
    closes, idx, days = synthetic_market(tickers)
    early = [CompanyInfo(c.ticker, c.name, c.sector, 1.0) for c in COMPANIES[:2]]
    late = [CompanyInfo(c.ticker, c.name, c.sector, 1.0) for c in COMPANIES]
    split = days[200]
    data = BacktestData([], late, closes, idx,
                        universe=[(days[0], split - timedelta(days=1), "v1", early), (split, days[-1], "v2", late)])
    recs = run_backtest(data, [days[190], days[210]], params, ("today",))
    v1 = {r.ticker for r in recs if r.as_of_date == days[190].isoformat()}
    v2 = {r.ticker for r in recs if r.as_of_date == days[210].isoformat()}
    assert v1 == set(tickers[:2]) and v2 == set(tickers)
    assert {r.universe_version for r in recs if r.as_of_date == days[190].isoformat()} == {"v1"}
    assert all(r.data_quality_flag == "NO_NEW_NEWS" for r in recs)


def test_daily_run_uses_news_collected_in_the_same_run(session, monkeypatch):
    """Regression: `daily` once passed its START time as the information time, so articles fetched
    by its own collection step were excluded."""
    _seed_prices(session)
    seen = {}
    clock = {"t": datetime(2026, 9, 29, 1, 0)}            # 06:30 IST, before the 09:15 open

    def utcnow():
        clock["t"] += timedelta(seconds=30)               # time passes while the run works
        return clock["t"]

    monkeypatch.setattr(orchestrator, "utcnow", utcnow)

    def fake_collect(s, start=None, end=None):
        a = _article(s, "Axis Bank wins large contract", utcnow() - timedelta(minutes=10, seconds=3), utcnow())
        orchestrator.analyze_news(s)
        seen["id"] = a.id
        return {"new": 1}

    monkeypatch.setattr(orchestrator, "collect_and_analyze", fake_collect)
    monkeypatch.setattr(orchestrator, "refresh_market_data", lambda s: {})
    steps = orchestrator.daily_run(session)
    preds = session.scalars(select(Prediction).where(Prediction.run_id == steps["prediction_run"])).all()
    used = {x["news_id"] for p in preds for x in (p.news_inputs or [])}
    assert seen["id"] in used
    run = session.get(PredictionRun, steps["prediction_run"])
    assert run.as_of >= session.get(NewsArticle, seen["id"]).fetched_at
    assert run.prediction_cutoff == market_open_utc(run.target_session) and run.as_of < run.prediction_cutoff
