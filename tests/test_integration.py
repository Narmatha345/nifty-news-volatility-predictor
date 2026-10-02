"""End-to-end: News -> System 1 -> System 2 -> LLM validation -> System 3 (DB-backed, offline)."""
import random
from datetime import date, datetime, time

import pandas as pd
import pytest

from backend.data.market_data import MarketDataProvider, refresh
from backend.database.models import BacktestRun, LLMValidation, ParameterSet, Prediction
from backend.services import orchestrator, parameters as params_svc
from backend.services.llm import LLMProvider
from backend.services.timeutil import market_open_utc
from backend.services.universe import active_companies, index_symbol
from backend.system1_news.collector import collect
from backend.system2_prediction.validator import validate_run
from backend.system3_backtesting import runner
from tests.conftest import FakeProvider, raw, synthetic_market


class FakeMarket(MarketDataProvider):
    name = "fake-market"

    def __init__(self, closes: dict[str, pd.Series]):
        self.closes = closes

    def fetch_daily(self, symbol, start, end):
        s = self.closes[symbol]
        s = s[(s.index >= start) & (s.index <= end)]
        return pd.DataFrame({"open": s, "high": s, "low": s, "close": s, "volume": 1e6})


class FakeLLM(LLMProvider):
    name, model = "fake-llm", "fake-1"

    def __init__(self):
        self.payloads = []

    def complete_json(self, system, user, schema):
        self.payloads.append(user)
        return {"validation_status": "weakly_supported", "reason": "fake review", "key_supporting_factors": [],
                "key_contradicting_factors": ["test"], "data_quality_issues": [], "cutoff_check": "passed",
                "missing_information": []}, "{}"


HEADLINES = {1: ["{n} profit surges, beats estimates", "{n} wins record order", "{n} upgraded by brokerage"],
             -1: ["{n} shares plunge after weak guidance", "{n} faces probe, penalty", "{n} downgraded on concerns"]}


@pytest.fixture
def world(session):
    """Synthetic world: 300 sessions of prices; news before the open moves the stock that day."""
    comps = active_companies(session, date(2024, 1, 1))
    rng = random.Random(11)
    from tests.conftest import weekdays
    days = weekdays(date(2024, 1, 1), 300)
    effects, articles = {}, []
    for d in days[40:]:
        if rng.random() < 0.4:
            c = rng.choice(comps)
            sign = rng.choice([1, -1])
            title = rng.choice(HEADLINES[sign]).format(n=c.name)
            articles.append(raw(title, datetime.combine(d, time(2, 7)), source="Reuters"))
            effects[(c.yahoo_symbol, d)] = 0.025 * sign
    articles.append(raw("RBI announces rate cut as inflation eases", datetime.combine(days[200], time(1, 13))))
    symbols = [c.yahoo_symbol for c in comps]
    closes, idx, days = synthetic_market(symbols, n=300, news_effect=effects)
    closes[index_symbol()] = idx
    return {"days": days, "articles": articles, "closes": closes, "effects": effects}


def test_full_pipeline(session, world):
    days = world["days"]
    provider = FakeProvider(world["articles"])
    stats = collect(session, providers=[provider], queries=["q1", "q2"])   # 2nd query returns exact duplicates
    assert stats.new == len(world["articles"]) and stats.exact_duplicates == len(world["articles"])
    assert orchestrator.analyze_news(session) == len(world["articles"])
    res = refresh(session, list(world["closes"]), days[0], days[-1], provider=FakeMarket(world["closes"]))
    assert all(v["status"] == "ok" for v in res.values())

    # --- System 1 + 2 live-style run at a historical as_of
    as_of = market_open_utc(days[250])
    out = orchestrator.predict(session, as_of)
    s1 = out["sentiment"]
    assert s1["horizons"][0]["horizon_type"] == "today"
    assert {"macro_score", "nifty_score", "nifty_company_weighted_score"} <= set(s1["horizons"][0])
    preds = out["predictions"]
    assert preds and all(p["status"] == "OK" for p in preds if p["horizon_type"] == "today")
    p = preds[0]
    assert p["estimated_price"] == pytest.approx(p["previous_close"] * (1 + p["predicted_movement_percent"] / 100),
                                                 abs=0.01)

    # --- LLM validation is stored separately and never alters predictions
    before = {x.id: x.predicted_movement for x in session.query(Prediction).filter_by(run_id=out["run_id"])}
    llm = FakeLLM()
    results = validate_run(session, out["run_id"], llm)
    assert set(results) == {c.ticker for c in active_companies(session, days[250])}   # universe of that date
    after = {x.id: x.predicted_movement for x in session.query(Prediction).filter_by(run_id=out["run_id"])}
    assert before == after
    assert session.query(LLMValidation).count() == len(before)
    assert "news_evidence" in llm.payloads[0]

    # --- System 3 backtest + tuning
    active_before = params_svc.get_active(session)[0]
    run = runner.create_run(session, days[100], days[220], horizon_types=("today", "month_end"), tune_params=True,
                            n_trials=8)
    run = runner.execute(session, run.id)
    assert run.status == "completed", run.error
    rep = run.report_json
    assert rep["total_predictions"] > 0 and rep["metrics"]["n"] == rep["total_predictions"]
    assert "## Company-wise results" in run.report_md
    # On days with same-day company news the synthetic effect must be picked up. (Stale news carries
    # no signal in this synthetic world, so it is excluded from this check.)
    sym = {c.ticker: c.yahoo_symbol for c in active_companies(session, days[250])}
    same_day = [x for x in session.query(Prediction).filter(Prediction.horizon_type == "today")
                if x.actual is not None and (sym[x.ticker], x.target_date) in world["effects"]]
    assert len(same_day) > 20
    hits = sum((x.predicted_movement > 0) == (world["effects"][(sym[x.ticker], x.target_date)] > 0)
               for x in same_day)
    assert hits / len(same_day) > 0.75
    # tuned parameters are only candidates
    assert params_svc.get_active(session)[0] == active_before
    statuses = {p.version: p.status for p in session.query(ParameterSet)}
    assert statuses[active_before] == "active"
    assert all(s in ("candidate", "active") for s in statuses.values())
    # backtest predictions + actuals are stored and traceable
    stored = session.query(Prediction).filter(Prediction.run_id.like("p-%")).all()
    assert any(x.actual is not None and x.evidence_news_ids for x in stored)


def test_backtest_without_market_data_fails_clearly(session):
    run = runner.create_run(session, date(2024, 1, 1), date(2024, 3, 1))
    run = runner.execute(session, run.id)
    assert run.status == "failed" and "market data" in run.error
    assert session.get(BacktestRun, run.id).status == "failed"
