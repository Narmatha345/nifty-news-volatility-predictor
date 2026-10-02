from datetime import date, datetime, timedelta

import pytest

from backend.services.calendar import calendar_from_sessions, get_calendar, prediction_session
from backend.services.horizons import generate_horizons, quarter_label
from backend.system1_news.aggregation import aggregate, nifty_scores
from tests.conftest import item, weekdays

AS_OF = datetime(2026, 10, 1, 3, 45)


def test_weighted_score_not_simple_average(params):
    p1 = params["system1"]
    items = [item(1, "HDFCBANK", 80, AS_OF - timedelta(hours=2), relevance=1.0, reliability=0.95),
             item(2, "HDFCBANK", -30, AS_OF - timedelta(hours=2), relevance=0.3, reliability=0.5)]
    s = aggregate("HDFCBANK", items, AS_OF, AS_OF, p1)
    assert s.raw_mean > (80 - 30) / 2          # high-relevance, reliable source dominates
    assert s.news_count == 2
    assert abs(s.score) < abs(s.raw_mean)       # evidence shrinkage


def test_shrinkage_pulls_thin_evidence_to_zero(params):
    p1 = params["system1"]
    weak = aggregate("X", [item(1, "X", 90, AS_OF, relevance=0.3, strength=0.3, reliability=0.5)], AS_OF, AS_OF, p1)
    strong = aggregate("X", [item(i, "X", 90, AS_OF) for i in range(1, 6)], AS_OF, AS_OF, p1)
    assert 0 < weak.score < strong.score < 90


def test_duplicates_count_once(params):
    p1 = params["system1"]
    one = aggregate("X", [item(1, "X", 60, AS_OF)], AS_OF, AS_OF, p1)
    dups = [item(i, "X", 60, AS_OF, cluster_id=1, source=f"S{i}") for i in range(1, 6)]
    clustered = aggregate("X", dups, AS_OF, AS_OF, p1)
    assert clustered.news_count == 1 and clustered.article_count == 5
    independent = aggregate("X", [item(i, "X", 60, AS_OF) for i in range(1, 6)], AS_OF, AS_OF, p1)
    assert one.evidence_weight < clustered.evidence_weight < independent.evidence_weight


def test_recency_decay_and_horizon_projection(params):
    p1 = params["system1"]
    items = [item(1, "X", 50, AS_OF - timedelta(days=1), half_life=10),
             item(2, "X", -50, AS_OF - timedelta(days=30), half_life=10)]
    s = aggregate("X", items, AS_OF, AS_OF, p1)
    assert s.raw_mean > 0
    far = aggregate("X", items, AS_OF, AS_OF + timedelta(days=90), p1)
    assert far.evidence_weight < s.evidence_weight   # news fades for distant targets


def test_no_lookahead_in_aggregation(params):
    p1 = params["system1"]
    past = item(1, "X", 40, AS_OF - timedelta(hours=1))
    future = item(2, "X", -100, AS_OF + timedelta(minutes=1))
    assert aggregate("X", [past, future], AS_OF, AS_OF, p1).score == aggregate("X", [past], AS_OF, AS_OF, p1).score


def test_min_relevance_filter(params):
    p1 = {**params["system1"], "min_relevance": 0.5}
    assert aggregate("X", [item(1, "X", 90, AS_OF, relevance=0.4)], AS_OF, AS_OF, p1).news_count == 0


def test_nifty_scores_use_weights_and_stay_separate(params):
    p1 = params["system1"]
    a = aggregate("A", [item(1, "A", 80, AS_OF)], AS_OF, AS_OF, p1)
    b = aggregate("B", [item(2, "B", -80, AS_OF)], AS_OF, AS_OF, p1)
    macro = aggregate("MACRO", [item(3, "MACRO", 20, AS_OF)], AS_OF, AS_OF, p1)
    n = nifty_scores({"A": a, "B": b}, {"A": 3.0, "B": 1.0}, macro, p1)
    assert n["company_weighted"] == pytest.approx((3 * a.score + b.score) / 4, abs=0.01)
    assert n["macro"] == macro.score
    mw = p1["macro_weight_in_nifty"]
    assert n["overall"] == pytest.approx((1 - mw) * n["company_weighted"] + mw * macro.score, abs=0.01)


def test_horizons_for_2026_10_01():
    cal = get_calendar()
    hs = generate_horizons(date(2026, 10, 1), cal, "indian_fy")
    by = {}
    for h in hs:
        by.setdefault(h.horizon_type, []).append(h)
    assert by["today"][0].target_date == date(2026, 10, 1)
    assert by["today"][0].trading_days_ahead == 1
    months = [h.target_date for h in by["month_end"]]
    assert len(months) == 3
    assert months[0] == date(2026, 10, 30)          # Oct 31 2026 is a Saturday
    assert [m.month for m in months] == [10, 11, 12]
    quarters = by["quarter_end"]
    assert len(quarters) == 6
    assert [q.calendar_date for q in quarters][:2] == [date(2026, 12, 31), date(2027, 3, 31)]
    assert quarters[0].label == "Q3 FY27 end" and quarters[1].label == "Q4 FY27 end"
    assert all(h.trading_days_ahead >= 1 for h in hs)


def test_quarter_conventions():
    assert quarter_label(date(2026, 12, 31), "indian_fy") == "Q3 FY27"
    assert quarter_label(date(2027, 6, 30), "indian_fy") == "Q1 FY28"
    assert quarter_label(date(2026, 12, 31), "calendar") == "Q4 2026"


def test_prediction_session_never_targets_an_opened_session():
    cal = calendar_from_sessions(weekdays(date(2026, 9, 1), 60))
    # 2026-10-01 09:15 IST = 03:45 UTC
    assert prediction_session(datetime(2026, 10, 1, 2, 0), cal) == date(2026, 10, 1)    # before open
    assert prediction_session(datetime(2026, 10, 1, 3, 45), cal) == date(2026, 10, 1)   # at open
    assert prediction_session(datetime(2026, 10, 1, 9, 0), cal) == date(2026, 10, 2)    # after open
    assert prediction_session(datetime(2026, 10, 2, 12, 0), cal) == date(2026, 10, 5)   # Fri evening -> Mon


def test_horizon_on_non_trading_day_uses_next_session():
    cal = calendar_from_sessions(weekdays(date(2026, 9, 1), 60))
    hs = generate_horizons(date(2026, 10, 3), cal)   # Saturday
    assert hs[0].target_date == date(2026, 10, 5) and hs[0].label.startswith("Next session")
