"""Regression tests for the fixes made after the May-Sep 2026 real-data backtest
(relative sentiment, event context, availability integrity, baselines, chronological tuning).
All data here is SYNTHETIC test data."""
from __future__ import annotations

import dataclasses
import random
from datetime import date, datetime, time, timedelta

import numpy as np
import pytest

from backend.services import parameters as params_svc
from backend.services.calendar import calendar_from_sessions, prediction_session
from backend.services.horizons import generate_horizons
from backend.services.timeutil import market_close_utc, market_open_utc
from backend.system1_news.aggregation import AnalysisItem, news_flow
from backend.system1_news.availability import classify, effective_available_at, timestamp_precision
from backend.system1_news.pipeline import group_items, score_horizons, score_news_flow
from backend.system1_news.providers.base import ValidationError, validate_article
from backend.system1_news.relevance import is_price_report
from backend.system1_news.sentiment import LexiconSentimentEngine
from backend.system2_prediction.linear import fit_logistic, fit_ridge
from backend.system2_prediction.predictor import build_predictions
from backend.system3_backtesting.backtester import BacktestData, evaluation_dates, run_backtest
from backend.system3_backtesting.experiments import run_experiment
from backend.system3_backtesting.metrics import compute_metrics, feature_analysis, paired_improvement_test, t_sf
from backend.system3_backtesting.tuner import chronological_split, tune
from tests.conftest import COMPANIES, raw, synthetic_market, weekdays
from tests.test_system3 import _data_with_signal

ENGINE = LexiconSentimentEngine()


def it(nid, entity, sentiment, published_at, fetched_at=None, price_report=False, relevance=1.0, cluster=None):
    return AnalysisItem(news_id=nid, cluster_id=cluster or nid, entity=entity, is_macro=entity == "MACRO",
                        category="other", relevance=relevance, sentiment=sentiment, strength=1.0, reliability=1.0,
                        half_life_days=5.0, published_at=published_at, source=f"src{nid}", title=f"t{nid}",
                        fetched_at=fetched_at, price_report=price_report)


# ------------------------------------------------------------------------------------ sentiment
@pytest.mark.parametrize("headline,sign", [
    ("Reliance Q2 profit beats estimates", 1),                       # positive event
    ("L&T wins a large contract worth Rs 5,000 crore", 1),           # positive event
    ("ITC faces regulatory action", -1),                             # negative event
    ("TCS profit rises but misses estimates", -1),                   # contextual negative
    ("Inflation reaches a 16-month high", -1),                       # positive word, negative impact
    ("HDFC Bank revenue falls less than expected", 1),               # negative word, positive surprise
    ("Axis Bank posts better-than-expected Q1 net profit", 1),
    ("Infosys revenue rises less than expected", -1),
])
def test_contextual_sentiment(headline, sign):
    assert ENGINE.score(headline, None).score * sign > 15, headline


def test_contrast_clause_dominates():
    plain = ENGINE.score("Profit rises", None).score
    contrast = ENGINE.score("Profit rises but misses estimates", None).score
    assert plain > 0 > contrast


@pytest.mark.parametrize("title,expected", [
    ("HDFC Bank shares fall 3% after Q1 results", True),
    ("Infosys slips 7% to 52-week low on weak guidance", True),
    ("Sensex jumps 500 points, Nifty ends above 25,000", True),
    ("Bharti Airtel stock hits record high", True),
    ("Infosys profit rises 12% on strong deal wins", False),
    ("SBI net profit hits record high in Q1", False),
    ("RBI holds repo rate at 5.5%", False),
    ("ITC announces dividend", False),
])
def test_price_report_detection(title, expected):
    assert is_price_report(title) is expected


# --------------------------------------------------------------------------- relative sentiment
def _baseline_items(entity, start_id, values, end, n_repeat=3):
    out, nid = [], start_id
    for k in range(n_repeat):
        for j, v in enumerate(values):
            nid += 1
            out.append(it(nid, entity, v, end - timedelta(days=2 + 3 * (k * len(values) + j))))
    return out, nid


def test_relative_sentiment_is_measured_against_the_company_baseline(params):
    p1 = params["system1"]
    window_start = datetime(2026, 6, 1, 3, 45)
    as_of = window_start + timedelta(days=1)
    base, nid = _baseline_items("HDFCBANK", 0, [20, 25, 15, 30], window_start)     # normal tone: +22.5
    new = [it(nid + 1, "HDFCBANK", 35, window_start + timedelta(hours=5))]
    f = news_flow("HDFCBANK", base + new, window_start, as_of, p1)
    assert f.status == "ok" and f.baseline_n == 12
    assert f.baseline_mean == pytest.approx(22.5)
    assert f.relative_sentiment == pytest.approx(12.5)          # NOT +35: only the surprise counts
    assert f.sentiment_zscore == pytest.approx(12.5 / f.baseline_std, rel=1e-3)
    assert f.news_count == 1


def test_relative_sentiment_no_new_news_is_zero_and_insufficient_history_is_missing(params):
    p1 = params["system1"]
    ws = datetime(2026, 6, 1, 3, 45)
    base, _ = _baseline_items("INFY", 0, [10, 20, 30, 40], ws)
    quiet = news_flow("INFY", base, ws, ws + timedelta(days=1), p1)
    assert quiet.status == "no_new_news" and quiet.relative_sentiment == 0.0
    thin = news_flow("INFY", base[:3] + [it(99, "INFY", 50, ws + timedelta(hours=1))], ws, ws + timedelta(days=1), p1)
    assert thin.status == "insufficient_history" and thin.relative_sentiment is None and thin.sentiment_zscore is None
    # pooled universe baseline is used when the company's own history is too thin
    pooled = news_flow("INFY", base[:3] + [it(99, "INFY", 50, ws + timedelta(hours=1))], ws, ws + timedelta(days=1),
                       p1, pooled_baseline=[0.0, 10.0] * 6)
    assert pooled.status == "ok_pooled_baseline" and pooled.relative_sentiment == pytest.approx(45.0)


def test_relative_sentiment_never_uses_future_news(params):
    p1 = params["system1"]
    ws = datetime(2026, 6, 1, 3, 45)
    as_of = ws + timedelta(days=1)
    base, nid = _baseline_items("TCS", 0, [10, 20, 30, 40], ws)
    new = [it(nid + 1, "TCS", 60, ws + timedelta(hours=3))]
    ref = news_flow("TCS", base + new, ws, as_of, p1)
    future = [it(500, "TCS", -100, as_of + timedelta(minutes=1)), it(501, "TCS", -100, as_of + timedelta(days=3))]
    assert news_flow("TCS", base + new + future, ws, as_of, p1) == ref


def test_price_reports_are_excluded_from_relative_sentiment(params):
    p1 = params["system1"]
    assert p1["price_report_weight"] == 0.0
    ws = datetime(2026, 6, 1, 3, 45)
    base, nid = _baseline_items("SBIN", 0, [0, 10, -10, 5], ws)
    pr = it(nid + 1, "SBIN", 90, ws + timedelta(hours=2), price_report=True)
    f = news_flow("SBIN", base + [pr], ws, ws + timedelta(days=1), p1)
    assert f.news_count == 0 and f.price_reports_excluded == 1 and f.relative_sentiment == 0.0


# ----------------------------------------------------------------------------------- prediction
def _predict(params, items, sentiment_input="relative", n=320, d_index=250):
    tickers = [c.ticker for c in COMPANIES]
    closes, idx, days = synthetic_market(tickers, n=n)
    cal = calendar_from_sessions(days)
    d = days[d_index]
    as_of = market_open_utc(d)
    hz = [h for h in generate_horizons(d, cal) if h.calendar_date <= days[-1]]
    grouped = group_items(items(d))
    scores = score_horizons(grouped, COMPANIES, as_of, hz, params["system1"])
    flows = score_news_flow(grouped, COMPANIES, market_open_utc(cal.previous_session(d)), as_of, params["system1"])
    p2 = {**params["system2"], "sentiment_input": sentiment_input}
    rows = build_predictions(scores, COMPANIES, closes, idx, d, p2, flows=flows)
    return {(r.ticker, r.horizon_type): r for r in rows}, d


def _tone(d, new_value, base_value=30.0):
    """HDFCBANK: 20 baseline stories at +base_value, plus one new story at +new_value."""
    ws = market_open_utc(d) - timedelta(days=7)          # safely before the previous session's open
    items = [it(i, "HDFCBANK", base_value, ws - timedelta(days=1 + 2 * i)) for i in range(20)]
    if new_value is not None:
        items.append(it(99, "HDFCBANK", new_value, market_open_utc(d) - timedelta(hours=2)))
    return items


@pytest.mark.parametrize("new_value,expected", [(95, "UP"), (-40, "DOWN"), (30, "NEUTRAL"), (None, "NEUTRAL")])
def test_factor_v2_direction_follows_relative_not_raw_sentiment(params, new_value, expected):
    rows, _ = _predict(params, lambda d: _tone(d, new_value))
    assert rows[("HDFCBANK", "today")].predicted_direction == expected
    assert rows[("HDFCBANK", "today")].sentiment_score > 0       # the raw tone is positive in every case


def test_habitually_positive_tone_no_longer_forces_up(params):
    """The May-Sep 2026 failure mode: constant positive headlines -> UP under raw input."""
    raw_rows, _ = _predict(params, lambda d: _tone(d, 30), sentiment_input="raw")
    rel_rows, _ = _predict(params, lambda d: _tone(d, 30), sentiment_input="relative")
    assert raw_rows[("HDFCBANK", "today")].predicted_movement > 0
    assert rel_rows[("HDFCBANK", "today")].contributions["company_sentiment"] == 0


def test_zero_sentiment_everywhere_is_neutral(params):
    rows, _ = _predict(params, lambda d: [])
    assert all(r.predicted_direction == "NEUTRAL" for k, r in rows.items() if k[1] == "today")


def test_missing_and_stale_price_are_reported_not_faked(params):
    closes, idx, days = synthetic_market(["HDFCBANK"])
    cal = calendar_from_sessions(days)
    d = days[250]
    hz = generate_horizons(d, cal)[:1]
    scores = score_horizons({}, COMPANIES, market_open_utc(d), hz, params["system1"])
    rows = build_predictions(scores, COMPANIES, closes, idx, d, params["system2"])
    assert {r.ticker: r.status for r in rows}["RELIANCE"] == "MISSING_DATA"
    stale = {"HDFCBANK": closes["HDFCBANK"].drop(days[249])}
    rows = build_predictions(scores, COMPANIES[:1], stale, idx, d, params["system2"], expected_previous_date=days[249])
    assert rows[0].status == "MISSING_DATA" and "refresh market data" in rows[0].missing_reason


def test_prediction_session_handles_holidays_and_market_open():
    days = weekdays(date(2026, 9, 28), 10)                 # Mon 28 Sep ..
    holiday = date(2026, 10, 2)                            # treat Fri 2 Oct as an exchange holiday
    cal = calendar_from_sessions([d for d in days if d != holiday])
    thu = date(2026, 10, 1)
    assert prediction_session(market_open_utc(thu) - timedelta(minutes=1), cal) == thu   # before 09:15 IST
    assert prediction_session(market_open_utc(thu) + timedelta(minutes=1), cal) == date(2026, 10, 5)  # opened
    assert prediction_session(datetime(2026, 10, 2, 1, 0), cal) == date(2026, 10, 5)  # holiday -> Monday
    assert prediction_session(datetime(2026, 10, 3, 12, 0), cal) == date(2026, 10, 5)  # weekend -> Monday


def test_linear_model_fit_and_multi_horizon_uses_shock_only(params):
    rng = np.random.default_rng(1)
    x = rng.normal(0, 2, 400)
    y = 0.3 + 0.5 * x + rng.normal(0, 0.1, 400)
    m = fit_ridge([{"previous_return": v} for v in x], list(y), ["previous_return"], alpha=1e-6)
    assert m.intercept == pytest.approx(y.mean())
    assert m.predict({"previous_return": 2.0}) == pytest.approx(0.3 + 1.0, abs=0.05)
    assert m.predict({"previous_return": None}) == pytest.approx(m.intercept)       # imputed with training mean
    assert fit_ridge([{}] * 3, [1.0, 2.0, 3.0], [], 1.0).predict({}) == pytest.approx(2.0)
    lg = fit_logistic([{"previous_return": v} for v in x], [int(v > 0) for v in y], ["previous_return"], 0.01)
    assert lg.predict({"previous_return": 3.0}) > 0.9 and lg.predict({"previous_return": -3.0}) < 0.1
    p2 = {**params["system2"], "model": "linear", "linear": m.to_dict()}
    rows, _ = _predict({**params, "system2": p2}, lambda d: [])
    today, month = rows[("HDFCBANK", "today")], rows[("HDFCBANK", "month_end")]
    shock = today.predicted_movement - m.intercept
    assert month.predicted_movement == pytest.approx(shock, abs=1e-3)
    assert today.contributions["intercept"] == pytest.approx(m.intercept, abs=1e-4)


# --------------------------------------------------------------------------------- availability
def test_coarse_timestamps_are_available_only_from_the_end_of_their_date():
    stamp = datetime(2026, 6, 1, 7, 0, 0)
    assert timestamp_precision(stamp, "google_news_rss") == "date_only"
    assert timestamp_precision(stamp) == "hour"
    assert timestamp_precision(datetime(2026, 6, 1, 7, 13)) == "minute"
    assert timestamp_precision(datetime(2026, 6, 1, 7, 13, 5)) == "exact"
    assert effective_available_at(stamp, "date_only") == datetime(2026, 6, 2, 0, 0)
    assert effective_available_at(stamp, "hour") == datetime(2026, 6, 1, 8, 0)       # end of the hour
    assert effective_available_at(datetime(2026, 6, 1, 7, 13, 5)) == datetime(2026, 6, 1, 7, 13, 5)


def test_availability_status_classification():
    cal = calendar_from_sessions(weekdays(date(2026, 6, 1), 10))        # Mon 1 Jun ..
    tue = date(2026, 6, 2)
    exact = datetime(2026, 6, 1, 12, 34)                                 # Monday afternoon -> first usable Tue
    assert classify(exact, datetime(2026, 6, 1, 13, 0), cal) == "verified_pre_market"
    assert classify(exact, market_open_utc(tue) + timedelta(hours=1), cal) == "published_before_cutoff"
    assert classify(exact, market_close_utc(tue) + timedelta(days=30), cal) == "collected_after_event"
    coarse = datetime(2026, 6, 1, 7, 0)
    assert classify(coarse, datetime(2026, 6, 1, 20, 0), cal, "date_only") == "verified_pre_market"
    assert classify(coarse, datetime(2026, 9, 1, 0, 0), cal, "date_only") == "unknown"
    assert classify(exact, None, cal) == "unknown"


def test_future_timestamps_are_rejected():
    now = datetime(2026, 6, 1, 3, 0)
    with pytest.raises(ValidationError, match="future"):
        validate_article(raw("Infosys wins deal", now + timedelta(hours=2)), now)
    assert validate_article(raw("Infosys wins deal", now - timedelta(hours=2)), now)


def test_verified_mode_drops_news_fetched_after_the_cutoff():
    closes, idx, days = synthetic_market(["HDFCBANK"], n=60)
    d = days[50]
    as_of = market_open_utc(d)
    pub = as_of - timedelta(hours=3)
    held = it(1, "HDFCBANK", 50, pub, fetched_at=pub + timedelta(minutes=5))
    late = it(2, "HDFCBANK", 50, pub, fetched_at=as_of + timedelta(days=90))
    data = BacktestData([held, late], COMPANIES[:1], closes, idx)
    assert {i.news_id for i in data.items_window(as_of, 30)["HDFCBANK"]} == {1, 2}
    assert {i.news_id for i in data.items_window(as_of, 30, "verified")["HDFCBANK"]} == {1}


# ---------------------------------------------------------------------------------- backtesting
def test_target_previous_close_and_baselines_are_point_in_time(params):
    data, days = _data_with_signal(params)
    d = days[200]
    recs = run_backtest(data, [d], params, ("today",))
    for r in recs:
        s = data.closes[r.ticker]
        assert r.target_date == d.isoformat() and r.previous_close_date == days[199].isoformat()
        assert r.actual == pytest.approx((s.loc[d] / s.loc[days[199]] - 1) * 100, abs=1e-3)
        assert r.baseline("no_change") == 0.0
        past = (s.pct_change() * 100).loc[days[1]:days[199]]
        assert r.baseline("company_mean") == pytest.approx(past.mean(), abs=1e-4)
        assert r.baseline("rolling_mean_20d") == pytest.approx(past.iloc[-20:].mean(), abs=1e-4)
    tampered = {t: s.copy() for t, s in data.closes.items()}
    for s in tampered.values():
        s.loc[s.index >= d] *= 3
    recs2 = run_backtest(BacktestData(data.items, data.companies, tampered, data.index_closes), [d], params, ("today",))
    assert [r.baselines for r in recs2] == [r.baselines for r in recs]
    assert [r.predicted for r in recs2] == [r.predicted for r in recs]


def test_metrics_include_baselines_confusion_and_features(params):
    data, days = _data_with_signal(params, effect=0.03)
    recs = run_backtest(data, days[100:160], params, ("today",))
    m = compute_metrics(recs)
    cm = m["confusion_matrix"]
    assert sum(cm[p][a] for p in cm for a in cm[p]) == len(recs)
    assert set(m["baseline_comparison"]["baselines"]) == {"no_change", "historical_mean", "company_mean",
                                                          "rolling_mean_20d"}
    fa = feature_analysis(recs)
    assert fa["relative_sentiment"]["n"] + fa["relative_sentiment"]["missing"] == len(recs)
    assert fa["relative_sentiment"]["correlation"] > 0.2          # the synthetic news effect is detected


def test_chronological_split_never_shuffles():
    ds = [date(2026, 5, 1) + timedelta(days=i) for i in range(100)]
    tr, va, te, _ = chronological_split(list(reversed(ds)))
    assert max(tr) < min(va) and max(va) < min(te) and len(tr) + len(va) + len(te) == 100
    tr, va, te, _ = chronological_split(ds, date(2026, 6, 30), date(2026, 7, 31))
    assert max(tr) == date(2026, 6, 30) and min(va) == date(2026, 7, 1) and min(te) == date(2026, 8, 1)


def test_paired_improvement_test():
    assert t_sf(2.086, 20) == pytest.approx(0.025, abs=1e-3)
    dates = [f"d{i // 5}" for i in range(100)]
    actual = [((i * 37) % 11 - 5) / 3 for i in range(100)]
    good = [a + 0.1 for a in actual]
    bad = [a + 1.0 + 0.2 * ((i * 7) % 3) for i, a in enumerate(actual)]
    assert paired_improvement_test(good, bad, actual, dates)["significant"]
    assert not paired_improvement_test(bad, good, actual, dates)["significant"]
    assert not paired_improvement_test(good, good, actual, dates)["significant"]


# ------------------------------------------------------------------------------------- tuning
def test_tuner_selection_is_blind_to_the_test_period(params):
    data, days = _data_with_signal(params, effect=0.03)
    dates = evaluation_dates(data, days[60], days[-1], step=2)
    space = {"system2.sensitivity": [0.0, 3.0]}
    weak = {**params, "system2": {**params["system2"], "sensitivity": 0.05}}
    a = tune(data, dates, weak, space, n_trials=6)
    test_start = date.fromisoformat(a.test_dates[0])
    rng = random.Random(0)
    scrambled = {t: s.copy() for t, s in data.closes.items()}
    for s in scrambled.values():
        for k in s.index:
            if k >= test_start:
                s.loc[k] *= 1 + rng.uniform(-0.2, 0.2)
    b = tune(BacktestData(data.items, data.companies, scrambled, data.index_closes), dates, weak, space, n_trials=6)
    assert a.best_params == b.best_params                   # same choice
    assert a.best_validation["mae"] == b.best_validation["mae"]
    assert a.best_test["mae"] != b.best_test["mae"]         # only the (untouched) test score moved
    assert a.test_dates[2] > 0 and a.split_method


def test_candidate_creation_versioning_and_explicit_activation(session):
    active, params = params_svc.get_active(session)
    cand_params = {**params, "system2": {**params["system2"], "sensitivity": 0.9}}
    cand = params_svc.create_candidate(session, cand_params, active, "test", None, {})
    assert cand.status == "candidate" and cand.version.startswith("tuned-")
    assert params_svc.get_active(session)[0] == active                 # never auto-activated
    assert params_svc.create_candidate(session, cand_params, active, "test", None, {}).version == cand.version
    params_svc.activate(session, cand.version)
    assert params_svc.get_active(session)[0] == cand.version
    assert params_svc.get_set(session, active).status == "archived"


def test_new_default_version_is_seeded_as_candidate_when_a_set_is_active(session, monkeypatch):
    active = params_svc.get_active(session)[0]
    monkeypatch.setattr(params_svc, "DEFAULT_VERSION", "default-test-next")
    params_svc.ensure_default(session)
    assert params_svc.get_set(session, "default-test-next").status == "candidate"
    assert params_svc.get_active(session)[0] == active


def test_legacy_parameter_sets_keep_their_original_behaviour():
    legacy = {"system1": {"min_relevance": 0.3}, "system2": {"w_company": 1.0}}
    full = params_svc.complete(legacy)
    assert full["system2"]["sentiment_input"] == "raw" and full["system1"]["price_report_weight"] == 1.0
    assert "sentiment_input" not in legacy["system2"]            # input untouched


def test_experiment_selection_ignores_test_outcomes(params):
    data, days = _data_with_signal(params, effect=0.03)
    recs = run_backtest(data, days[60:220], params, ("today",))
    a = run_experiment(recs)
    test_dates = {r.as_of_date for r in recs if r.as_of_date >= a["periods"]["test"]["start"]}
    rng = random.Random(5)
    noisy = [dataclasses.replace(r, actual=rng.gauss(0, 3)) if r.as_of_date in test_dates else r for r in recs]
    b = run_experiment(noisy)
    assert a["selected_feature_set"] == b["selected_feature_set"]
    assert {k: v["alpha"] for k, v in a["models"].items()} == {k: v["alpha"] for k, v in b["models"].items()}
    assert a["periods"]["train"]["end"] < a["periods"]["validation"]["start"] <= a["periods"]["validation"]["end"] \
        < a["periods"]["test"]["start"]
