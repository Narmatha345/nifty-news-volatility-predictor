from datetime import datetime, time, timedelta

import pytest

from backend.services.timeutil import market_open_utc
from backend.system3_backtesting.backtester import BacktestData, evaluation_dates, run_backtest
from backend.system3_backtesting.metrics import EvalRecord, compute_metrics
from backend.system3_backtesting.reports import build_reports
from backend.system3_backtesting.tuner import tune
from backend.services.parameters import search_space
from tests.conftest import COMPANIES, item, synthetic_market


def rec(pred, act, direction=None, sigma=1.0):
    direction = direction or ("UP" if pred > 0.1 else "DOWN" if pred < -0.1 else "NEUTRAL")
    return EvalRecord("2025-01-01", "X", "today", "Today", "2025-01-01", 1, 100.0, 100 + act, pred, act, sigma,
                      direction, 0.0, 1, 0.1 * sigma)


def test_metric_calculation_by_hand():
    recs = [rec(1.0, 2.0), rec(-1.0, -0.5), rec(0.5, -1.0), rec(0.0, 0.05)]
    m = compute_metrics(recs)
    assert m["n"] == 4
    assert m["mae"]["value"] == pytest.approx((1 + 0.5 + 1.5 + 0.05) / 4)
    assert m["rmse"]["value"] == pytest.approx(((1 + 0.25 + 2.25 + 0.0025) / 4) ** 0.5, rel=1e-4)
    assert m["directional_accuracy"]["value"] == pytest.approx(3 / 4)   # UP/UP, DOWN/DOWN, UP/DOWN x, NEUTRAL/NEUTRAL
    assert m["positive_prediction_accuracy"] == {"value": 0.5, "n": 2}
    assert m["negative_prediction_accuracy"] == {"value": 1.0, "n": 1}
    assert m["neutral_prediction_accuracy"] == {"value": 1.0, "n": 1}
    assert m["mape"]["n"] == 3                                           # |actual| < 0.5% excluded
    assert m["baseline_zero_change"]["mae"] == pytest.approx((2 + 0.5 + 1 + 0.05) / 4)


def test_empty_metrics_are_explicit():
    assert compute_metrics([])["n"] == 0


def _data_with_signal(params, effect=0.02, n=320):
    """News published before the open on day d; the stock moves on day d in the news direction."""
    tickers = [c.ticker for c in COMPANIES]
    import random
    rng = random.Random(3)
    from tests.conftest import weekdays
    from datetime import date
    days = weekdays(date(2024, 1, 1), n)
    effects, items, nid = {}, [], 0
    for i, d in enumerate(days[30:], start=30):
        if rng.random() < 0.25:
            t = rng.choice(tickers)
            sign = rng.choice([1, -1])
            nid += 1
            items.append(item(nid, t, 70 * sign, datetime.combine(d, time(1, 0)), half_life=1))
            effects[(t, d)] = effect * sign
    closes, idx, days = synthetic_market(tickers, n=n, news_effect=effects)
    return BacktestData(items, COMPANIES, closes, idx), days


def test_backtest_no_lookahead(params):
    data, days = _data_with_signal(params)
    d = days[260]
    base = run_backtest(data, [d], params, ("today",))
    # (1) news published after the cut-off must not change the prediction
    late = item(99999, "HDFCBANK", -100, market_open_utc(d) + timedelta(minutes=1))
    data2 = BacktestData(data.items + [late], data.companies, data.closes, data.index_closes)
    assert [r.predicted for r in run_backtest(data2, [d], params, ("today",))] == [r.predicted for r in base]
    # (2) changing prices on/after d must not change the prediction (only the actual)
    closes3 = {t: s.copy() for t, s in data.closes.items()}
    for s in closes3.values():
        s.loc[s.index >= d] *= 1.5
    data3 = BacktestData(data.items, data.companies, closes3, data.index_closes)
    after = run_backtest(data3, [d], params, ("today",))
    assert [r.predicted for r in after] == [r.predicted for r in base]
    assert [r.actual for r in after] != [r.actual for r in base]


def test_predicted_vs_actual_alignment(params):
    data, days = _data_with_signal(params)
    recs = run_backtest(data, days[100:110], params, ("today", "month_end"))
    for r in recs:
        s = data.closes[r.ticker]
        assert r.actual_close == pytest.approx(float(s.loc[datetime.fromisoformat(r.target_date).date()]))
        prev_idx = list(s.index).index(datetime.fromisoformat(r.as_of_date).date()) - 1
        assert r.previous_close == pytest.approx(float(s.iloc[prev_idx]))
    assert {r.horizon_type for r in recs} == {"today", "month_end"}


def test_backtest_detects_real_signal_better_than_noise(params):
    data, days = _data_with_signal(params, effect=0.03)
    recs = run_backtest(data, evaluation_dates(data, days[60], days[-1]), params, ("today",))
    with_news = [r for r in recs if r.news_count > 0]   # includes days where the news is already stale
    m = compute_metrics(with_news)
    assert m["sign_accuracy"]["value"] > 0.65 and m["sign_accuracy"]["p_value_vs_coinflip"] < 0.01
    assert m["correlation"]["value"] > 0.3


def test_tuner_improves_and_never_activates(params):
    data, days = _data_with_signal(params, effect=0.03, n=640)   # long enough for a powered test period
    weak = {**params, "system2": {**params["system2"], "sensitivity": 0.05}}
    # every session: the untouched test period needs enough dates for the significance gate
    dates = evaluation_dates(data, days[60], days[-1], step=1)
    res = tune(data, dates, weak, {"system2.sensitivity": [0.0, 3.0]}, n_trials=12)
    assert res.best_params["system2"]["sensitivity"] > 0.05
    assert res.recommended, res.reason
    assert res.best_test["significance_vs_no_change"]["significant"]
    assert weak["system2"]["sensitivity"] == 0.05      # input params untouched


def test_reports_contain_required_sections(params):
    data, days = _data_with_signal(params)
    recs = run_backtest(data, days[100:140], params, ("today", "month_end"))
    meta = {"run_id": "bt-test", "generated_at": "2026-10-01T00:00:00Z", "start": str(days[100]),
            "end": str(days[139]), "param_version": "default-v1", "model_version": "factor-v1@default-v1",
            "tickers": [c.ticker for c in COMPANIES], "horizon_types": ["today", "month_end"], "n_news": len(data.items),
            "n_clusters": len(data.items), "backfilled_share": 0.0}
    js, md = build_reports(meta, recs, {i.news_id: i.title for i in data.items}, None, [])
    assert js["total_predictions"] == len(recs) and js["metrics"]["n"] == len(recs)
    assert set(js["metrics_by_horizon"]) == {"today", "month_end"}
    for section in ("Backtest setup", "Overall results", "Company-wise results", "Parameter changes",
                    "Failure cases", "Important observations"):
        assert f"## {section}" in md
