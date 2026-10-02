import math
from datetime import date, datetime

import pytest

from backend.services.calendar import calendar_from_sessions
from backend.services.horizons import generate_horizons
from backend.services.timeutil import market_open_utc
from backend.system1_news.pipeline import group_items, score_horizons
from backend.system2_prediction.features import MissingData, market_features
from backend.system2_prediction.model import predict
from backend.system2_prediction.predictor import build_predictions
from tests.conftest import COMPANIES, item, synthetic_market


def test_features_use_only_prior_closes(params):
    closes, idx, days = synthetic_market(["HDFCBANK"])
    d = days[200]
    f = market_features(closes["HDFCBANK"], idx, d, params["system2"])
    assert f.previous_close_date == days[199]
    assert f.previous_close == closes["HDFCBANK"].iloc[199]
    tampered = closes["HDFCBANK"].copy()
    tampered.iloc[200:] *= 5           # change the future: features must not move
    f2 = market_features(tampered, idx, d, params["system2"])
    assert f2.daily_vol == f.daily_vol and f2.previous_close == f.previous_close
    assert f.beta_source == "estimated" and 0.3 < f.beta < 2.0


def test_missing_history_is_reported_not_faked(params):
    closes, idx, days = synthetic_market(["HDFCBANK"], n=10)
    with pytest.raises(MissingData):
        market_features(closes["HDFCBANK"], idx, days[5], params["system2"])


def test_stale_previous_close_is_missing(params):
    closes, idx, days = synthetic_market(["HDFCBANK"])
    stale = closes["HDFCBANK"].drop(days[199])          # yesterday's close not stored yet
    with pytest.raises(MissingData, match="refresh market data"):
        market_features(stale, idx, days[200], params["system2"], expected_previous_date=days[199])


def test_estimated_price_formula_and_direction(params):
    p2 = params["system2"]
    out = predict(1650.0, 0.015, 0.0, 1.0, 1, 72, 0, 1.0, 0, p2)
    assert out.predicted_movement_pct > 0 and out.direction == "UP"
    assert out.estimated_price == pytest.approx(1650 * (1 + out.predicted_movement_pct / 100), abs=0.01)
    neg = predict(1650.0, 0.015, 0.0, 1.0, 1, -72, 0, 1.0, 0, p2)
    assert neg.direction == "DOWN" and neg.predicted_movement_pct < 0
    flat = predict(1650.0, 0.015, 0.0, 1.0, 1, 0, 0, 1.0, 0, p2)
    assert flat.direction == "NEUTRAL" and flat.predicted_movement_pct == 0


def test_prediction_is_clipped_and_uncertainty_scales_with_horizon(params):
    p2 = {**params["system2"], "sensitivity": 100.0}
    out = predict(100.0, 0.01, 0.0, 1.0, 4, 100, 0, 1.0, 0, p2)
    sigma = 0.01 * math.sqrt(4)
    assert out.predicted_movement_pct == pytest.approx((math.exp(p2["max_sigma_clip"] * sigma) - 1) * 100, rel=1e-4)
    assert out.uncertainty_1sigma_pct == pytest.approx(sigma * 100)
    short = predict(100.0, 0.01, 0.0, 1.0, 1, 0, 0, 1.0, 0, params["system2"])
    long = predict(100.0, 0.01, 0.0, 1.0, 120, 0, 0, 1.0, 0, params["system2"])
    assert long.uncertainty_1sigma_pct > short.uncertainty_1sigma_pct


def test_contributions_sum_to_prediction(params):
    out = predict(100.0, 0.012, 0.0004, 1.1, 5, 40, -20, 1.3, 10, params["system2"])
    total = sum(out.contributions_pct.values()) / 100
    assert out.predicted_movement_pct == pytest.approx((math.exp(total) - 1) * 100, rel=1e-3)


def test_build_predictions_all_horizons(params):
    """factor-v1 mechanics (raw sentiment input)."""
    params = {**params, "system2": {**params["system2"], "sentiment_input": "raw"}}
    tickers = [c.ticker for c in COMPANIES]
    closes, idx, days = synthetic_market(tickers)
    cal = calendar_from_sessions(days)
    d = days[250]
    as_of = market_open_utc(d)
    horizons = [h for h in generate_horizons(d, cal) if h.calendar_date <= days[-1]]
    items = [item(1, "HDFCBANK", 70, datetime.combine(d, datetime.min.time()))]
    scores = score_horizons(group_items(items), COMPANIES, as_of, horizons, params["system1"])
    rows = build_predictions(scores, COMPANIES, closes, idx, d, params["system2"])
    assert len(rows) == len(horizons) * len(COMPANIES)
    hdfc_today = next(r for r in rows if r.ticker == "HDFCBANK" and r.horizon_type == "today")
    infy_today = next(r for r in rows if r.ticker == "INFY" and r.horizon_type == "today")
    assert hdfc_today.status == "OK" and hdfc_today.sentiment_score > 0
    assert hdfc_today.contributions["company_sentiment"] > 0
    assert infy_today.contributions["company_sentiment"] == 0
    assert hdfc_today.evidence_news_ids == [1]
    assert {r.horizon_type for r in rows} >= {"today", "month_end"}


def test_missing_ticker_data_yields_missing_status(params):
    closes, idx, days = synthetic_market(["HDFCBANK"])
    cal = calendar_from_sessions(days)
    d = days[250]
    horizons = generate_horizons(d, cal)[:1]
    scores = score_horizons({}, COMPANIES, market_open_utc(d), horizons, params["system1"])
    rows = build_predictions(scores, COMPANIES, closes, idx, d, params["system2"])
    rel = next(r for r in rows if r.ticker == "RELIANCE")
    assert rel.status == "MISSING_DATA" and rel.predicted_movement is None and rel.predicted_direction == "MISSING"
