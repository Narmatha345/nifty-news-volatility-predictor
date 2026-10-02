"""Historical replay of System 1 + System 2 without look-ahead.

For each evaluation session d (the session being predicted):
  * information cut-off  as_of = 09:15 IST on d (market open)
  * System 1 sees only analyses whose effective availability time <= as_of (see availability.py);
    availability_mode selects which eligibility classes may be used (availability.POLICIES):
      verified            fetched_at <= cut-off only: a true live-news backtest (= live predictions)
      exact_timestamp     + precise provider timestamps before the cut-off (fetched later)
      provider_timestamp  + date-only / unknown-timezone items (Phase 2 behaviour; UNCERTAIN)
  * the universe and NIFTY weights are those effective on d (date-aware versions), never today's
  * relative sentiment compares news available in (open(previous session), as_of] with the entity's
    own earlier baseline
  * System 2 sees only closes on dates < d (previous close = base price)
  * outcome = close on the horizon's target session (must exist in price data)

TARGET (next-session horizon, "today"):
    actual_return_% = (close[d] - close[previous session]) / close[previous session] * 100
  d is an NSE session that exists in the price data, so weekends and exchange holidays are skipped
  by construction; "previous session" is the last stored session before d. Closes are Yahoo Finance
  split/dividend-adjusted closes (see data/market_data.py).

BASELINES (all computed from closes strictly before d, scaled by the horizon length h in sessions):
  no_change           0
  historical_mean     mean daily return of ALL universe stocks over all stored history before d
  company_mean        the company's mean daily return over all stored history before d
  rolling_mean_20d    the company's mean daily return over the last 20 sessions before d
Horizons whose target lies beyond the available price data are skipped, never estimated.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from backend.services.calendar import calendar_from_sessions
from backend.services.horizons import generate_horizons
from backend.services.timeutil import market_open_utc
from backend.system1_news.aggregation import AnalysisItem
from backend.system1_news.availability import POLICIES
from backend.system1_news.pipeline import CompanyInfo, score_horizons, score_news_flow
from backend.system2_prediction.predictor import build_predictions
from backend.system3_backtesting.metrics import EvalRecord

AVAILABILITY_MODES = ("provider_timestamp", "exact_timestamp", "verified")
BASELINES = ("no_change", "historical_mean", "company_mean", "rolling_mean_20d")
ROLLING_WINDOW = 20


@dataclass
class BacktestData:
    items: list[AnalysisItem]
    companies: list[CompanyInfo]
    closes: dict[str, pd.Series]        # ticker -> closes
    index_closes: pd.Series | None
    # Optional date-aware universe: [(from, to, version, companies-with-weights-effective-then)].
    # Without it `companies` is used for every date (synthetic tests).
    universe: list | None = None

    def companies_on(self, d: date) -> tuple[str | None, list[CompanyInfo]]:
        for lo, hi, version, comps in self.universe or []:
            if lo <= d <= hi:
                return version, comps
        return None, self.companies

    def __post_init__(self):
        self._by_entity: dict[str, tuple[list, list]] = {}
        for it in sorted(self.items, key=lambda i: i.published_at):
            lst, ts = self._by_entity.setdefault(it.entity, ([], []))
            lst.append(it)
            ts.append(it.published_at)
        dates = set()
        for s in self.closes.values():
            dates.update(s.index)
        self.calendar = calendar_from_sessions(sorted(dates))
        self.feature_cache: dict = {}
        # daily simple returns in %, indexed by the date of the closing price
        self.returns = {t: (s.sort_index().pct_change() * 100).dropna() for t, s in self.closes.items()}
        pooled = sorted((x, v) for s in self.returns.values() for x, v in s.items())
        self._pooled_dates = [x for x, _ in pooled]
        self._pooled_cumsum = np.cumsum([v for _, v in pooled]) if pooled else np.array([])
        self._baseline_cache: dict = {}

    def items_window(self, as_of, lookback_days: float, mode: str = "provider_timestamp"
                     ) -> dict[str, list[AnalysisItem]]:
        """Point-in-time slice: available in [as_of - lookback, as_of]. Nothing later is returned.
        mode="verified" also drops items fetched after `as_of`."""
        lo = as_of - timedelta(days=lookback_days)
        allowed = POLICIES[mode]
        out = {}
        for ent, (lst, ts) in self._by_entity.items():
            sl = lst[bisect_left(ts, lo): bisect_right(ts, as_of)]
            if mode != "provider_timestamp":
                sl = [i for i in sl if i.eligibility(as_of) in allowed]
            out[ent] = sl
        return out

    def daily_baselines(self, ticker: str, d: date) -> dict[str, float | None]:
        """Per-session expected return (%) of each baseline, from returns dated strictly before d."""
        key = (ticker, d)
        if key in self._baseline_cache:
            return self._baseline_cache[key]
        r = self.returns.get(ticker, pd.Series(dtype=float))
        past = r[[x < d for x in r.index]]
        k = bisect_left(self._pooled_dates, d)          # pooled returns dated strictly before d
        out = {"no_change": 0.0,
               "historical_mean": float(self._pooled_cumsum[k - 1] / k) if k else None,
               "company_mean": float(past.mean()) if len(past) else None,
               "rolling_mean_20d": float(past.iloc[-ROLLING_WINDOW:].mean()) if len(past) >= ROLLING_WINDOW
               else None}
        self._baseline_cache[key] = out
        return out


def evaluation_dates(data: BacktestData, start: date, end: date, step: int = 1) -> list[date]:
    sessions = [d for d in data.calendar.sessions if start <= d <= end]
    return sessions[::max(1, step)]


def run_backtest(data: BacktestData, dates: list[date], params: dict,
                 horizon_types: tuple[str, ...] = ("today", "month_end", "quarter_end"),
                 convention: str = "indian_fy", availability_mode: str = "provider_timestamp"
                 ) -> list[EvalRecord]:
    if availability_mode not in AVAILABILITY_MODES:
        raise ValueError(f"availability_mode must be one of {AVAILABILITY_MODES}")
    p1, p2 = params["system1"], params["system2"]
    cal = data.calendar
    last_session = cal.sessions[-1]
    lookback = max(p1["max_lookback_days"], p1.get("baseline_days", 60) + 7)
    records: list[EvalRecord] = []
    for d in dates:
        try:
            prev = cal.previous_session(d)
        except ValueError:
            continue
        horizons = [h for h in generate_horizons(d, cal, convention)
                    if h.horizon_type in horizon_types and h.calendar_date <= last_session]
        if not horizons:
            continue
        as_of = market_open_utc(d)
        uversion, companies = data.companies_on(d)
        window = data.items_window(as_of, lookback, availability_mode)
        scores = score_horizons(window, companies, as_of, horizons, p1)
        flows = score_news_flow(window, companies, market_open_utc(prev), as_of, p1)
        rows = build_predictions(scores, companies, data.closes, data.index_closes, d, p2,
                                 feature_cache=data.feature_cache, flows=flows)
        by_id = {i.news_id: i for lst in window.values() for i in lst}
        macro_new = flows["MACRO"].news_ids
        for r in rows:
            if r.status != "OK":
                continue
            s = data.closes[r.ticker]
            if r.target_date not in s.index:
                continue
            actual_close = float(s.loc[r.target_date])
            actual = (actual_close / r.previous_close - 1.0) * 100.0
            h = r.trading_days_ahead
            base = {k: None if v is None else round(v * h, 5)
                    for k, v in data.daily_baselines(r.ticker, d).items()}
            flow = r.features.get("news_flow") or {}
            new_ids = list((flows[r.ticker].news_ids if r.ticker in flows else [])) + list(macro_new)
            elig = [by_id[n].eligibility(as_of) for n in new_ids if n in by_id]
            quality = "NO_NEW_NEWS" if not elig else ("VERIFIED" if all(e == "verified" for e in elig)
                                                       else "UNCERTAIN")
            records.append(EvalRecord(
                as_of_date=d.isoformat(), ticker=r.ticker, horizon_type=r.horizon_type,
                horizon_label=r.horizon_label, target_date=r.target_date.isoformat(),
                trading_days_ahead=h, previous_close=r.previous_close,
                actual_close=actual_close, predicted=r.predicted_movement, actual=round(actual, 4),
                sigma_pct=r.uncertainty_1sigma_pct, predicted_direction=r.predicted_direction,
                sentiment_score=r.sentiment_score, news_count=r.features.get("company_news_count", 0),
                neutral_threshold_pct=p2["neutral_threshold_sigma"] * r.uncertainty_1sigma_pct,
                evidence_news_ids=tuple(r.evidence_news_ids),
                contributions=tuple(sorted(r.contributions.items())),
                previous_close_date=r.previous_close_date.isoformat() if r.previous_close_date else "",
                features=tuple(sorted((r.features.get("model_inputs") or {}).items())),
                baselines=tuple(sorted(base.items())),
                new_news_count=int(flow.get("news_count") or 0),
                news_flow_status=flow.get("status") or "", universe_version=uversion or "",
                data_quality_flag=quality))
    return records
