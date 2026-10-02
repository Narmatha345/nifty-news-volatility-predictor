"""Feature generation for System 2. Market features use ONLY closes strictly before the base date."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import pandas as pd

from backend.config.loader import macro_sensitivity_config
from backend.system1_news.aggregation import AggregateScore


@dataclass
class MarketFeatures:
    previous_close: float
    previous_close_date: date
    daily_vol: float          # std of daily log returns
    daily_drift: float        # mean daily log return
    beta: float               # vs NIFTY 50
    beta_source: str          # "estimated" | "default (index data missing)"
    n_obs: int
    previous_return_pct: float | None = None   # close(t-1) / close(t-2) - 1, in %
    return_5d_pct: float | None = None         # close(t-1) / close(t-6) - 1, in %
    return_20d_pct: float | None = None        # close(t-1) / close(t-21) - 1, in %

    def to_dict(self) -> dict:
        d = asdict(self)
        d["previous_close_date"] = self.previous_close_date.isoformat()
        return d


class MissingData(Exception):
    """Raised with a human-readable reason when a feature cannot be computed from real data."""


def market_features(closes: pd.Series, index_closes: pd.Series | None, base_date: date,
                    p2: dict, expected_previous_date: date | None = None) -> MarketFeatures:
    hist = closes[[d < base_date for d in closes.index]].dropna()
    if hist.empty:
        raise MissingData(f"no closing price before {base_date}")
    if expected_previous_date is not None and hist.index[-1] != expected_previous_date:
        raise MissingData(f"previous close for {expected_previous_date} not available (latest stored close is "
                          f"{hist.index[-1]}); refresh market data")
    if len(hist) < p2["min_history_days"] + 1:
        raise MissingData(f"only {len(hist)} closes before {base_date}; need {p2['min_history_days'] + 1}")
    r = np.log(hist).diff().dropna()
    vol = float(r.iloc[-p2["vol_lookback_days"]:].std(ddof=1))
    drift = float(r.iloc[-p2["drift_lookback_days"]:].mean())
    beta, beta_src = 1.0, "default (index data missing)"
    if index_closes is not None and not index_closes.empty:
        idx = index_closes[[d < base_date for d in index_closes.index]].dropna()
        ri = np.log(idx).diff().dropna()
        joined = pd.concat([r, ri], axis=1, join="inner").iloc[-p2["beta_lookback_days"]:]
        if len(joined) >= p2["min_history_days"] and joined.iloc[:, 1].var() > 0:
            beta = float(joined.cov().iloc[0, 1] / joined.iloc[:, 1].var())
            beta_src = "estimated"
    if not math.isfinite(vol) or vol <= 0:
        raise MissingData("volatility could not be estimated")

    def ret(n: int) -> float | None:
        return float(hist.iloc[-1] / hist.iloc[-1 - n] - 1) * 100 if len(hist) > n else None

    return MarketFeatures(float(hist.iloc[-1]), hist.index[-1], vol, drift, beta, beta_src, len(hist),
                          ret(1), ret(5), ret(20))


def macro_sensitivity(sector: str, macro: AggregateScore) -> float:
    """Evidence-weighted mean sector sensitivity across the macro categories present."""
    cfg = macro_sensitivity_config()
    table = cfg.get("sectors", {}).get(sector, {})
    default = cfg.get("default", 1.0)
    total = sum(v["weight"] for v in macro.by_category.values())
    if total <= 0:
        return default
    return sum(v["weight"] * table.get(cat, default) for cat, v in macro.by_category.items()) / total


# Every feature System 2 can use. Names are stable: they key stored features, fitted linear-model
# coefficients and the System 3 feature analysis.
FEATURE_NAMES = [
    "raw_sentiment",          # System 1 company score (decayed average tone, legacy input)
    "relative_sentiment",     # new-news sentiment minus the company's own baseline
    "sentiment_zscore",       # relative_sentiment / baseline std
    "news_count",             # log(1 + new company stories since the previous cut-off)
    "news_relevance",         # mean relevance of those stories (0 if none)
    "news_recency",           # exp(-hours since newest new story / 24); 0 if none
    "macro_sentiment",        # System 1 macro score (raw)
    "macro_relative",         # macro relative sentiment x sector sensitivity
    "nifty_sentiment",        # NIFTY company-weighted raw score x beta
    "nifty_relative",         # NIFTY-weighted relative sentiment x beta
    "historical_volatility",  # daily log-return std, %
    "beta",
    "previous_return",        # last close-to-close return before the prediction, %
    "rolling_return_5d",
    "rolling_return_20d",
]


def prediction_features(raw_company: float, flow_company, flow_macro, nifty_rel: float | None,
                        raw_macro: float, raw_nifty: float, macro_sens: float, mf: MarketFeatures) -> dict:
    """Feature vector for one company at one cut-off. None = genuinely unknown (reported as missing;
    models impute the training mean). Inputs are all point-in-time (see aggregation.news_flow and
    market_features)."""
    fc = flow_company
    has_news = fc is not None and fc.news_count > 0
    return {
        "raw_sentiment": raw_company,
        "relative_sentiment": None if fc is None else fc.relative_sentiment,
        "sentiment_zscore": None if fc is None else fc.sentiment_zscore,
        "news_count": math.log1p(fc.news_count) if fc is not None else None,
        "news_relevance": (fc.news_relevance if has_news else 0.0) if fc is not None else None,
        "news_recency": (math.exp(-fc.news_recency_hours / 24) if has_news else 0.0) if fc is not None else None,
        "macro_sentiment": raw_macro,
        "macro_relative": None if flow_macro is None or flow_macro.relative_sentiment is None
        else flow_macro.relative_sentiment * macro_sens,
        "nifty_sentiment": raw_nifty * mf.beta,
        "nifty_relative": None if nifty_rel is None else nifty_rel * mf.beta,
        "historical_volatility": mf.daily_vol * 100,
        "beta": mf.beta,
        "previous_return": mf.previous_return_pct,
        "rolling_return_5d": mf.return_5d_pct,
        "rolling_return_20d": mf.return_20d_pct,
    }
