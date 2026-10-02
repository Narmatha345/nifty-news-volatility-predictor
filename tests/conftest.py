"""Test fixtures. All data below is SYNTHETIC test data, used only to verify the mechanics."""
from __future__ import annotations

import os
import tempfile

# Keep test reports out of the real reports/ folder (must run before settings are first read).
os.environ["REPORTS_DIR"] = os.path.join(tempfile.gettempdir(), "nvp-test-reports")

import math
import random
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from backend.database import db as dbmod
from backend.services import parameters as params_svc
from backend.services.universe import seed_universe
from backend.system1_news.aggregation import AnalysisItem
from backend.system1_news.pipeline import CompanyInfo
from backend.system1_news.providers.base import NewsProvider, RawArticle

COMPANIES = [CompanyInfo("HDFCBANK", "HDFC Bank", "Financials", 13.0),
             CompanyInfo("RELIANCE", "Reliance Industries", "Energy", 8.6),
             CompanyInfo("INFY", "Infosys", "Information Technology", 4.9)]


@pytest.fixture
def session():
    dbmod.init_engine("sqlite://")
    with dbmod.session_scope() as s:
        seed_universe(s)
        params_svc.ensure_default(s)
        yield s


@pytest.fixture
def params():
    return params_svc.default_params()


def weekdays(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def synthetic_market(tickers: list[str], start: date = date(2024, 1, 1), n: int = 320, seed: int = 7,
                     news_effect: dict | None = None) -> tuple[dict[str, pd.Series], pd.Series, list[date]]:
    """Random-walk closes. news_effect[(ticker, day)] = extra log return applied ON that day."""
    rng = random.Random(seed)
    days = weekdays(start, n)
    news_effect = news_effect or {}
    idx_level, idx = 20000.0, []
    rets_idx = [rng.gauss(0, 0.009) for _ in days]
    for r in rets_idx:
        idx_level *= math.exp(r)
        idx.append(idx_level)
    closes = {}
    for k, t in enumerate(tickers):
        level, vals = 1000.0 + 100 * k, []
        for i, d in enumerate(days):
            r = 0.9 * rets_idx[i] + rng.gauss(0, 0.01) + news_effect.get((t, d), 0.0)
            level *= math.exp(r)
            vals.append(level)
        closes[t] = pd.Series(vals, index=days, dtype=float)
    return closes, pd.Series(idx, index=days, dtype=float), days


def item(news_id, entity, sentiment, published_at, category="earnings", relevance=1.0, strength=1.0,
         reliability=1.0, half_life=10.0, cluster_id=None, source="Reuters") -> AnalysisItem:
    return AnalysisItem(news_id=news_id, cluster_id=cluster_id or news_id, entity=entity,
                        is_macro=entity == "MACRO", category=category, relevance=relevance, sentiment=sentiment,
                        strength=strength, reliability=reliability, half_life_days=half_life,
                        published_at=published_at, source=source, title=f"article {news_id}")


class FakeProvider(NewsProvider):
    """In-memory provider returning fixed articles; honours start/end like a real API."""
    name = "fake"
    requires_key = False
    supports_history = True
    min_interval_s = 0

    def __init__(self, articles: list[RawArticle]):
        self.articles = articles
        self.calls = 0

    def search(self, query, start=None, end=None, limit=50):
        self.calls += 1
        return [a for a in self.articles if (start is None or a.published_at >= start)
                and (end is None or a.published_at <= end)]


def raw(title, published_at: datetime, source="Reuters", url=None, summary=None) -> RawArticle:
    # Behaves like a keyed API that documents UTC timestamps (explicit timezone).
    return RawArticle(provider="fake", source=source, title=title,
                      url=url or f"https://news.example.com/{abs(hash((title, published_at)))}",
                      published_at=published_at, summary=summary, query="test",
                      published_timezone="UTC", timezone_status="explicit")
