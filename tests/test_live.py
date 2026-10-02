"""Real-API smoke tests. Skipped unless RUN_LIVE_TESTS=1 (needs network)."""
import os
from datetime import date, timedelta

import pytest

from backend.data.market_data import YFinanceProvider
from backend.system1_news.providers.google_news_rss import GoogleNewsRSSProvider

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1")]


def test_google_news_rss_live():
    arts = GoogleNewsRSSProvider().search('"HDFC Bank"', limit=20)
    assert arts, "no articles returned"
    assert all(a.published_at and a.url.startswith("http") for a in arts)


def test_yfinance_live():
    df = YFinanceProvider().fetch_daily("HDFCBANK.NS", date.today() - timedelta(days=30), date.today())
    assert not df.empty and (df["close"] > 0).all()
