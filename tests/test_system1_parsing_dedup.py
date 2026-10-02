from datetime import datetime

import pytest

from backend.database.models import NewsArticle
from backend.services.timeutil import parse_timestamp
from backend.system1_news.collector import CollectStats, store_articles
from backend.system1_news.dedup import canonical_url, find_near_duplicate, title_fingerprint, url_hash
from backend.system1_news.providers.base import ValidationError, validate_article
from backend.system1_news.providers.google_news_rss import parse_feed
from tests.conftest import raw

RSS = """<?xml version="1.0"?><rss><channel>
<item><title>HDFC Bank Q2 profit rises 12% - Reuters</title><link>https://news.google.com/a1</link>
<pubDate>Wed, 30 Sep 2026 04:30:00 GMT</pubDate><source url="https://reuters.com">Reuters</source></item>
<item><title>No date item - Mint</title><link>https://news.google.com/a2</link><source>Mint</source></item>
</channel></rss>"""


def test_rss_parsing_strips_publisher_and_keeps_utc_timestamp():
    arts = parse_feed(RSS, "HDFC Bank")
    assert len(arts) == 2
    a = arts[0]
    assert a.title == "HDFC Bank Q2 profit rises 12%"
    assert a.source == "Reuters"
    assert a.published_at == datetime(2026, 9, 30, 4, 30)
    assert a.timezone_status == "explicit" and a.published_timezone == "GMT" and a.timestamp_precision == "minute"
    # the item without pubDate is passed on with NO timestamp (never defaulted) so the collector
    # rejects and counts it
    undated = arts[1]
    assert undated.published_at is None and undated.timestamp_precision == "unknown"
    with pytest.raises(ValidationError):
        validate_article(undated)


def test_rss_date_window_filter():
    assert [a for a in parse_feed(RSS, "q", start=datetime(2026, 10, 1)) if a.published_at] == []


@pytest.mark.parametrize("value,expected", [
    ("2026-09-30T10:00:00Z", datetime(2026, 9, 30, 10, 0)),
    ("2026-09-30T15:30:00+05:30", datetime(2026, 9, 30, 10, 0)),
    ("Wed, 30 Sep 2026 10:00:00 GMT", datetime(2026, 9, 30, 10, 0)),
    (1790762400, datetime(2026, 9, 30, 10, 0)),
    ("not a date", None),
    (None, None),
])
def test_timestamp_normalisation_to_utc(value, expected):
    assert parse_timestamp(value) == expected


def test_validation_rejects_missing_fields():
    with pytest.raises(ValidationError):
        validate_article(raw("", datetime(2026, 1, 1)))
    a = raw("Title", datetime(2026, 1, 1))
    a.url = "ftp://x"
    with pytest.raises(ValidationError):
        validate_article(a)
    b = raw("Title", datetime(2026, 1, 1))
    b.published_at = None
    with pytest.raises(ValidationError):
        validate_article(b)


def test_canonical_url_removes_tracking():
    assert canonical_url("https://Site.com/a/?utm_source=x&id=5#frag") == "https://site.com/a?id=5"
    assert url_hash("https://site.com/a?utm_medium=y") == url_hash("https://site.com/a")


def test_near_duplicate_detection():
    t = datetime(2026, 9, 30, 6)
    cands = [(1, title_fingerprint("HDFC Bank Q2 net profit rises 12 percent"), t)]
    assert find_near_duplicate("HDFC Bank Q2 net profit rises 12 percent: report", t, cands) == 1
    assert find_near_duplicate("Infosys wins large deal", t, cands) is None
    # outside time window -> not a duplicate
    assert find_near_duplicate("HDFC Bank Q2 net profit rises 12 percent", datetime(2026, 10, 9), cands) is None


def test_store_articles_dedups_exact_and_near(session):
    t = datetime(2026, 9, 30, 6)
    stats = CollectStats()
    arts = [raw("HDFC Bank Q2 profit rises 12%", t, url="https://a.com/1?utm_source=g"),
            raw("HDFC Bank Q2 profit rises 12%", t, url="https://a.com/1"),          # exact dup
            raw("HDFC Bank Q2 profit rises 12% - analysis", t, source="Mint", url="https://b.com/2")]  # near dup
    stored = store_articles(session, arts, stats)
    assert stats.new == 2 and stats.exact_duplicates == 1 and stats.near_duplicates == 1
    assert stored[1].duplicate_of == stored[0].id
    assert session.query(NewsArticle).count() == 2
