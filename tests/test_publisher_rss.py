"""Publisher RSS provider (synthetic feed XML, no network)."""
from datetime import datetime

from backend.system1_news.providers.publisher_rss import PublisherRSSProvider, matches, parse_publisher_feed

FEED = """<?xml version="1.0"?><rss><channel>
<item><title>HDFC Bank Q2 profit rises 12%</title><link>https://example.com/a1</link>
<description>&lt;p&gt;HDFC Bank reported &lt;b&gt;higher&lt;/b&gt; profit.&lt;/p&gt;</description>
<pubDate>Fri, 02 Oct 2026 08:25:49 +0530</pubDate><guid>https://example.com/a1</guid></item>
<item><title>RBI keeps repo rate unchanged</title><link>https://example.com/a2</link>
<pubDate>Fri, 02 Oct 2026 10:05:00 +0530</pubDate></item>
<item><title>No date</title><link>https://example.com/a3</link></item>
</channel></rss>"""


def test_parse_keeps_exact_ist_timestamps_and_cleans_html():
    arts = parse_publisher_feed(FEED, "The Economic Times", PublisherRSSProvider())
    a = arts[0]
    assert a.source == "The Economic Times" and a.provider == "publisher_rss"
    assert a.published_at == datetime(2026, 10, 2, 2, 55, 49)              # 08:25:49 IST -> UTC
    assert (a.timestamp_precision, a.published_timezone, a.timezone_status) == ("exact", "+05:30", "explicit")
    assert a.summary == "HDFC Bank reported higher profit." and a.provider_article_id == "https://example.com/a1"
    assert arts[1].timestamp_precision == "minute"
    assert arts[2].published_at is None                                     # rejected + counted by the collector


def test_query_matching_and_no_fake_backfill(monkeypatch):
    p = PublisherRSSProvider()
    arts = parse_publisher_feed(FEED, "Mint", p)
    monkeypatch.setattr(p, "_all_items", lambda: arts)
    assert matches('"HDFC Bank"', arts[0]) and not matches('"HDFC Bank"', arts[1])
    assert matches("RBI repo rate", arts[1]) and not matches("India GDP growth", arts[1])
    assert [a.url for a in p.search('"HDFC Bank"')] == ["https://example.com/a1"]
    assert p.search('"HDFC Bank"')[0].query == '"HDFC Bank"'
    assert p.search('"HDFC Bank"', start=datetime(2026, 9, 1), end=datetime(2026, 9, 30)) == []   # no history
    assert p.supports_history is False
