"""Google News RSS search feed (no key). Returns headline, publisher, link and publish time.

Supports historical windows via the `after:` / `before:` search operators. Only headline metadata
is stored (no article body scraping, so article_text is always None).

Timestamp quality: pubDate is RFC-822 with an explicit GMT zone. Recent items carry a real time to
the second; older items (and most backfilled ones) come back as exactly 07:00:00 GMT, which is a
date with no time information. Those are classified `date_only`, never as a 07:00 publication.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, time, timedelta

from backend.services.http import request_with_retry
from backend.services.timeutil import TimestampInfo, parse_timestamp_info
from backend.system1_news.providers.base import NewsProvider, RawArticle

log = logging.getLogger(__name__)

FEED_URL = "https://news.google.com/rss/search"
DATE_ONLY_MARKER = time(7, 0, 0)


class GoogleNewsRSSProvider(NewsProvider):
    name = "google_news_rss"
    requires_key = False
    supports_history = True
    min_interval_s = 1.5
    timestamp_timezone = "GMT"
    timestamp_quality = ("RFC-822 pubDate in GMT. Recent items: to the second. Older/backfilled items: "
                         "date only (rendered 07:00:00 GMT). Headline only, no article text.")

    def classify_precision(self, info: TimestampInfo) -> str:
        if info.utc is not None and info.utc.time() == DATE_ONLY_MARKER:
            return "date_only"
        return info.precision

    def search(self, query, start=None, end=None, limit=50):
        q = query
        if start:
            q += f" after:{start.date().isoformat()}"
        if end:
            q += f" before:{(end.date() + timedelta(days=1)).isoformat()}"
        resp = request_with_retry("GET", FEED_URL, params={"q": q, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
                                  rate_key=self.name, min_interval_s=self.min_interval_s)
        return parse_feed(resp.text, query, start, end, self)[:limit]


def parse_feed(xml_text: str, query: str, start: datetime | None = None,
               end: datetime | None = None, provider: GoogleNewsRSSProvider | None = None) -> list[RawArticle]:
    provider = provider or GoogleNewsRSSProvider()
    out: list[RawArticle] = []
    root = ET.fromstring(xml_text)
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        info = parse_timestamp_info(item.findtext("pubDate"))
        source_el = item.find("source")
        source = source_el.text.strip() if source_el is not None and source_el.text else "unknown"
        # Google appends " - Publisher" to titles.
        if source != "unknown" and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3].strip()
        published = info.utc
        if published is not None and ((start and published < start) or (end and published > end)):
            continue
        # Items without a pubDate are returned (published_at=None) so the collector rejects and
        # COUNTS them instead of silently dropping them.
        guid = item.findtext("guid")
        out.append(provider.make_article(source, title, link, info, None, query, {"guid": guid}, guid))
    return out
