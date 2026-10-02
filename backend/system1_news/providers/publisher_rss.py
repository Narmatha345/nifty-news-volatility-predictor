"""Official publisher RSS feeds (Economic Times, Mint, Business Standard, CNBC-TV18 - see
config/news_feeds.yaml). No key, exact timestamps (RFC-822 with +0530), headline + description.

Feeds are not searchable: every feed is downloaded once per collection run (cached for 10 minutes)
and each query receives the items that mention it - a quoted company alias as a phrase, or most of the
words of a macro query. Feeds only carry recent items, so `supports_history` is False (backfills
cannot use this provider).
"""
from __future__ import annotations

import html
import logging
import re
import time as _time
import xml.etree.ElementTree as ET
from functools import lru_cache

import yaml

from backend.config.settings import CONFIG_DIR
from backend.services.http import request_with_retry
from backend.services.timeutil import parse_timestamp_info
from backend.system1_news.providers.base import NewsProvider, RawArticle

log = logging.getLogger(__name__)
CACHE_SECONDS = 600
_TAGS = re.compile(r"<[^>]+>")
_WORD = re.compile(r"[a-z0-9&]+")


@lru_cache
def feed_config() -> list[dict]:
    with open(CONFIG_DIR / "news_feeds.yaml", encoding="utf-8") as fh:
        return (yaml.safe_load(fh) or {}).get("feeds", [])


def _clean(text: str | None) -> str | None:
    if not text:
        return None
    t = " ".join(html.unescape(_TAGS.sub(" ", text)).split())
    return t or None


def parse_publisher_feed(xml_text: str, publisher: str, provider: "PublisherRSSProvider") -> list[RawArticle]:
    out = []
    root = ET.fromstring(xml_text)
    for item in root.iter("item"):
        title = _clean(item.findtext("title")) or ""
        link = (item.findtext("link") or "").strip()
        info = parse_timestamp_info(item.findtext("pubDate"))
        guid = (item.findtext("guid") or link).strip() or None
        out.append(provider.make_article(publisher, title, link, info, _clean(item.findtext("description")),
                                         None, {"guid": guid, "feed_publisher": publisher}, guid))
    return out


def matches(query: str, art: RawArticle) -> bool:
    text = f"{art.title} {art.summary or ''}".lower()
    q = query.strip()
    if q.startswith('"') and q.endswith('"'):
        phrase = q.strip('"').lower()
        return re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text) is not None
    words = [w for w in _WORD.findall(q.lower()) if len(w) >= 3]
    if not words:
        return False
    tokens = set(_WORD.findall(text))
    return sum(w in tokens for w in words) >= max(1, (len(words) + 1) // 2)


class PublisherRSSProvider(NewsProvider):
    name = "publisher_rss"
    requires_key = False
    supports_history = False
    min_interval_s = 1.0
    timestamp_timezone = None         # every pubDate carries its own +0530 offset (explicit)
    timestamp_quality = ("Publisher RSS (Economic Times, Mint, Business Standard, CNBC-TV18): RFC-822 pubDate "
                         "with explicit +0530 offset, to the second. Headline + description, no full text. "
                         "Recent items only (no history).")

    def __init__(self):
        self._cache: tuple[float, list[RawArticle]] | None = None
        self.errors: list[str] = []

    def _all_items(self) -> list[RawArticle]:
        if self._cache and _time.monotonic() - self._cache[0] < CACHE_SECONDS:
            return self._cache[1]
        items, seen, self.errors = [], set(), []
        for f in feed_config():
            try:
                resp = request_with_retry("GET", f["url"], rate_key=f"{self.name}:{f['url'].split('/')[2]}",
                                          min_interval_s=self.min_interval_s, max_retries=2)
                for a in parse_publisher_feed(resp.text, f["publisher"], self):
                    if a.url and a.url not in seen:
                        seen.add(a.url)
                        a.raw["feed_section"] = f.get("section")
                        items.append(a)
            except Exception as exc:           # one broken feed must not stop the others
                self.errors.append(f"{f['url']}: {exc}")
                log.warning("publisher feed failed %s: %s", f["url"], exc)
        self._cache = (_time.monotonic(), items)
        return items

    def search(self, query, start=None, end=None, limit=50):
        if start is not None or end is not None:
            return []            # feeds hold recent items only; never pretend to backfill
        out = []
        for a in self._all_items():
            if matches(query, a):
                out.append(RawArticle(**{**a.__dict__, "query": query, "raw": dict(a.raw)}))
        return out[:limit]
