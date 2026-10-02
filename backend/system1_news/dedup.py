"""Duplicate detection.

Exact duplicates: canonical URL hash (tracking params / fragments removed) -> unique DB key.
Near duplicates: the same story syndicated under different URLs. Detected by Jaccard similarity
of title token sets within a time window. Near-duplicates are stored but linked via
`duplicate_of`; aggregation counts a cluster once (with a small corroboration bonus).
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING_PREFIXES = ("utm_", "fbclid", "gclid", "mc_", "ref", "cmpid", "ocid")
_STOP = {"a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are", "at", "by", "with",
         "as", "from", "its", "it", "be", "this", "that", "after", "over"}
_TOKEN = re.compile(r"[a-z0-9&]+")


def canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith(TRACKING_PREFIXES)]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))


def url_hash(url: str) -> str:
    return hashlib.sha256(canonical_url(url).encode("utf-8")).hexdigest()


def title_tokens(title: str) -> set[str]:
    return {t for t in _TOKEN.findall(title.lower()) if t not in _STOP and len(t) > 1}


def title_fingerprint(title: str) -> str:
    return " ".join(sorted(title_tokens(title)))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_near_duplicate(title: str, published_at: datetime,
                        candidates: list[tuple[int, str, datetime]],
                        threshold: float = 0.8, window: timedelta = timedelta(hours=48)) -> int | None:
    """candidates: (id, title_fingerprint, published_at). Returns id of the earliest match."""
    toks = title_tokens(title)
    best: tuple[datetime, int] | None = None
    for cid, fp, ts in candidates:
        if abs(ts - published_at) > window:
            continue
        if jaccard(toks, set(fp.split())) >= threshold:
            if best is None or ts < best[0]:
                best = (ts, cid)
    return best[1] if best else None
