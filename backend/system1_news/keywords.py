"""Keyword extraction: taxonomy terms (known financial vocabulary) + statistical key phrases.

Statistical phrases use a RAKE-style method: split on stopwords/punctuation, score candidate
phrases by sum(word degree / word frequency). This surfaces terms not in the taxonomy
(e.g. "semiconductor", "spectrum auction").
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from backend.system1_news.relevance import match_categories

_STOPWORDS = set("""
a about above after again against all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had has
have having he her here hers him his how i if in into is it its itself just me more most my no nor
not now of off on once only or other our ours out over own same she should so some such than that the
their them then there these they this those through to too under until up very was we were what when
where which while who whom why will with would you your says said say new may year years today week
amid set get gets top per cent percent rs crore lakh ltd limited co inc share shares stock stocks
""".split())
_SPLIT = re.compile(r"[^a-zA-Z0-9&\-\s]|\s-\s")
_WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9&\-]*")


def rake_phrases(text: str, max_words: int = 3, top_n: int = 8) -> list[str]:
    phrases: list[list[str]] = []
    for chunk in _SPLIT.split(text.lower()):
        current: list[str] = []
        for w in _WORD.findall(chunk):
            if w in _STOPWORDS or len(w) < 2:
                if current:
                    phrases.append(current)
                current = []
            else:
                current.append(w)
        if current:
            phrases.append(current)
    # Headlines have few stopwords, so long runs are common: split them into max_words windows.
    phrases = [p[i:i + max_words] for p in phrases for i in range(0, len(p), max_words)]
    freq: Counter = Counter()
    degree: dict[str, int] = defaultdict(int)
    for p in phrases:
        for w in p:
            freq[w] += 1
            degree[w] += len(p) - 1
    scored = {}
    for p in phrases:
        key = " ".join(p)
        scored[key] = sum((degree[w] + freq[w]) / freq[w] for w in p)
    return [k for k, _ in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]]


def extract_keywords(title: str, summary: str | None = None, top_n: int = 10) -> list[str]:
    text = f"{title}. {summary or ''}"
    known = [kw for kws in match_categories(text).values() for kw in kws]
    seen, out = set(), []
    for kw in known + rake_phrases(text, top_n=top_n):
        if kw not in seen and not any(kw in s for s in seen):
            seen.add(kw)
            out.append(kw)
    return out[:top_n]
