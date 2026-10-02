"""Article sentiment engines. Output: score in [-100, 100] + evidence explaining the score.

`lexicon-v1` (default) is a deterministic financial lexicon scorer:
  * word polarity weights for financial news vocabulary (Loughran-McDonald style),
  * "inverted objects": direction words flip sign next to bad-is-up nouns
    ("losses widen" -> negative, "losses narrow" -> positive, "inflation eases" -> positive),
  * negation ("not", "no", "fails to") flips the next sentiment term within 3 tokens,
  * intensifiers ("sharply", "record") scale the next term,
  * title terms count 2x, summary terms 1x,
  * score = 100 * tanh(raw / 3)  -> bounded and saturating.
It is transparent but shallow; set SENTIMENT_ENGINE=finbert for a transformer model.

`lexicon-v3` adds financial CONTEXT, because positive wording is not the same as positive impact:
  * expectation-relative phrases override the raw direction word: "better/worse than expected",
    "above/below estimates", "falls less than expected" (+), "rises less than expected" (-);
  * event phrases: "regulatory action", "show-cause notice", "tax demand" (-),
    "wins/bags/secures ... contract/order/deal" (+);
  * contrastive clauses: terms before "but"/"however" count 0.5x and after them 1.5x
    ("profit rises but misses estimates" is negative); terms after a concession word
    ("despite", "although", "though", "even as") count 0.5x.
"""
from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

POSITIVE = {
    "beat": 1.5, "beats": 1.5, "surge": 1.5, "surges": 1.5, "surged": 1.5, "soar": 1.7, "soars": 1.7,
    "jump": 1.2, "jumps": 1.2, "jumped": 1.2, "rally": 1.2, "rallies": 1.2, "gain": 0.8, "gains": 0.8,
    "rise": 0.8, "rises": 0.8, "rose": 0.8, "climb": 0.8, "climbs": 0.8, "up": 0.4, "higher": 0.6,
    "record": 0.8, "strong": 1.0, "stronger": 1.0, "robust": 1.0, "growth": 0.7, "grows": 0.7,
    "upgrade": 1.3, "upgrades": 1.3, "upgraded": 1.3, "outperform": 1.2, "buy": 0.6, "bullish": 1.3,
    "wins": 0.9, "win": 0.8, "bags": 0.9, "secures": 0.9, "approval": 0.8, "approves": 0.7,
    "expansion": 0.7, "expands": 0.7, "boost": 1.0, "boosts": 1.0, "improve": 0.8, "improves": 0.8,
    "improved": 0.8, "recovery": 0.8, "rebound": 1.0, "rebounds": 1.0, "optimistic": 1.0,
    "positive": 0.8, "profit": 0.3, "dividend": 0.5, "buyback": 0.8, "beats estimates": 1.8,
    "all-time high": 1.4, "eases": 0.6, "easing": 0.6, "cut": 0.3, "cuts": 0.3, "outperforms": 1.2,
    "high": 0.4, "inflow": 0.8, "inflows": 0.8, "upside": 0.9, "beat estimates": 1.8,
}
NEGATIVE = {
    "miss": 1.4, "misses": 1.4, "missed": 1.4, "plunge": 1.7, "plunges": 1.7, "plunged": 1.7,
    "crash": 1.8, "slump": 1.5, "slumps": 1.5, "tumble": 1.5, "tumbles": 1.5, "fall": 0.8, "falls": 0.8,
    "fell": 0.8, "drop": 0.8, "drops": 0.8, "decline": 0.8, "declines": 0.8, "down": 0.4, "lower": 0.6,
    "weak": 1.0, "weaker": 1.0, "downgrade": 1.3, "downgrades": 1.3, "downgraded": 1.3,
    "underperform": 1.2, "sell": 0.6, "bearish": 1.3, "loss": 0.8, "losses": 0.8, "penalty": 1.2,
    "fine": 0.6, "fined": 1.2, "probe": 1.1, "investigation": 1.0, "fraud": 1.8, "ban": 1.2,
    "lawsuit": 1.0, "resigns": 0.9, "concern": 0.8, "concerns": 0.8, "worry": 0.8, "worries": 0.8,
    "risk": 0.5, "risks": 0.5, "pressure": 0.7, "slowdown": 1.0, "recession": 1.5, "default": 1.4,
    "cuts guidance": 1.8, "warning": 1.0, "volatile": 0.5, "selloff": 1.4, "sell-off": 1.4,
    "hike": 0.4, "hikes": 0.4, "negative": 0.8, "disappoint": 1.2, "disappoints": 1.2, "layoffs": 1.0,
    "low": 0.4, "downturn": 1.2, "outflow": 1.0, "outflows": 1.0, "slips": 0.8, "slip": 0.7,
    "downside": 0.9, "rout": 1.4, "slides": 0.9, "sinks": 1.2, "misses estimates": 1.8,
}
# Nouns where "up" is bad: a direction word next to them flips sign.
INVERTED_OBJECTS = {"loss", "losses", "inflation", "npa", "npas", "slippages", "costs", "cost",
                    "deficit", "debt", "unemployment", "attrition", "provisions", "yields", "repo",
                    "rates", "rate", "crude", "oil"}
DIRECTION_TERMS = {"rise", "rises", "rose", "jump", "jumps", "surge", "surges", "climb", "climbs",
                   "higher", "up", "widen", "widens", "fall", "falls", "fell", "drop", "drops",
                   "decline", "declines", "narrow", "narrows", "eases", "easing", "lower", "down",
                   "cut", "cuts", "hike", "hikes", "soar", "soars", "plunge", "plunges", "high", "low",
                   "dearer", "cheaper", "spikes", "spike"}
UP_TERMS = {"rise", "rises", "rose", "jump", "jumps", "surge", "surges", "climb", "climbs", "higher",
            "up", "widen", "widens", "hike", "hikes", "soar", "soars", "high", "dearer", "spikes", "spike"}
NEGATIONS = {"not", "no", "never", "without", "fails", "failed", "unlikely"}
INTENSIFIERS = {"sharply": 1.5, "significantly": 1.4, "massive": 1.5, "steep": 1.4, "record": 1.3,
                "biggest": 1.4, "strongly": 1.3, "slightly": 0.6, "marginally": 0.6, "modest": 0.7}
_TOKEN = re.compile(r"[a-z][a-z\-']*")

_RISE = r"(?:rises?|rose|grows?|grew|increases?|increased|jumps?|jumped|surges?|surged|climbs?|climbed|gains?|gained)"
_FALL = r"(?:falls?|fell|declines?|declined|drops?|dropped|dips?|dipped|shrinks?|shrank|contracts?|slips?|slipped)"
_EXPECT = r"(?:expected|estimated|estimates|expectations|anticipated|feared|forecast)"
_ESTIMATES = r"(?:street |analyst |analysts' |market )?(?:estimates|expectations|forecasts?|consensus)"
# (label, pattern, weight). Matched spans are replaced by one sentinel token so the words inside them are
# not scored again. Order matters: longer / more specific phrases first.
PHRASES: list[tuple[str, str, float]] = [
    ("falls less than expected", rf"\b{_FALL} (?:\w+ ){{0,2}}less than {_EXPECT}", 0.8),
    ("rises less than expected", rf"\b{_RISE} (?:\w+ ){{0,2}}less than {_EXPECT}", -0.8),
    ("rises more than expected", rf"\b{_RISE} (?:\w+ ){{0,2}}more than {_EXPECT}", 1.2),
    ("falls more than expected", rf"\b{_FALL} (?:\w+ ){{0,2}}more than {_EXPECT}", -1.2),
    ("better than expected", rf"\bbetter than {_EXPECT}", 1.6),
    ("worse than expected", rf"\bworse than {_EXPECT}", -1.6),
    ("above estimates", rf"\b(?:above|beats?|tops?|exceeds?|surpass(?:es)?) {_ESTIMATES}", 1.6),
    ("below estimates", rf"\b(?:below|miss(?:es|ed)?|lags?|trails?|falls short of) {_ESTIMATES}", -1.6),
    ("in line with estimates", r"\bin line with (?:estimates|expectations)", 0.0),
    ("regulatory action", r"\bregulatory action", -1.5),
    ("show cause", r"\bshow cause(?: notice)?", -1.2),
    ("tax demand", r"\btax demand", -1.0),
    ("wins contract/order", r"\b(?:wins?|won|bags?|bagged|secures?|secured|lands?|bags) (?:\w+ ){0,4}(?:contract|order|orders|deal)s?\b", 1.3),
]
_PHRASE_RES = [(re.compile(p), w) for _, p, w in PHRASES]
_PHRASE_TOKEN = {f"zzphrase{chr(97 + i)}zz": w for i, (_, _, w) in enumerate(PHRASES)}
CONTRAST = {"but", "however"}
CONCESSION = {"despite", "although", "though"}


def _normalise(text: str) -> str:
    t = text.lower().replace("’", "'")
    t = re.sub(r"(?<=[a-z])-(?=than\b)|(?<=\bthan)-(?=[a-z])", " ", t)   # better-than-expected
    t = t.replace("show-cause", "show cause").replace("even as", "though")
    for i, (rx, _) in enumerate(_PHRASE_RES):
        t = rx.sub(f" zzphrase{chr(97 + i)}zz ", t)
    return t


def _clause_weights(toks: list[str]) -> list[float]:
    """Contrast: before 'but/however' 0.5x, after 1.5x. Concession: clause after 'despite' 0.5x."""
    w = [1.0] * len(toks)
    for i, t in enumerate(toks):
        if t in CONTRAST:
            for j in range(i):
                w[j] *= 0.5
            for j in range(i + 1, len(toks)):
                w[j] *= 1.5
            break
    for i, t in enumerate(toks):
        if t in CONCESSION:
            for j in range(i + 1, len(toks)):
                if toks[j] in CONTRAST:
                    break
                w[j] *= 0.5
    return w


@dataclass
class SentimentResult:
    score: float                       # -100..100
    raw: float
    intensity: float                   # |raw|, a proxy for event strength from wording
    engine: str
    evidence: dict = field(default_factory=dict)


class SentimentEngine(ABC):
    name = "base"

    @abstractmethod
    def score(self, title: str, summary: str | None) -> SentimentResult: ...


class LexiconSentimentEngine(SentimentEngine):
    # Bump the version whenever the lexicon changes: analyses are keyed by engine name, so old
    # scores stay traceable and articles are re-analyzed under the new version.
    name = "lexicon-v3"

    def score(self, title, summary):
        t_raw, t_terms = self._score_text(title)
        s_raw, s_terms = self._score_text(summary or "")
        raw = 2.0 * t_raw + s_raw
        score = round(100.0 * math.tanh(raw / 3.0), 2)
        return SentimentResult(score, round(raw, 3), round(abs(raw), 3), self.name,
                               {"title_terms": t_terms, "summary_terms": s_terms})

    def _score_text(self, text: str) -> tuple[float, list[str]]:
        toks = _TOKEN.findall(_normalise(text))
        clause = _clause_weights(toks)
        total, terms = 0.0, []
        i = 0
        while i < len(toks):
            tok = toks[i]
            cw = clause[i]
            tag = "" if cw == 1.0 else f"x{cw:g}"
            if tok in _PHRASE_TOKEN:
                w = _PHRASE_TOKEN[tok]
                total += self._modifiers(toks, i) * w * cw
                terms.append(f"[{PHRASES[ord(tok[8]) - 97][0]}]:{w:+.1f}{tag}")
                i += 1
                continue
            bigram = f"{tok} {toks[i + 1]}" if i + 1 < len(toks) else None
            if bigram and (bigram in POSITIVE or bigram in NEGATIVE):
                w = POSITIVE.get(bigram, 0) - NEGATIVE.get(bigram, 0)
                total += self._modifiers(toks, i) * w * cw
                terms.append(f"{bigram}:{w:+.1f}{tag}")
                i += 2
                continue
            if tok in DIRECTION_TERMS and self._near_inverted(toks, i):
                # direction applied to a bad-is-up noun
                w = -1.4 if tok in UP_TERMS else 1.4
                total += self._modifiers(toks, i) * w * cw
                terms.append(f"{tok}(inverted):{w:+.1f}{tag}")
            elif tok in POSITIVE or tok in NEGATIVE:
                w = POSITIVE.get(tok, 0) - NEGATIVE.get(tok, 0)
                total += self._modifiers(toks, i) * w * cw
                terms.append(f"{tok}:{w:+.1f}{tag}")
            i += 1
        return total, terms

    @staticmethod
    def _near_inverted(toks: list[str], i: int) -> bool:
        window = toks[max(0, i - 5): i] + toks[i + 1: i + 3]
        return any(t in INVERTED_OBJECTS for t in window)

    @staticmethod
    def _modifiers(toks: list[str], i: int) -> float:
        m = 1.0
        for t in toks[max(0, i - 3): i]:
            if t in NEGATIONS:
                m *= -1.0
            m *= INTENSIFIERS.get(t, 1.0)
        if i + 1 < len(toks):
            m *= INTENSIFIERS.get(toks[i + 1], 1.0)
        return m


class FinBERTSentimentEngine(SentimentEngine):
    """ProsusAI/finbert via transformers. score = 100 * (P(pos) - P(neg))."""
    name = "finbert-prosus"

    def __init__(self):
        from transformers import pipeline  # optional dependency
        self._pipe = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)

    def score(self, title, summary):
        text = title if not summary else f"{title}. {summary}"
        probs = {d["label"].lower(): d["score"] for d in self._pipe(text[:512])[0]}
        raw = probs.get("positive", 0.0) - probs.get("negative", 0.0)
        return SentimentResult(round(100 * raw, 2), round(raw, 4), abs(raw) * 3, self.name, {"probs": probs})


_engine_cache: dict[str, SentimentEngine] = {}


def get_engine(name: str = "lexicon") -> SentimentEngine:
    if name not in _engine_cache:
        if name == "finbert":
            _engine_cache[name] = FinBERTSentimentEngine()
        elif name == "lexicon":
            _engine_cache[name] = LexiconSentimentEngine()
        else:
            raise ValueError(f"unknown sentiment engine {name!r}")
    return _engine_cache[name]


def label_for(score: float, neutral_band: float = 15) -> str:
    if score > neutral_band:
        return "POSITIVE"
    if score < -neutral_band:
        return "NEGATIVE"
    return "NEUTRAL"
