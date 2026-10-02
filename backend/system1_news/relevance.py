"""Company identification, macro detection, categorisation and relevance scoring."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache

from backend.config.loader import taxonomy_config


@dataclass
class CompanyRef:
    ticker: str
    name: str
    sector: str
    aliases: list[str]
    case_sensitive_aliases: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)


@dataclass
class EntityMatch:
    entity: str            # ticker or "MACRO"
    is_macro: bool
    relevance: float       # 0..1
    category: str
    strength: float
    half_life_days: float
    matched: list[str]
    price_report: bool = False   # headline reports a price move that is already in the close


# --- price-report detection -------------------------------------------------------------------
# "HDFC Bank shares fall 3%", "Infosys slips 7% to 52-week low", "Sensex jumps 500 points" describe
# moves that have ALREADY happened (and are in the previous close by the time they can be used).
# "Profit rises 12%" is NOT a price report: the move word's subject is a fundamental quantity.
_MOVE_WORDS = {
    "rise", "rises", "rose", "jump", "jumps", "jumped", "gain", "gains", "gained", "surge", "surges",
    "surged", "soar", "soars", "soared", "rally", "rallies", "rallied", "climb", "climbs", "climbed",
    "advance", "advances", "advanced", "fall", "falls", "fell", "drop", "drops", "dropped", "slip", "slips",
    "slipped", "slide", "slides", "slid", "decline", "declines", "declined", "plunge", "plunges",
    "plunged", "tumble", "tumbles", "tumbled", "sink", "sinks", "sank", "crash", "crashes", "crashed",
    "tank", "tanks", "tanked", "skid", "skids", "dip", "dips", "dipped", "rebound", "rebounds", "up",
    "down", "higher", "lower", "spurt", "spurts", "zoom", "zooms",
}
_FUNDAMENTAL_NOUNS = {
    "profit", "profits", "revenue", "revenues", "sales", "income", "margin", "margins", "ebitda", "pat",
    "nii", "npa", "npas", "loans", "deposits", "orders", "order", "book", "dividend", "exports", "imports",
    "gdp", "inflation", "cpi", "wpi", "iip", "pmi", "output", "earnings", "eps", "volume", "volumes",
    "subscribers", "arpu", "production", "growth", "credit", "advances", "assets", "aum", "loss", "losses",
    "debt", "guidance", "forecast", "spending", "capex", "rate", "rates", "yield", "yields", "prices",
    "tcv", "inflows", "outflows", "premium", "premiums", "disbursements",
}
_MAGNITUDE = re.compile(r"\d+(?:\.\d+)?\s?(?:%|per ?cent\b|percent\b|pts?\b|points?\b)", re.IGNORECASE)
_PRICE_LEVEL = re.compile(r"\b(?:52-week|52 week|all-time|lifetime|record) (?:high|low)s?\b|\btop (?:gainers?|losers?)\b"
                          r"|\b(?:stocks?|shares?) to (?:watch|buy)\b|\bbuzzing stocks?\b|\bstocks? in (?:focus|news)\b"
                          r"|\bmarket cap(?:itali[sz]ation)?\b|\bm-?cap\b", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9][a-z0-9\-'&]*")


def is_price_report(title: str) -> bool:
    toks = _WORD.findall(title.lower())

    def fundamental_before(i: int) -> bool:
        return any(t in _FUNDAMENTAL_NOUNS for t in toks[max(0, i - 4): i])

    m = _PRICE_LEVEL.search(title)
    if m:
        # "profit hits record high" is fundamental news; "shares hit 52-week high" is a price report
        i = len(_WORD.findall(title[: m.start()].lower()))
        if not fundamental_before(i):
            return True
    if not _MAGNITUDE.search(title):
        return False
    return any(t in _MOVE_WORDS and not fundamental_before(i) for i, t in enumerate(toks))


def _alias_regex(alias: str, case_sensitive: bool) -> re.Pattern:
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])", flags)


@lru_cache
def _category_patterns() -> dict[str, tuple[dict, list[tuple[str, re.Pattern]]]]:
    cats = taxonomy_config()["categories"]
    return {name: (spec, [(kw, _alias_regex(kw, False)) for kw in spec["keywords"]])
            for name, spec in cats.items()}


def match_categories(text: str) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for name, (_, pats) in _category_patterns().items():
        found = [kw for kw, p in pats if p.search(text)]
        if found:
            hits[name] = found
    return hits


def _strip_excluded(text: str, patterns: list[str]) -> str:
    for p in patterns:
        text = re.sub(re.escape(p), " ", text, flags=re.IGNORECASE)
    return text


def count_mentions(text: str, company: CompanyRef) -> int:
    text = _strip_excluded(text, company.exclude_patterns)
    n = 0
    for alias in company.aliases:
        cs = alias in company.case_sensitive_aliases
        n += len(_alias_regex(alias, cs).findall(text))
    return n


def detect_entities(title: str, summary: str | None, companies: list[CompanyRef],
                    min_relevance: float = 0.3) -> list[EntityMatch]:
    """Return relevant entities for one article. Empty list -> article is unrelated (ignored)."""
    tax = taxonomy_config()
    body = summary or ""
    full = f"{title}. {body}"
    title_cats = match_categories(title)
    all_cats = match_categories(full)

    company_hits = []
    for c in companies:
        in_title = count_mentions(title, c)
        in_body = count_mentions(body, c)
        if in_title or in_body:
            company_hits.append((c, in_title, in_body))

    out: list[EntityMatch] = []
    n_companies = len(company_hits)
    price_report = is_price_report(title)
    for c, in_title, in_body in company_hits:
        rel = (0.7 if in_title else 0.4) + min(0.3, 0.1 * (in_title + in_body - 1))
        if n_companies > 2:  # market wraps / lists: less specific to any one company
            rel /= math.sqrt(n_companies - 1)
        rel = round(min(1.0, rel), 3)
        if rel < min_relevance:
            continue
        cat, spec = _pick_category(all_cats, title_cats, scope="company", title_only=price_report)
        if cat is None and price_report:
            cat = tax.get("price_action_category", "price_action")
            spec = {"base_strength": tax.get("price_action_strength", 0.3),
                    "half_life_days": tax.get("price_action_half_life_days", 2)}
        elif cat is None:
            cat = tax["default_company_category"]
            spec = {"base_strength": tax["default_company_strength"],
                    "half_life_days": tax["default_company_half_life_days"]}
        out.append(EntityMatch(c.ticker, False, rel, cat, spec["base_strength"], spec["half_life_days"],
                               all_cats.get(cat, []), price_report))

    cat, spec = _pick_category(all_cats, title_cats, scope="macro")
    if cat is not None:
        n_title = sum(len(v) for k, v in title_cats.items() if _scope(k) == "macro")
        n_all = sum(len(v) for k, v in all_cats.items() if _scope(k) == "macro")
        rel = (0.5 + 0.15 * min(n_title, 3)) if n_title else (0.25 + 0.05 * min(n_all, 3))
        if company_hits:  # company story that merely mentions macro terms
            rel *= 0.6
        rel = round(min(1.0, rel), 3)
        # An index move report ("Sensex falls 1%") is a price report unless the title also carries
        # a macro event (e.g. "... after RBI hikes repo rate").
        macro_events = [k for k in title_cats if _scope(k) == "macro" and k != "market_wrap"]
        if rel >= min_relevance:
            out.append(EntityMatch("MACRO", True, rel, cat, spec["base_strength"], spec["half_life_days"],
                                   all_cats[cat], price_report and not macro_events))
    return out


def _scope(cat: str) -> str:
    return taxonomy_config()["categories"][cat]["scope"]


def _pick_category(all_cats: dict, title_cats: dict, scope: str, title_only: bool = False):
    """Highest base_strength category of the given scope; title matches win ties.
    `title_only`: for price reports only an event named in the headline itself counts."""
    cats = taxonomy_config()["categories"]
    best = None
    for name, kws in (title_cats if title_only else all_cats).items():
        spec = cats[name]
        if spec["scope"] != scope:
            continue
        key = (name in title_cats, spec["base_strength"], len(kws))
        if best is None or key > best[0]:
            best = (key, name, spec)
    return (best[1], best[2]) if best else (None, None)
