"""Collective sentiment scores. Pure functions: no DB access, so System 3 can replay them exactly.

For an entity e, information cut-off `as_of` and evaluation time `target` (>= as_of):

  1. Use only analyses with  published_at <= as_of  and  age(as_of) <= max_lookback_days.
  2. Collapse near-duplicate clusters (same story, many outlets) into one item:
        s_cluster = mean(s_i weighted by reliability_i * relevance_i)
  3. Weight each cluster:
        w = reliability * relevance * strength * decay * (1 + corroboration_bonus * ln(cluster_size))
        decay = 0.5 ** (age_days(target) / (half_life_days(category) * recency_half_life_scale))
  4. Weighted mean and evidence shrinkage:
        S = sum(w * s) / sum(w);   W = sum(w)
        score = S * W / (W + shrinkage_k)          (little evidence -> pulled toward 0)

NIFTY scores (kept separate, never mixed silently):
  company_weighted = sum(nifty_weight_c * score_c) / sum(nifty_weight_c)   over active companies
  macro            = aggregate of MACRO analyses
  overall          = (1 - macro_weight_in_nifty) * company_weighted + macro_weight_in_nifty * macro

Price reports ("shares fall 3%") have their weight multiplied by `price_report_weight`: they restate
a move that is already in the previous close.

RELATIVE SENTIMENT (news_flow) - why: the score above is a long, decayed average, i.e. mostly the
entity's habitual headline tone. Company headlines skew positive (and macro headlines negative), so
the raw score is a near-constant offset that pushed almost every prediction UP in the 2026 backtest.
What can move a price is news that DIFFERS from what is normally said about the company:

  new news   N = clusters available in (window_start, as_of]       (since the previous cut-off)
  baseline   B = clusters available in (window_start - baseline_days, window_start]
  current            = weighted mean sentiment of N            (None if N is empty)
  baseline_mean/std  = mean / std of per-cluster sentiment in B (needs >= min_baseline_n clusters)
  relative_sentiment = current - baseline_mean                 (0 when there is no new news)
  sentiment_zscore   = relative_sentiment / baseline_std
If a company has fewer than min_baseline_n baseline stories, the POOLED baseline of all universe
companies over the same window is used instead (status "ok_pooled_baseline": it still removes the
general positive skew of company headlines). If that is also too thin the relative sentiment is
None ("insufficient_history") - never guessed.
Everything is computed from items available at or before `as_of`; the baseline window ends where
the new-news window starts, so a story never forms part of its own baseline.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass(frozen=True)
class AnalysisItem:
    news_id: int
    cluster_id: int
    entity: str
    is_macro: bool
    category: str
    relevance: float
    sentiment: float
    strength: float
    reliability: float
    half_life_days: float
    published_at: datetime                 # effective availability time (see analyzer.load_items)
    source: str = ""
    title: str = ""
    fetched_at: datetime | None = None     # when this system stored the article
    price_report: bool = False
    availability: str = "unknown"          # article-level status (availability.py)
    provider_published_at: datetime | None = None   # provider stamp as stored (not the effective time)
    timestamp_precision: str = "exact"
    timezone_status: str = "explicit"
    engine: str = ""
    event_type: str = ""

    def eligibility(self, cutoff: datetime) -> str:
        """verified | timestamp_before | post_cutoff | uncertain for a prediction cut-off."""
        from backend.system1_news.availability import eligibility
        return eligibility(self.provider_published_at or self.published_at, self.timestamp_precision,
                           self.timezone_status, self.fetched_at, cutoff)


@dataclass
class AggregateScore:
    entity: str
    score: float
    raw_mean: float | None
    evidence_weight: float
    news_count: int                       # distinct story clusters used
    article_count: int                    # articles incl. duplicates
    contributors: list[dict] = field(default_factory=list)
    by_category: dict[str, dict] = field(default_factory=dict)

    def to_details(self) -> dict:
        return {"article_count": self.article_count, "contributors": self.contributors,
                "by_category": self.by_category}


def _cluster(items: list[AnalysisItem]) -> list[tuple[AnalysisItem, float, int]]:
    """Returns (representative, cluster_sentiment, cluster_size) per cluster."""
    groups: dict[int, list[AnalysisItem]] = defaultdict(list)
    for it in items:
        groups[it.cluster_id].append(it)
    out = []
    for members in groups.values():
        rep = min(members, key=lambda m: (m.published_at, m.news_id))
        ws = [max(1e-9, m.reliability * m.relevance) for m in members]
        s = sum(w * m.sentiment for w, m in zip(ws, members)) / sum(ws)
        best_rel = max(members, key=lambda m: m.relevance * m.reliability)
        rep = AnalysisItem(**{**rep.__dict__, "relevance": best_rel.relevance,
                              "reliability": best_rel.reliability})
        out.append((rep, s, len({m.source for m in members}) or 1))
    return out


def aggregate(entity: str, items: list[AnalysisItem], as_of: datetime, target: datetime,
              p1: dict, top_n: int = 5) -> AggregateScore:
    max_age = p1["max_lookback_days"]
    usable = [it for it in items if it.entity == entity and it.published_at <= as_of
              and it.relevance >= p1["min_relevance"]
              and (as_of - it.published_at).total_seconds() / 86400 <= max_age]
    clusters = _cluster(usable)
    num = wsum = 0.0
    contribs, cat_num, cat_w = [], defaultdict(float), defaultdict(float)
    pr_w = p1.get("price_report_weight", 1.0)
    for rep, s, size in clusters:
        age_days = max(0.0, (target - rep.published_at).total_seconds() / 86400)
        hl = max(1e-6, rep.half_life_days * p1["recency_half_life_scale"])
        decay = 0.5 ** (age_days / hl)
        w = (rep.reliability * rep.relevance * rep.strength * decay
             * (1 + p1["corroboration_bonus"] * math.log(max(1, size)))
             * (pr_w if rep.price_report else 1.0))
        if w <= 0:
            continue
        num += w * s
        wsum += w
        cat_num[rep.category] += w * s
        cat_w[rep.category] += w
        contribs.append({"news_id": rep.news_id, "title": rep.title, "source": rep.source,
                         "published_at": rep.published_at.isoformat() + "Z", "category": rep.category,
                         "sentiment": round(s, 2), "weight": round(w, 5), "cluster_size": size})
    k = p1["shrinkage_k"]
    if wsum <= 0:
        return AggregateScore(entity, 0.0, None, 0.0, len(clusters), len(usable))
    raw_mean = num / wsum
    score = max(-100.0, min(100.0, raw_mean * wsum / (wsum + k)))
    contribs.sort(key=lambda c: -c["weight"])
    by_cat = {c: {"mean": round(cat_num[c] / cat_w[c], 2), "weight": round(cat_w[c], 5)} for c in cat_w}
    return AggregateScore(entity, round(score, 2), round(raw_mean, 2), round(wsum, 5), len(clusters),
                          len(usable), contribs[:top_n], by_cat)


@dataclass
class NewsFlow:
    """Point-in-time 'what is new' summary for one entity (see module docstring)."""
    entity: str
    status: str                        # ok | ok_pooled_baseline | no_new_news | insufficient_history
    current_sentiment: float | None    # weighted mean of the new clusters
    relative_sentiment: float | None   # current - baseline mean (0.0 if no new news)
    sentiment_zscore: float | None
    baseline_mean: float | None
    baseline_std: float | None
    baseline_n: int
    news_count: int                    # new clusters
    news_relevance: float | None       # mean relevance of the new clusters
    news_recency_hours: float | None   # hours from the newest new cluster to as_of
    price_reports_excluded: int = 0
    news_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "news_ids"}


ZSCORE_CLIP = 5.0


def baseline_sentiments(items: list[AnalysisItem], window_start: datetime, p1: dict) -> list[float]:
    """Per-cluster sentiments in the baseline window (window_start - baseline_days, window_start]."""
    lo = window_start - timedelta(days=p1.get("baseline_days", 60))
    pr_w = p1.get("price_report_weight", 1.0)
    usable = [it for it in items if lo < it.published_at <= window_start and it.relevance >= p1["min_relevance"]]
    return [s for rep, s, _ in _cluster(usable) if not (rep.price_report and pr_w <= 0)]


def news_flow(entity: str, items: list[AnalysisItem], window_start: datetime, as_of: datetime,
              p1: dict, pooled_baseline: list[float] | None = None) -> NewsFlow:
    """Pure. Uses only items with published_at (= availability time) <= as_of.
    `pooled_baseline`: universe-wide baseline sentiments, used when the entity's own is too thin."""
    base_days = p1.get("baseline_days", 60)
    min_n = p1.get("min_baseline_n", 10)
    pr_w = p1.get("price_report_weight", 1.0)
    lo = window_start - timedelta(days=base_days)
    usable = [it for it in items if it.entity == entity and lo < it.published_at <= as_of
              and it.relevance >= p1["min_relevance"]]
    new_c, base_c, excluded = [], [], 0
    for rep, s, size in _cluster(usable):
        if rep.price_report and pr_w <= 0:
            excluded += rep.published_at > window_start
            continue
        (new_c if rep.published_at > window_start else base_c).append((rep, s, size))
    b = [s for _, s, _ in base_c]
    pooled = False
    if len(b) < min_n and pooled_baseline is not None and len(pooled_baseline) >= min_n:
        b, pooled = list(pooled_baseline), True
    b_mean = sum(b) / len(b) if b else None
    b_std = math.sqrt(sum((x - b_mean) ** 2 for x in b) / (len(b) - 1)) if len(b) > 1 else None
    enough = len(b) >= min_n
    z_ok = enough and b_std is not None and b_std > 0      # a constant baseline has no z-score
    newest = max((rep.published_at for rep, _, _ in new_c), default=None)
    common = dict(entity=entity, baseline_mean=None if b_mean is None else round(b_mean, 3),
                  baseline_std=None if b_std is None else round(b_std, 3), baseline_n=len(b),
                  news_count=len(new_c), price_reports_excluded=excluded,
                  news_ids=[rep.news_id for rep, _, _ in new_c])
    if not new_c:
        return NewsFlow(status="no_new_news" if enough else "insufficient_history", current_sentiment=None,
                        relative_sentiment=0.0 if enough else None, sentiment_zscore=0.0 if z_ok else None,
                        news_relevance=None, news_recency_hours=None, **common)
    ws = [max(1e-9, rep.reliability * rep.relevance * rep.strength
              * (pr_w if rep.price_report else 1.0)) for rep, _, _ in new_c]
    cur = sum(w * s for w, (_, s, _) in zip(ws, new_c)) / sum(ws)
    rel_mean = sum(rep.relevance for rep, _, _ in new_c) / len(new_c)
    recency = (as_of - newest).total_seconds() / 3600
    if not enough:
        return NewsFlow(status="insufficient_history", current_sentiment=round(cur, 3), relative_sentiment=None,
                        sentiment_zscore=None, news_relevance=round(rel_mean, 3),
                        news_recency_hours=round(recency, 2), **common)
    rel = cur - b_mean
    z = max(-ZSCORE_CLIP, min(ZSCORE_CLIP, rel / b_std)) if z_ok else None
    return NewsFlow(status="ok_pooled_baseline" if pooled else "ok", current_sentiment=round(cur, 3),
                    relative_sentiment=round(rel, 3),
                    sentiment_zscore=None if z is None else round(z, 4), news_relevance=round(rel_mean, 3),
                    news_recency_hours=round(recency, 2), **common)


def nifty_relative(flows: dict[str, NewsFlow], weights: dict[str, float]) -> float | None:
    """NIFTY-weighted relative sentiment over companies whose relative sentiment is known."""
    known = [(weights.get(t, 0.0), f.relative_sentiment) for t, f in flows.items()
             if t != "MACRO" and f.relative_sentiment is not None]
    tw = sum(w for w, _ in known)
    return round(sum(w * r for w, r in known) / tw, 3) if tw > 0 else None


def nifty_scores(company_scores: dict[str, AggregateScore], weights: dict[str, float],
                 macro: AggregateScore, p1: dict) -> dict[str, float]:
    total_w = sum(weights.get(t, 0.0) for t in company_scores)
    cw = (sum(weights.get(t, 0.0) * s.score for t, s in company_scores.items()) / total_w) if total_w else 0.0
    mw = p1["macro_weight_in_nifty"]
    covered = sum(weights.get(t, 0.0) for t, s in company_scores.items() if s.news_count > 0)
    return {"company_weighted": round(cw, 2), "macro": macro.score,
            "overall": round((1 - mw) * cw + mw * macro.score, 2),
            "weight_coverage": round(covered / total_w, 4) if total_w else 0.0}
