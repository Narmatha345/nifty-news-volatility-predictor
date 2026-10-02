"""Accuracy / error metrics. Every metric is reported with the sample count it was computed on."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

MAPE_MIN_ABS_ACTUAL = 0.5   # % - MAPE is meaningless when the actual move is near zero


@dataclass(frozen=True)
class EvalRecord:
    as_of_date: str
    ticker: str
    horizon_type: str
    horizon_label: str
    target_date: str
    trading_days_ahead: int
    previous_close: float
    actual_close: float
    predicted: float          # %
    actual: float             # %
    sigma_pct: float          # 1-sigma horizon uncertainty, %
    predicted_direction: str
    sentiment_score: float
    news_count: int
    neutral_threshold_pct: float
    evidence_news_ids: tuple = ()
    contributions: tuple = ()  # tuple of (name, pct) for hashability
    previous_close_date: str = ""
    features: tuple = ()       # (name, value) - System 2 model inputs at the cut-off
    baselines: tuple = ()      # (name, predicted %) - see backtester.BASELINES
    new_news_count: int = 0    # company stories that arrived since the previous cut-off
    news_flow_status: str = ""
    universe_version: str = ""
    data_quality_flag: str = ""   # VERIFIED | UNCERTAIN | NO_NEW_NEWS (new news vs the cut-off)

    @property
    def month(self) -> str:
        return self.as_of_date[:7]

    def baseline(self, name: str) -> float | None:
        return dict(self.baselines).get(name)

    def feature(self, name: str):
        return dict(self.features).get(name)

    @property
    def actual_direction(self) -> str:
        if self.actual > self.neutral_threshold_pct:
            return "UP"
        if self.actual < -self.neutral_threshold_pct:
            return "DOWN"
        return "NEUTRAL"

    @property
    def error(self) -> float:
        return self.predicted - self.actual


def _safe(x: float | None, nd: int = 4):
    return None if x is None or not math.isfinite(x) else round(float(x), nd)


def _ratio(num: int, den: int) -> dict:
    return {"value": _safe(num / den) if den else None, "n": den}


def compute_metrics(records: Iterable[EvalRecord]) -> dict:
    recs = list(records)
    n = len(recs)
    if n == 0:
        return {"n": 0, "note": "no evaluable predictions (missing price data or empty period)"}
    pred = np.array([r.predicted for r in recs])
    act = np.array([r.actual for r in recs])
    err = pred - act
    mae, rmse = float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))
    mae0, rmse0 = float(np.mean(np.abs(act))), float(np.sqrt(np.mean(act ** 2)))  # "no change" baseline

    mape_mask = np.abs(act) >= MAPE_MIN_ABS_ACTUAL
    mape = float(np.mean(np.abs(err[mape_mask] / act[mape_mask])) * 100) if mape_mask.any() else None

    corr = None
    if n >= 3 and np.std(pred) > 0 and np.std(act) > 0:
        corr = float(np.corrcoef(pred, act)[0, 1])

    dir_match = sum(r.predicted_direction == r.actual_direction for r in recs)
    signed = [r for r in recs if r.predicted_direction in ("UP", "DOWN") and r.actual != 0]
    sign_hits = sum((r.predicted > 0) == (r.actual > 0) for r in signed)
    ups = [r for r in recs if r.predicted_direction == "UP"]
    downs = [r for r in recs if r.predicted_direction == "DOWN"]
    neutrals = [r for r in recs if r.predicted_direction == "NEUTRAL"]

    return {
        "n": n,
        "directional_accuracy": _ratio(dir_match, n),
        "sign_accuracy": {**_ratio(sign_hits, len(signed)), **_binomial_vs_half(sign_hits, len(signed))},
        "positive_prediction_accuracy": _ratio(sum(r.actual > 0 for r in ups), len(ups)),
        "negative_prediction_accuracy": _ratio(sum(r.actual < 0 for r in downs), len(downs)),
        "neutral_prediction_accuracy": _ratio(sum(r.actual_direction == "NEUTRAL" for r in neutrals),
                                              len(neutrals)),
        "mae": {"value": _safe(mae), "n": n},
        "rmse": {"value": _safe(rmse), "n": n},
        "mape": {"value": _safe(mape), "n": int(mape_mask.sum()),
                 "note": f"only samples with |actual| >= {MAPE_MIN_ABS_ACTUAL}%"},
        "correlation": {"value": _safe(corr), "n": n},
        "baseline_zero_change": {"mae": _safe(mae0), "rmse": _safe(rmse0), "n": n},
        "mae_skill_vs_zero": _safe(1 - mae / mae0) if mae0 > 0 else None,
        "mean_predicted": _safe(float(pred.mean())),
        "mean_actual": _safe(float(act.mean())),
        "within_1sigma": _ratio(sum(abs(r.error) <= r.sigma_pct for r in recs), n),
        "bias": _safe(float(err.mean())),              # mean(predicted - actual)
        "prediction_distribution": direction_distribution(recs),
        "confusion_matrix": confusion_matrix(recs),
        "baseline_comparison": baseline_comparison(recs),
    }


DIRECTIONS = ("UP", "DOWN", "NEUTRAL")


def direction_distribution(recs: list[EvalRecord]) -> dict:
    n = len(recs)
    pred = {d: sum(r.predicted_direction == d for r in recs) for d in DIRECTIONS}
    act = {d: sum(r.actual_direction == d for r in recs) for d in DIRECTIONS}
    return {"n": n,
            "predicted": {d: {"count": c, "share": _safe(c / n) if n else None} for d, c in pred.items()},
            "actual": {d: {"count": c, "share": _safe(c / n) if n else None} for d, c in act.items()},
            "actual_sign": {"UP": sum(r.actual > 0 for r in recs), "DOWN": sum(r.actual < 0 for r in recs),
                            "ZERO": sum(r.actual == 0 for r in recs)},
            "neutral_band_note": "actual NEUTRAL = |actual| within the same neutral band as the prediction "
                                 "(neutral_threshold_sigma x horizon sigma)"}


def confusion_matrix(recs: list[EvalRecord]) -> dict:
    """rows = predicted direction, columns = actual direction (same neutral band)."""
    return {p: {a: sum(r.predicted_direction == p and r.actual_direction == a for r in recs)
                for a in DIRECTIONS} for p in DIRECTIONS}


def _error_stats(pred: np.ndarray, act: np.ndarray, neutral: np.ndarray) -> dict:
    err = pred - act
    pdir = np.where(pred > neutral, 1, np.where(pred < -neutral, -1, 0))
    adir = np.where(act > neutral, 1, np.where(act < -neutral, -1, 0))
    signed = (pdir != 0) & (act != 0)
    hits = int(((pred > 0) == (act > 0))[signed].sum())
    return {"mae": _safe(float(np.mean(np.abs(err)))), "rmse": _safe(float(np.sqrt(np.mean(err ** 2)))),
            "directional_accuracy": _safe(float(np.mean(pdir == adir))),
            "sign_accuracy": {**_ratio(hits, int(signed.sum())), **_binomial_vs_half(hits, int(signed.sum()))},
            "bias": _safe(float(err.mean())), "n": int(len(act))}


def baseline_comparison(recs: list[EvalRecord]) -> dict:
    """Model vs each transparent baseline on EXACTLY the same records (those where every baseline is
    defined). Skill = 1 - MAE_model / MAE_baseline (> 0 means the model is better)."""
    names = sorted({k for r in recs for k, _ in r.baselines})
    if not names:
        return {}
    usable = [r for r in recs if all(r.baseline(b) is not None for b in names)]
    if not usable:
        return {"n": 0}
    act = np.array([r.actual for r in usable])
    neutral = np.array([r.neutral_threshold_pct for r in usable])
    model = _error_stats(np.array([r.predicted for r in usable]), act, neutral)
    out = {"n": len(usable), "model": model, "baselines": {}}
    for b in names:
        st = _error_stats(np.array([r.baseline(b) for r in usable]), act, neutral)
        st["model_mae_skill"] = _safe(1 - model["mae"] / st["mae"]) if st["mae"] else None
        st["model_beats_baseline_mae"] = model["mae"] < st["mae"]
        out["baselines"][b] = st
    out["beats_all_baselines_mae"] = all(v["model_beats_baseline_mae"] for v in out["baselines"].values())
    return out


def _rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank().to_numpy()


def _corr_p(r: float | None, n: int) -> float | None:
    if r is None or n < 4 or abs(r) >= 1:
        return None
    t = r * math.sqrt((n - 2) / (1 - r * r))
    return _safe(math.erfc(abs(t) / math.sqrt(2)))   # normal approximation of the t test


def feature_analysis(recs: list[EvalRecord], features: list[str] | None = None) -> dict:
    """Univariate relationship of every model input with the realised return. Descriptive only:
    correlation is not causation, the tests are not corrected for multiple comparisons, and for
    month/quarter horizons consecutive samples overlap (use the next-session horizon)."""
    names = features or sorted({k for r in recs for k, _ in r.features})
    out = {}
    for f in names:
        pairs = [(r.feature(f), r.actual) for r in recs]
        vals = [(x, y) for x, y in pairs if x is not None and np.isfinite(x)]
        miss = len(pairs) - len(vals)
        if len(vals) < 3:
            out[f] = {"n": len(vals), "missing": miss}
            continue
        x = np.array([v[0] for v in vals], dtype=float)
        y = np.array([v[1] for v in vals], dtype=float)
        pear = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
        spear = float(np.corrcoef(_rank(x), _rank(y))[0, 1]) if x.std() > 0 and y.std() > 0 else None
        nz = (x != 0) & (y != 0)
        agree = float(np.mean(np.sign(x[nz]) == np.sign(y[nz]))) if nz.sum() >= 10 else None
        q_lo, q_hi = np.quantile(x, [0.2, 0.8])
        top, bot = y[x >= q_hi], y[x <= q_lo]
        spread = float(top.mean() - bot.mean()) if len(top) and len(bot) and q_hi > q_lo else None
        p = _corr_p(pear, len(vals))
        out[f] = {"n": len(vals), "missing": miss, "mean": _safe(float(x.mean())), "std": _safe(float(x.std())),
                  "share_nonzero": _safe(float(np.mean(x != 0))),
                  "correlation": _safe(pear), "p_value": p, "rank_correlation": _safe(spear),
                  "sign_agreement": {"value": _safe(agree), "n": int(nz.sum())},
                  "top_minus_bottom_quintile_return": _safe(spread),
                  "direction": None if pear is None else ("positive" if pear > 0 else "negative"),
                  "informative_at_5pct": p is not None and p < 0.05}
    return out


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(c) > 1e-300 else 1e-300
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def t_sf(t: float, df: int) -> float:
    """One-sided P(T > t) for Student's t with `df` degrees of freedom."""
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) + lbeta) if 0 < x < 1 else 0.0
    ibeta = front * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - front * _betacf(b, a, 1 - x) / b
    p_two_half = 0.5 * ibeta          # = P(T > |t|)
    return p_two_half if t > 0 else 1 - p_two_half


def paired_improvement_test(model: list[float], baseline: list[float], actual: list[float],
                            dates: list[str]) -> dict:
    """Is the model's absolute error SMALLER than the baseline's? Diebold-Mariano style test on the
    loss differential d = |model error| - |baseline error|, averaged per prediction DATE first (the 10
    stocks of one day share market moves, so they are not independent), then a one-sided t test
    over dates. p < 0.05 with mean < 0 = statistically significant improvement."""
    per: dict[str, list[float]] = {}
    for m, b, a, d in zip(model, baseline, actual, dates):
        per.setdefault(d, []).append(abs(m - a) - abs(b - a))
    daily = np.array([np.mean(v) for v in per.values()])
    n = len(daily)
    if n < 5:
        return {"n_dates": n, "mean_loss_diff": None, "t": None, "p_value_improvement": None,
                "significant": False}
    sd = float(daily.std(ddof=1))
    mean = float(daily.mean())
    if sd == 0:
        return {"n_dates": n, "mean_loss_diff": _safe(mean), "t": None, "p_value_improvement": None,
                "significant": False}
    t = mean / (sd / math.sqrt(n))
    p = t_sf(-t, n - 1)     # P(T < t): small when the model's loss is lower
    return {"n_dates": n, "mean_loss_diff": _safe(mean, 5), "t": _safe(t), "p_value_improvement": _safe(p),
            "significant": bool(mean < 0 and p < 0.05)}


def _binomial_vs_half(hits: int, n: int) -> dict:
    """Normal approximation of a two-sided binomial test against a 50% coin flip."""
    if n < 10:
        return {"p_value_vs_coinflip": None}
    z = (hits - 0.5 * n) / math.sqrt(0.25 * n)
    return {"p_value_vs_coinflip": _safe(math.erfc(abs(z) / math.sqrt(2)))}


def grouped_metrics(records: list[EvalRecord], key: str) -> dict[str, dict]:
    groups: dict[str, list[EvalRecord]] = {}
    for r in records:
        groups.setdefault(getattr(r, key), []).append(r)
    return {k: compute_metrics(v) for k, v in sorted(groups.items())}


def objective(metrics: dict, name: str) -> float:
    """Lower is better."""
    if metrics.get("n", 0) == 0:
        return float("inf")
    if name == "mae":
        return metrics["mae"]["value"]
    if name == "rmse":
        return metrics["rmse"]["value"]
    if name == "directional_accuracy":
        return -(metrics["directional_accuracy"]["value"] or 0.0)
    if name == "correlation":
        return -(metrics["correlation"]["value"] or 0.0)
    raise ValueError(f"unknown objective {name!r}")
