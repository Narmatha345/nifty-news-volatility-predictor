"""Out-of-sample model evaluation for the next-session horizon (System 3, phases: feature analysis,
incremental feature sets, model selection, time-series-safe training).

Methodology (nothing here ever shuffles observations):
  * Dataset: one row per (prediction session d, company) built by the backtester with the same
    point-in-time rules as live predictions (news cut-off 09:15 IST on d, closes before d), target
    = next-session return close[d] / close[previous session] - 1.
  * Chronological split: TRAIN < VALIDATION < TEST (explicit dates or 60/20/20 of sessions).
  * For every feature set and every ridge penalty `alpha`, a model is fitted on TRAIN only;
    `alpha` is chosen on VALIDATION; the feature set of the "selected" model is also chosen on
    VALIDATION. The TEST period is scored only after those choices are fixed.
  * Walk-forward (expanding window, monthly): for each month M with >= 2 earlier months, alpha is
    chosen on the month before M using a model trained on the months before that, then the model is
    refitted on all months before M and evaluated on M.
  * Every model is compared with the same transparent baselines on the same rows: no-change,
    pooled historical mean, company historical mean and 20-session rolling mean.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date

import numpy as np

from backend.system2_prediction.features import FEATURE_NAMES
from backend.system2_prediction.linear import LinearModel, fit_logistic, fit_ridge
from backend.system3_backtesting.backtester import BASELINES
from backend.system3_backtesting.metrics import (
    EvalRecord, _binomial_vs_half, _safe, feature_analysis, paired_improvement_test,
)
from backend.system3_backtesting.tuner import chronological_split

ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0)
NEWS = ["relative_sentiment", "sentiment_zscore", "news_count", "news_relevance", "news_recency",
        "macro_relative", "nifty_relative"]
MARKET = ["historical_volatility", "beta", "previous_return", "rolling_return_5d", "rolling_return_20d"]
# Phase 8: start small, add one block at a time.
FEATURE_SETS: dict[str, list[str]] = {
    "intercept only (training mean)": [],
    "+ raw sentiment": ["raw_sentiment"],
    "+ relative sentiment": ["relative_sentiment", "sentiment_zscore"],
    "+ news features": NEWS,
    "recommended initial set": ["relative_sentiment", "news_count", "news_recency", "macro_relative",
                                "historical_volatility", "beta"],
    "market features only": MARKET,
    "news + market features": NEWS + MARKET,
    "full": list(FEATURE_NAMES),
}


def _rows(recs: list[EvalRecord]) -> list[dict]:
    return [dict(r.features) for r in recs]


def _neutral(recs: list[EvalRecord]) -> np.ndarray:
    return np.array([r.neutral_threshold_pct for r in recs])


def score_predictions(pred: np.ndarray, recs: list[EvalRecord]) -> dict:
    """Error / direction metrics of a prediction vector against the records' actual returns."""
    if not recs:
        return {"n": 0}
    act = np.array([r.actual for r in recs])
    neutral = _neutral(recs)
    err = pred - act
    pdir = np.where(pred > neutral, "UP", np.where(pred < -neutral, "DOWN", "NEUTRAL"))
    adir = np.array([r.actual_direction for r in recs])
    signed = (pdir != "NEUTRAL") & (act != 0)
    hits = int(((pred > 0) == (act > 0))[signed].sum())
    corr = float(np.corrcoef(pred, act)[0, 1]) if len(act) > 2 and pred.std() > 0 and act.std() > 0 else None
    return {"n": len(recs), "mae": _safe(float(np.mean(np.abs(err)))),
            "rmse": _safe(float(np.sqrt(np.mean(err ** 2)))), "bias": _safe(float(err.mean())),
            "correlation": _safe(corr), "directional_accuracy": _safe(float(np.mean(pdir == adir))),
            "sign_accuracy": {"value": _safe(hits / signed.sum()) if signed.sum() else None, "n": int(signed.sum()),
                              **_binomial_vs_half(hits, int(signed.sum()))},
            "predicted": {d: int((pdir == d).sum()) for d in ("UP", "DOWN", "NEUTRAL")},
            "actual": {d: int((adir == d).sum()) for d in ("UP", "DOWN", "NEUTRAL")}}


def baseline_scores(recs: list[EvalRecord]) -> dict:
    return {b: score_predictions(np.array([r.baseline(b) for r in recs], dtype=float), recs)
            for b in BASELINES}


def usable(recs: list[EvalRecord]) -> list[EvalRecord]:
    """Rows on which every baseline is defined (so all comparisons use identical samples)."""
    return [r for r in recs if all(r.baseline(b) is not None for b in BASELINES)]


def _fit_select(train: list[EvalRecord], valid: list[EvalRecord], feats: list[str]) -> tuple[LinearModel, dict]:
    """Fit ridge on train for every alpha; choose alpha by validation MAE."""
    y = [r.actual for r in train]
    rows_t, rows_v = _rows(train), _rows(valid)
    tried = {}
    best = None
    for a in (ALPHAS if feats else (0.0,)):
        m = fit_ridge(rows_t, y, feats, a, (train[0].as_of_date, train[-1].as_of_date))
        pv = np.array([m.predict(x) for x in rows_v])
        mae = float(np.mean(np.abs(pv - np.array([r.actual for r in valid]))))
        tried[a] = round(mae, 5)
        if best is None or mae < best[0]:
            best = (mae, m)
    return best[1], tried


def _direction_model(train, valid, test, feats) -> dict:
    """L2 logistic regression for P(return > 0); compared with always predicting the training
    majority sign. Penalty chosen on validation log-loss."""
    def ll(m, recs):
        p = np.clip([m.predict(x) for x in _rows(recs)], 1e-6, 1 - 1e-6)
        y = np.array([r.actual > 0 for r in recs], dtype=float)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    yt = [int(r.actual > 0) for r in train]
    best = None
    for a in (ALPHAS if feats else (0.0,)):
        m = fit_logistic(_rows(train), yt, feats, a)
        v = ll(m, valid)
        if best is None or v < best[0]:
            best = (v, m)
    m = best[1]
    majority_up = np.mean(yt) >= 0.5

    def acc(recs):
        if not recs:
            return {"n": 0}
        y = np.array([r.actual > 0 for r in recs])
        p = np.array([m.predict(x) for x in _rows(recs)])
        hits = int(((p >= 0.5) == y).sum())
        base_hits = int((y == majority_up).sum())
        brier = float(np.mean((p - y) ** 2))
        brier0 = float(np.mean((np.mean(yt) - y) ** 2))
        return {"n": len(recs), "accuracy": _safe(hits / len(recs)), **_binomial_vs_half(hits, len(recs)),
                "majority_class_accuracy": _safe(base_hits / len(recs)),
                "brier": _safe(brier), "brier_base_rate": _safe(brier0),
                "share_predicted_up": _safe(float(np.mean(p >= 0.5)))}
    return {"alpha": m.alpha, "train_up_rate": _safe(float(np.mean(yt))),
            "validation": acc(valid), "test": acc(test)}


def walk_forward(recs: list[EvalRecord], feats: list[str]) -> list[dict]:
    by_month: dict[str, list[EvalRecord]] = defaultdict(list)
    for r in recs:
        by_month[r.month].append(r)
    months = sorted(by_month)
    out = []
    for i, mth in enumerate(months):
        if i < 2:
            continue
        inner_train = [r for m in months[:i - 1] for r in by_month[m]]
        inner_valid = by_month[months[i - 1]]
        _, tried = _fit_select(inner_train, inner_valid, feats)
        alpha = min(tried, key=tried.get)
        train = [r for m in months[:i] for r in by_month[m]]
        model = fit_ridge(_rows(train), [r.actual for r in train], feats, alpha)
        test = by_month[mth]
        pred = np.array([model.predict(x) for x in _rows(test)])
        out.append({"month": mth, "train_months": months[:i], "alpha": alpha,
                    "model": score_predictions(pred, test), "baselines": baseline_scores(test)})
    return out


def run_experiment(recs: list[EvalRecord], train_end: date | None = None, valid_end: date | None = None,
                   reference_models: dict[str, list[EvalRecord]] | None = None) -> dict:
    """`recs`: next-session records (any params - only features, actuals and baselines are used).
    `reference_models`: name -> records of fixed-formula models (e.g. factor-v1 / factor-v2) on the
    same dates, scored on the same splits for the before/after comparison."""
    recs = sorted(usable([r for r in recs if r.horizon_type == "today"]), key=lambda r: (r.as_of_date, r.ticker))
    dates = sorted({date.fromisoformat(r.as_of_date) for r in recs})
    tr_d, va_d, te_d, how = chronological_split(dates, train_end, valid_end)
    tr_s, va_s, te_s = ({d.isoformat() for d in x} for x in (tr_d, va_d, te_d))
    split = {"train": [r for r in recs if r.as_of_date in tr_s],
             "validation": [r for r in recs if r.as_of_date in va_s],
             "test": [r for r in recs if r.as_of_date in te_s]}
    if not split["train"] or not split["validation"]:
        raise ValueError("not enough dated rows for a train/validation split")

    def span(rs):
        return {"start": rs[0].as_of_date, "end": rs[-1].as_of_date, "rows": len(rs),
                "sessions": len({r.as_of_date for r in rs})} if rs else {"rows": 0}

    models = {}
    for name, feats in FEATURE_SETS.items():
        m, tried = _fit_select(split["train"], split["validation"], feats)
        res = {"features": feats, "alpha": m.alpha, "validation_mae_by_alpha": tried,
               "coefficients": dict(zip(m.features, m.coef)), "intercept": m.intercept, "model": m.to_dict()}
        for part in ("train", "validation", "test"):
            rs = split[part]
            res[part] = score_predictions(np.array([m.predict(x) for x in _rows(rs)]), rs) if rs else {"n": 0}
        res["direction_model"] = _direction_model(split["train"], split["validation"], split["test"], feats)
        models[name] = res

    selected = min((k for k in models if k != "intercept only (training mean)"),
                   key=lambda k: models[k]["validation"]["mae"])
    refs = {}
    for name, rrecs in (reference_models or {}).items():
        by_key = {(r.as_of_date, r.ticker): r for r in rrecs if r.horizon_type == "today"}
        refs[name] = {}
        for part, rs in split.items():
            matched = [(by_key[(r.as_of_date, r.ticker)], r) for r in rs if (r.as_of_date, r.ticker) in by_key]
            refs[name][part] = score_predictions(np.array([m.predicted for m, _ in matched]),
                                                 [r for _, r in matched]) if matched else {"n": 0}

    sel = models[selected]
    sel_model = LinearModel.from_dict(sel["model"])
    beats = {}
    for part in ("validation", "test"):
        rs = split[part]
        if not rs:
            continue
        pred = [sel_model.predict(x) for x in _rows(rs)]
        beats[part] = {b: {"mae_lower": sel[part]["mae"] < baseline_scores(rs)[b]["mae"],
                           **paired_improvement_test(pred, [r.baseline(b) for r in rs], [r.actual for r in rs],
                                                     [r.as_of_date for r in rs])}
                       for b in BASELINES}
    return {
        "split_method": how,
        "periods": {k: span(v) for k, v in split.items()},
        "baselines": {part: baseline_scores(rs) for part, rs in split.items() if rs},
        "feature_analysis": {"train": feature_analysis(split["train"], FEATURE_NAMES),
                             "all": feature_analysis(recs, FEATURE_NAMES)},
        "models": models,
        "selected_feature_set": selected,
        "selection_rule": "lowest VALIDATION MAE among non-trivial feature sets (test not used)",
        "selected_beats_baselines": beats,
        "recommendation_rule": "recommend only if the selected model's MAE improvement over EVERY baseline is "
                               "statistically significant (paired, date-clustered one-sided t test, p < 0.05) on "
                               "BOTH validation and test",
        "reference_models": refs,
        "walk_forward": {"full": walk_forward(recs, list(FEATURE_NAMES)),
                         "selected": walk_forward(recs, FEATURE_SETS[selected])},
    }
