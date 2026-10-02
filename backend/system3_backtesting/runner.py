"""DB orchestration for System 3: load point-in-time data, backtest, tune, persist, report."""
from __future__ import annotations

import copy
import json
import logging
import uuid
from datetime import date, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.config.settings import get_settings
from backend.data.market_data import load_closes
from backend.database.models import (
    ActualResult, BacktestRun, Company, NewsAnalysis, NewsArticle, ParameterChange, Prediction, PredictionRun,
)
from backend.services import parameters as params_svc
from backend.services.timeutil import market_close_utc, market_open_utc, utcnow
from backend.services.universe import active_companies, index_symbol, universe_periods
from backend.system1_news import availability
from backend.system1_news.analyzer import load_items
from backend.system1_news.pipeline import CompanyInfo
from backend.system1_news.sentiment import get_engine
from backend.system2_prediction.model import model_version
from backend.system3_backtesting.backtester import BacktestData, evaluation_dates, run_backtest
from backend.system3_backtesting.metrics import EvalRecord, compute_metrics
from backend.system3_backtesting.reports import build_reports
from backend.system3_backtesting.tuner import tune

log = logging.getLogger(__name__)
PRICE_WARMUP_DAYS = 400


def load_backtest_data(session: Session, start: date, end: date, tickers: list[str] | None,
                       lookback_days: int, engine_name: str | None = None) -> BacktestData:
    # Date-aware universe: each period keeps the membership and weights effective THEN.
    catalog = {c.ticker: c for c in session.scalars(select(Company))}
    periods = []
    for lo, hi, snap in universe_periods(session, start, end):
        comps = [CompanyInfo(m.ticker, catalog[m.ticker].name, catalog[m.ticker].sector, m.weight or 0.0)
                 for m in snap.members if m.ticker in catalog and (not tickers or m.ticker in tickers)]
        periods.append((lo, hi, snap.version, comps))
    infos = periods[0][3] if periods else []
    all_tickers = {c.ticker for p in periods for c in p[3]}
    p_start = start - timedelta(days=PRICE_WARMUP_DAYS)
    p_end = end + timedelta(days=200)  # so month/quarter targets after `end` can still be scored
    closes = {t: load_closes(session, catalog[t].yahoo_symbol, p_start, p_end) for t in all_tickers}
    idx = load_closes(session, index_symbol(), p_start, p_end)
    engine_name = engine_name or get_engine(get_settings().sentiment_engine).name
    items = load_items(session, engine_name, until=market_open_utc(end) + timedelta(days=200),
                       since=market_open_utc(start) - timedelta(days=lookback_days))
    return BacktestData(items, infos, closes, idx if not idx.empty else None, universe=periods)


def execute(session: Session, run_id: str) -> BacktestRun:
    run = session.get(BacktestRun, run_id)
    run.status = "running"
    session.commit()
    try:
        _execute(session, run)
        run.status, run.finished_at = "completed", utcnow()
    except Exception as exc:
        log.exception("backtest %s failed", run_id)
        session.rollback()
        run = session.get(BacktestRun, run_id)
        run.status, run.error, run.finished_at = "failed", str(exc), utcnow()
    session.commit()
    return run


def create_run(session: Session, start: date, end: date, *, tickers=None,
               horizon_types=("today", "month_end", "quarter_end"), step: int = 1, tune_params: bool = False,
               n_trials: int = 30, objective: str = "mae", param_version: str | None = None,
               availability_mode: str = "provider_timestamp", train_end: date | None = None,
               valid_end: date | None = None) -> BacktestRun:
    version = param_version or params_svc.get_active(session)[0]
    if params_svc.get_set(session, version) is None:
        raise KeyError(f"unknown parameter version {version}")
    run = BacktestRun(id=f"bt-{uuid.uuid4().hex[:10]}", status="pending", kind="tune" if tune_params else "backtest",
                      created_at=utcnow(), start_date=start, end_date=end, param_version=version,
                      config={"tickers": tickers, "horizon_types": list(horizon_types), "step": step,
                              "tune": tune_params, "n_trials": n_trials, "objective": objective,
                              "availability_mode": availability_mode,
                              "train_end": str(train_end) if train_end else None,
                              "valid_end": str(valid_end) if valid_end else None})
    session.add(run)
    session.commit()
    return run


def _execute(session: Session, run: BacktestRun) -> None:
    cfg = run.config
    params = params_svc.get_params(session, run.param_version)
    mode = cfg.get("availability_mode") or "provider_timestamp"
    data = load_backtest_data(session, run.start_date, run.end_date, cfg.get("tickers"),
                              max(params["system1"]["max_lookback_days"], params["system1"]["baseline_days"] + 7))
    if not data.calendar.sessions:
        raise RuntimeError("no market data stored for the requested tickers - run market-data refresh first")
    dates = evaluation_dates(data, run.start_date, run.end_date, cfg.get("step", 1))
    convention = get_settings().quarter_convention
    records = run_backtest(data, dates, params, tuple(cfg["horizon_types"]), convention, mode)

    tuning, changes = None, []
    if cfg.get("tune"):
        d = (lambda v: date.fromisoformat(v) if v else None)
        tuning = tune(data, dates, params, params_svc.search_space(), cfg.get("n_trials", 30),
                      cfg.get("objective", "mae"), convention=convention, train_end=d(cfg.get("train_end")),
                      valid_end=d(cfg.get("valid_end")), availability_mode=mode)
        if tuning.recommended:
            summary = {"objective": tuning.objective_name,
                       "validation_before": tuning.baseline_validation.get("mae"),
                       "validation_after": tuning.best_validation.get("mae"),
                       "validation_directional_before": tuning.baseline_validation.get("directional_accuracy"),
                       "validation_directional_after": tuning.best_validation.get("directional_accuracy"),
                       "test_before": tuning.baseline_test.get("mae"),
                       "test_after": tuning.best_test.get("mae")}
            cand = params_svc.create_candidate(session, tuning.best_params, run.param_version, tuning.reason,
                                               run.id, summary)
            changes = [{"to_version": cand.version, "parameter_name": c.parameter_name,
                        "old_value": c.old_value, "new_value": c.new_value, "reason": c.reason}
                       for c in session.scalars(select(ParameterChange)
                                                .where(ParameterChange.to_version == cand.version))]

    _persist_records(session, run, records, params["system2"])
    used_ids = {i.news_id for i in data.items}
    arts = session.execute(select(NewsArticle.id, NewsArticle.title, NewsArticle.backfilled,
                                  NewsArticle.duplicate_of).where(NewsArticle.id.in_(used_ids))).all() \
        if used_ids else []
    meta = {"run_id": run.id, "generated_at": utcnow().isoformat() + "Z", "start": str(run.start_date),
            "end": str(run.end_date), "param_version": run.param_version,
            "model_version": model_version(run.param_version, params["system2"]),
            "tickers": [c.ticker for c in data.companies],
            "horizon_types": cfg["horizon_types"], "n_news": len(arts),
            "n_clusters": len({a.duplicate_of or a.id for a in arts}),
            "backfilled_share": (sum(a.backfilled for a in arts) / len(arts)) if arts else 0.0,
            "availability_mode": mode,
            "data_quality": data_quality(session, data, run.start_date, run.end_date)}
    report, md = build_reports(meta, records, {a.id: a.title for a in arts}, tuning, changes)
    run.metrics = report["metrics"]
    run.report_json = json.loads(json.dumps(report, default=str))
    run.report_md = md
    out_dir = get_settings().reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run.id}.md").write_text(md, encoding="utf-8")
    (out_dir / f"{run.id}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def _persist_records(session: Session, run: BacktestRun, records: list[EvalRecord], p2: dict) -> None:
    """Store every evaluated backtest prediction + its actual outcome (traceable in the DB)."""
    by_day: dict[str, list[EvalRecord]] = {}
    for r in records:
        by_day.setdefault(r.as_of_date, []).append(r)
    for day, recs in by_day.items():
        as_of = market_open_utc(date.fromisoformat(day))
        prun = PredictionRun(id=f"p-{uuid.uuid4().hex[:12]}", created_at=utcnow(), as_of=as_of,
                             param_version=run.param_version, model_version=model_version(run.param_version, p2),
                             sentiment_run_id="backtest-replay", is_backtest=True, backtest_run_id=run.id)
        session.add(prun)
        for r in recs:
            p = Prediction(run_id=prun.id, ticker=r.ticker, company=r.ticker, prediction_timestamp=as_of,
                           horizon_type=r.horizon_type, horizon_label=r.horizon_label,
                           target_date=date.fromisoformat(r.target_date), trading_days_ahead=r.trading_days_ahead,
                           previous_close=r.previous_close, sentiment_score=r.sentiment_score,
                           predicted_movement=r.predicted, uncertainty_1sigma_pct=r.sigma_pct,
                           predicted_price=round(r.previous_close * (1 + r.predicted / 100), 2),
                           predicted_direction=r.predicted_direction, status="OK",
                           features={"model_inputs": dict(r.features), "baselines": dict(r.baselines),
                                     "news_flow_status": r.news_flow_status, "new_news_count": r.new_news_count},
                           contributions=dict(r.contributions), evidence_news_ids=list(r.evidence_news_ids),
                           model_version=prun.model_version, param_version=run.param_version)
            p.actual = ActualResult(actual_close_date=date.fromisoformat(r.target_date),
                                    actual_close=r.actual_close, actual_movement=r.actual, error=round(r.error, 4),
                                    direction_correct=r.predicted_direction == r.actual_direction,
                                    evaluated_at=utcnow())
            session.add(p)
    session.flush()


def score_live_predictions(session: Session, now: datetime | None = None) -> dict:
    """Attach actual outcomes to matured LIVE predictions. Writes `actual_results` rows only; the
    prediction rows themselves are never modified.

    A prediction matures after its target session's 15:30 IST close. The target is always a session
    from the exchange calendar (never a weekend/holiday), so a holiday is never scored as a zero-return
    day. If the target close is missing from the price data although LATER sessions are present, the
    prediction is reported as `missing_target_price` (data gap or unscheduled closure) and left
    unscored instead of being guessed."""
    now = now or utcnow()
    pending = session.scalars(select(Prediction).join(PredictionRun)
                              .where(PredictionRun.is_backtest.is_(False))
                              .where(Prediction.status == "OK")
                              .where(~Prediction.id.in_(select(ActualResult.prediction_id)))).all()
    symbols = {c.ticker: c.yahoo_symbol for c in session.scalars(select(Company))}
    out = {"pending": len(pending), "scored": 0, "not_matured": 0, "awaiting_price": 0,
           "missing_target_price": [], "scored_ids": []}
    for p in pending:
        if now < market_close_utc(p.target_date):
            out["not_matured"] += 1
            continue
        sym = symbols.get(p.ticker)
        s = load_closes(session, sym, p.target_date, p.target_date) if sym else None
        if s is None or s.empty or p.previous_close is None:
            later = load_closes(session, sym, p.target_date + timedelta(days=1)) if sym else None
            if later is not None and not later.empty:
                out["missing_target_price"].append({"prediction_id": p.id, "ticker": p.ticker,
                                                    "target_date": str(p.target_date)})
            else:
                out["awaiting_price"] += 1
            continue
        actual_close = float(s.iloc[0])
        actual = (actual_close / p.previous_close - 1) * 100
        neutral = p.uncertainty_1sigma_pct * params_svc.get_params(session, p.param_version)["system2"][
            "neutral_threshold_sigma"]
        adir = "UP" if actual > neutral else "DOWN" if actual < -neutral else "NEUTRAL"
        err = p.predicted_movement - actual
        session.add(ActualResult(prediction_id=p.id, actual_close_date=p.target_date, actual_close=actual_close,
                                 actual_movement=round(actual, 4), error=round(err, 4),
                                 direction_correct=adir == p.predicted_direction, actual_direction=adir,
                                 abs_error=round(abs(err), 4), squared_error=round(err * err, 6),
                                 previous_close_date=p.previous_close_date, evaluated_at=utcnow()))
        out["scored"] += 1
        out["scored_ids"].append(p.id)
    session.flush()
    return out


def after_close_run(session: Session, now: datetime | None = None) -> dict:
    """After 15:30 IST: refresh prices, then attach outcomes to matured live predictions."""
    from backend.services.orchestrator import refresh_market_data
    prices = {k: v.get("status") for k, v in refresh_market_data(session).items()}
    session.commit()
    res = score_live_predictions(session, now)
    return {"market_data": prices, **res, "live_metrics": live_metrics(session)}


def live_metrics(session: Session) -> dict:
    rows = session.execute(select(Prediction, ActualResult).join(ActualResult).join(PredictionRun)
                           .where(PredictionRun.is_backtest.is_(False))).all()
    neutral_sigma: dict[str, float] = {}
    recs = []
    for p, a in rows:
        if p.param_version not in neutral_sigma:
            ps = params_svc.get_set(session, p.param_version)
            neutral_sigma[p.param_version] = ps.params["system2"]["neutral_threshold_sigma"] if ps else 0.1
        recs.append(EvalRecord(str(p.prediction_timestamp.date()), p.ticker, p.horizon_type, p.horizon_label,
                               str(p.target_date), p.trading_days_ahead, p.previous_close, a.actual_close,
                               p.predicted_movement, a.actual_movement, p.uncertainty_1sigma_pct,
                               p.predicted_direction, p.sentiment_score or 0.0,
                               int((p.features or {}).get("company_news_count", 0)),
                               neutral_sigma[p.param_version] * p.uncertainty_1sigma_pct))
    return compute_metrics(recs)


def data_quality(session: Session, data: BacktestData, start: date, end: date) -> dict:
    """News availability + price coverage for the backtest period (articles available up to the last
    prediction cut-off, i.e. published before 09:15 IST on `end`)."""
    lo, hi = market_open_utc(start) - timedelta(days=1), market_open_utc(end)
    rows = session.execute(select(NewsArticle.availability_status, NewsArticle.timestamp_precision,
                                  NewsArticle.duplicate_of, NewsArticle.fetched_at)
                           .where(NewsArticle.published_at.between(lo, hi))).all()
    n = len(rows)
    status = {k: sum((r[0] or "unknown") == k for r in rows) for k in availability.STATUSES}
    in_period = [i for i in data.items if lo <= i.published_at <= hi]
    sessions = [d for d in data.calendar.sessions if start <= d <= end]
    missing = {t: sum(d not in s.index for d in sessions) for t, s in data.closes.items()}
    idx_missing = None if data.index_closes is None else sum(d not in data.index_closes.index for d in sessions)
    return {
        "news_window": {"from": lo.isoformat() + "Z", "to": hi.isoformat() + "Z"},
        "total_news_articles": n,
        "availability_status": status,
        "verified_pre_market": status["verified_pre_market"],
        "post_event_articles": status["collected_after_event"],
        "unknown_availability": status["unknown"],
        "coarse_timestamps": sum(r[1] == "coarse" for r in rows),
        "duplicate_rate": round(sum(r[2] is not None for r in rows) / n, 4) if n else None,
        "first_fetched_at": min((r[3] for r in rows), default=None),
        "relevant_analyses_in_period": len(in_period),
        "price_report_analyses": sum(i.price_report for i in in_period),
        "price_sessions": len(sessions),
        "missing_price_sessions_by_ticker": missing,
        "missing_index_sessions": idx_missing,
        "price_coverage": round(1 - sum(missing.values()) / (len(sessions) * len(missing)), 4)
        if sessions and missing else None,
    }


def run_experiment_db(session: Session, start: date, end: date, train_end: date | None = None,
                      valid_end: date | None = None, param_version: str | None = None,
                      previous_version: str = "default-v2", previous_engine: str | None = "lexicon-v2",
                      availability_mode: str = "provider_timestamp") -> dict:
    """Before/after + out-of-sample model study for the next-session horizon. Writes
    reports/experiment-<id>.md/.json. Proposes (never activates) a linear-model candidate."""
    from backend.system3_backtesting.experiments import run_experiment
    from backend.system3_backtesting.reports import experiment_markdown

    version = param_version or params_svc.DEFAULT_VERSION
    params = params_svc.get_params(session, version)
    if params is None:
        raise KeyError(f"unknown parameter version {version}")
    convention = get_settings().quarter_convention
    lookback = max(params["system1"]["max_lookback_days"], params["system1"]["baseline_days"] + 7)
    data = load_backtest_data(session, start, end, None, lookback)
    if not data.calendar.sessions:
        raise RuntimeError("no market data stored - run market-data refresh first")
    dates = evaluation_dates(data, start, end)
    revised = run_backtest(data, dates, params, ("today",), convention, availability_mode)
    refs = {f"factor-v2 relative sentiment ({version})": revised}
    prev_params = params_svc.get_params(session, previous_version)
    if prev_params is not None:
        engine_has_rows = previous_engine and session.scalar(
            select(NewsAnalysis.id).where(NewsAnalysis.engine == previous_engine).limit(1)) is not None
        prev_data = load_backtest_data(session, start, end, None, prev_params["system1"]["max_lookback_days"],
                                       engine_name=previous_engine if engine_has_rows else None)
        refs[f"previous model factor-v1 ({previous_version}, "
             f"{previous_engine if engine_has_rows else 'current engine'})"] = run_backtest(
            prev_data, dates, prev_params, ("today",), convention, availability_mode)
    verified = run_backtest(data, dates, params, ("today",), convention, "verified")
    result = run_experiment(revised, train_end, valid_end, refs)
    exp_id = f"exp-{uuid.uuid4().hex[:8]}"
    sel = result["models"][result["selected_feature_set"]]
    test_beats = result["selected_beats_baselines"].get("test", {})
    valid_beats = result["selected_beats_baselines"].get("validation", {})
    recommend = bool(test_beats) and bool(valid_beats) and \
        all(v["significant"] for v in test_beats.values()) and all(v["significant"] for v in valid_beats.values())
    candidate = None
    if recommend:
        cand_params = copy.deepcopy(params)
        cand_params["system2"]["model"] = "linear"
        cand_params["system2"]["linear"] = sel["model"]
        ps = params_svc.create_candidate(session, cand_params, version,
                                         f"{exp_id}: ridge on '{result['selected_feature_set']}' significantly "
                                         f"beat every baseline on validation and test", None,
                                         {"validation": sel["validation"], "test": sel["test"]})
        candidate = ps.version
    result.update({
        "experiment_id": exp_id, "generated_at": utcnow().isoformat() + "Z",
        "period": {"start": str(start), "end": end.isoformat()}, "param_version": version,
        "availability_mode": availability_mode,
        "verified_news_only": {
            "rows": len(verified),
            "rows_with_any_new_company_news": sum(r.new_news_count > 0 for r in verified),
            "note": "news this system held before each 09:15 IST cut-off (fetched_at <= cut-off)"},
        "data_quality": data_quality(session, data, start, end),
        "candidate_created": candidate,
        "recommendation": "candidate stored (NOT activated)" if candidate else
        "no candidate: the selected model does not SIGNIFICANTLY beat every baseline on both validation and test",
        "diagnostics": {name: compute_metrics(r) for name, r in refs.items()},
        "diagnostics_by_company": {name: {t: compute_metrics([x for x in r if x.ticker == t])
                                          for t in sorted({x.ticker for x in r})} for name, r in refs.items()},
    })
    out_dir = get_settings().reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    md = experiment_markdown(result)
    (out_dir / f"{exp_id}.md").write_text(md, encoding="utf-8")
    (out_dir / f"{exp_id}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    result["report_markdown"] = md
    return result
