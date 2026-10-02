"""Command-line entry points.

  python -m backend.cli init                      seed universe + default parameters
  python -m backend.cli market-data [--start D]   download daily prices (yfinance) into the DB
  python -m backend.cli collect                   live news collection + analysis
  python -m backend.cli backfill --start D --end D   historical news (providers that support it)
  python -m backend.cli sentiment                 System 1 scores (all horizons) -> JSON
  python -m backend.cli predict [--validate]      System 2 predictions (+ LLM validation)
  python -m backend.cli backtest --start D --end D [--tune --trials N] [--train-end D --valid-end D]
                                                  [--availability-mode provider_timestamp|verified]   System 3
  python -m backend.cli experiment --start D --end D [--train-end D --valid-end D]
                                                  before/after + out-of-sample model study (next session)
  python -m backend.cli seed-universe [--update]  write universe.yaml (weights, aliases) into the DB
  python -m backend.cli analyze                   (re)analyze stored articles with the current engine
  python -m backend.cli audit-news [--start D --end D] [--recompute]
                                                  availability audit: overall / provider / company / trading date
  python -m backend.cli universe [--on D] [--confirm VERSION]
                                                  active + historical Top-10 membership; explicit confirmation
  python -m backend.cli outcomes                  after 15:30 IST: refresh prices, store actual outcomes
  python -m backend.cli validate [--run-id R] [--tickers T ...]
                                                  LLM review of an existing run (never changes predictions)
  python -m backend.cli daily [--validate]        collect -> audit -> score -> prices -> predict (schedule
                                                  before 09:15 IST; after the open it targets the NEXT session)
  python -m backend.cli score-live                attach actual outcomes to matured live predictions
  python -m backend.cli serve                     start API + dashboard on http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, time, timedelta, timezone

from backend.database.db import init_engine, session_scope
from backend.services import orchestrator


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _d(v: str | None) -> date | None:
    return date.fromisoformat(v) if v else None


def _day_start(d: str) -> datetime:
    return datetime.combine(date.fromisoformat(d), time.min)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="backend.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    m = sub.add_parser("market-data")
    m.add_argument("--start")
    m.add_argument("--end")
    sub.add_parser("collect")
    b = sub.add_parser("backfill")
    b.add_argument("--start", required=True)
    b.add_argument("--end", required=True)
    b.add_argument("--chunk-days", type=int, default=7, help="query window size (providers cap results per call)")
    b.add_argument("--companies", nargs="*", help="only these tickers' company queries (e.g. new universe members)")
    sub.add_parser("sentiment")
    p = sub.add_parser("predict")
    p.add_argument("--validate", action="store_true")
    t = sub.add_parser("backtest")
    t.add_argument("--start", required=True)
    t.add_argument("--end", required=True)
    t.add_argument("--tickers", nargs="*")
    t.add_argument("--horizons", nargs="*", default=["today", "month_end", "quarter_end"])
    t.add_argument("--step", type=int, default=1)
    t.add_argument("--tune", action="store_true")
    t.add_argument("--trials", type=int, default=30)
    t.add_argument("--objective", default="mae", choices=["mae", "rmse", "directional_accuracy", "correlation"])
    t.add_argument("--availability-mode", default="provider_timestamp", choices=["provider_timestamp", "exact_timestamp", "verified"])
    t.add_argument("--train-end", help="last date of the tuning TRAIN period (with --valid-end)")
    t.add_argument("--valid-end", help="last date of the VALIDATION period; later dates are the untouched test")
    t.add_argument("--param-version")
    e = sub.add_parser("experiment")
    e.add_argument("--start", required=True)
    e.add_argument("--end", required=True)
    e.add_argument("--train-end")
    e.add_argument("--valid-end")
    e.add_argument("--param-version", help="revised parameter set (default: the current default version)")
    e.add_argument("--previous-version", default="default-v2")
    e.add_argument("--previous-engine", default="lexicon-v2")
    e.add_argument("--availability-mode", default="provider_timestamp", choices=["provider_timestamp", "exact_timestamp", "verified"])
    sub.add_parser("analyze")
    su = sub.add_parser("seed-universe")
    su.add_argument("--update", action="store_true", help="overwrite DB rows from universe.yaml")
    a = sub.add_parser("audit-news")
    a.add_argument("--recompute", action="store_true")
    a.add_argument("--start")
    a.add_argument("--end")
    un = sub.add_parser("universe")
    un.add_argument("--on", help="date to resolve the universe for (default today)")
    un.add_argument("--confirm", help="mark a universe version as confirmed (explicit, irreversible)")
    sub.add_parser("outcomes")
    va = sub.add_parser("validate")
    va.add_argument("--run-id", help="prediction run to review (default: latest live run)")
    va.add_argument("--tickers", nargs="*")
    dl = sub.add_parser("daily")
    dl.add_argument("--validate", action="store_true")
    sub.add_parser("score-live")
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    # Windows consoles default to cp1252; reports contain e.g. "₹". Printing must never abort a run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "serve":
        import uvicorn
        uvicorn.run("backend.api.app:app", host=args.host, port=args.port)
        return 0

    init_engine()
    with session_scope() as db:
        orchestrator.bootstrap(db)
        if args.cmd == "init":
            print("universe and default parameters ready")
        elif args.cmd == "market-data":
            _print(orchestrator.refresh_market_data(
                db, date.fromisoformat(args.start) if args.start else None,
                date.fromisoformat(args.end) if args.end else None))
        elif args.cmd == "collect":
            _print(orchestrator.collect_and_analyze(db))
        elif args.cmd == "backfill":
            cur, end = _day_start(args.start), _day_start(args.end) + timedelta(days=1)
            queries = None
            if args.companies:
                from sqlalchemy import select
                from backend.database.models import Company
                rows = db.scalars(select(Company).where(Company.ticker.in_(args.companies))).all()
                queries = [f'"{c.aliases[0] if c.aliases else c.name}"' for c in rows]
            while cur < end:
                nxt = min(end, cur + timedelta(days=args.chunk_days))
                res = orchestrator.collect_and_analyze(db, cur, nxt - timedelta(seconds=1), queries)
                print(f"{cur.date()}..{nxt.date()}: new={res['new']} fetched={res['fetched']} "
                      f"errors={len(res['errors'])}", flush=True)
                cur = nxt
        elif args.cmd == "sentiment":
            out, *_ = orchestrator.sentiment(db)
            _print(out)
        elif args.cmd == "predict":
            out = orchestrator.predict(db)
            if args.validate:
                out["llm_validation"] = orchestrator.validate_safely(db, out["run_id"])
            _print(out)
        elif args.cmd == "backtest":
            from backend.system3_backtesting import runner
            run = runner.create_run(db, date.fromisoformat(args.start), date.fromisoformat(args.end),
                                    tickers=args.tickers, horizon_types=tuple(args.horizons), step=args.step,
                                    tune_params=args.tune, n_trials=args.trials, objective=args.objective,
                                    param_version=args.param_version, availability_mode=args.availability_mode,
                                    train_end=_d(args.train_end), valid_end=_d(args.valid_end))
            run = runner.execute(db, run.id)
            if run.status != "completed":
                print(f"backtest failed: {run.error}", file=sys.stderr)
                return 1
            print(run.report_md)
            print(f"\nReports written to reports/{run.id}.md and reports/{run.id}.json")
        elif args.cmd == "experiment":
            from backend.system3_backtesting import runner
            res = runner.run_experiment_db(db, date.fromisoformat(args.start), date.fromisoformat(args.end),
                                           _d(args.train_end), _d(args.valid_end), args.param_version,
                                           args.previous_version, args.previous_engine, args.availability_mode)
            print(res["report_markdown"])
            print(f"\nReports written to reports/{res['experiment_id']}.md and .json")
        elif args.cmd == "seed-universe":
            from backend.services.universe import seed_universe
            print(f"companies written: {seed_universe(db, update=args.update)}")
        elif args.cmd == "analyze":
            print(f"analyzed {orchestrator.analyze_news(db)} articles")
        elif args.cmd == "audit-news":
            from backend.system1_news import availability
            from backend.system3_backtesting.reports import write_audit_report
            if args.recompute:
                availability.audit(db, recompute=True)
            rep = availability.audit_report(db, _d(args.start), _d(args.end))
            md_path = write_audit_report(rep)
            print(md_path.read_text(encoding="utf-8"))
            print(f"\nAudit written to {md_path} (+ .json)")
        elif args.cmd == "universe":
            from backend.services import universe as uni
            if args.confirm:
                n = uni.confirm_version(db, args.confirm)
                print(f"confirmed {args.confirm} ({n} members). It now applies from its effective_from date.")
            _print(uni.universe_report(db, _d(args.on)))
        elif args.cmd == "validate":
            from sqlalchemy import select
            from backend.database.models import PredictionRun
            from backend.services.llm import provider_status
            run_id = args.run_id or db.scalar(select(PredictionRun.id).where(PredictionRun.is_backtest.is_(False))
                                              .order_by(PredictionRun.created_at.desc()).limit(1))
            if run_id is None:
                print("no live prediction run to validate", file=sys.stderr)
                return 1
            _print({"run_id": run_id, **provider_status(),
                    "results": orchestrator.validate_safely(db, run_id, tickers=args.tickers)})
        elif args.cmd == "outcomes":
            from backend.system3_backtesting import runner
            _print(runner.after_close_run(db))
        elif args.cmd == "daily":
            _print(orchestrator.daily_run(db, validate=args.validate))
        elif args.cmd == "score-live":
            from backend.system3_backtesting import runner
            _print({**runner.score_live_predictions(db), "live_metrics": runner.live_metrics(db)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
