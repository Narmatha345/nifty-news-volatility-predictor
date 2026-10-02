"""System 3 endpoints: /backtest, /backtest/results."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.api.schemas import parse_date
from backend.database.db import get_db, session_scope
from backend.database.models import BacktestRun
from backend.services.timeutil import iso_z
from backend.system3_backtesting import runner

router = APIRouter()


class BacktestRequest(BaseModel):
    start: str
    end: str
    tickers: list[str] | None = None
    horizon_types: list[str] = Field(default_factory=lambda: ["today", "month_end", "quarter_end"])
    step: int = 1
    tune: bool = False
    n_trials: int = 30
    objective: str = "mae"
    param_version: str | None = None
    availability_mode: str = "provider_timestamp"   # or "verified": only news held before each cut-off
    train_end: str | None = None                    # tuning split: train <= train_end < validation <= valid_end
    valid_end: str | None = None                    # < untouched test (default: 60/20/20 of sessions)


def _bg(run_id: str) -> None:
    with session_scope() as s:
        runner.execute(s, run_id)


@router.post("/backtest")
def start_backtest(req: BacktestRequest, bg: BackgroundTasks, db: Session = Depends(get_db)):
    bad = set(req.horizon_types) - {"today", "month_end", "quarter_end"}
    if bad:
        raise HTTPException(422, f"unknown horizon types {sorted(bad)}")
    if req.availability_mode not in ("provider_timestamp", "exact_timestamp", "verified"):
        raise HTTPException(422, "availability_mode must be provider_timestamp, exact_timestamp or verified")
    try:
        run = runner.create_run(db, parse_date(req.start), parse_date(req.end), tickers=req.tickers,
                                horizon_types=tuple(req.horizon_types), step=req.step, tune_params=req.tune,
                                n_trials=req.n_trials, objective=req.objective, param_version=req.param_version,
                                availability_mode=req.availability_mode,
                                train_end=parse_date(req.train_end) if req.train_end else None,
                                valid_end=parse_date(req.valid_end) if req.valid_end else None)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    bg.add_task(_bg, run.id)
    return {"run_id": run.id, "status": run.status, "poll": f"/backtest/results/{run.id}"}


@router.get("/backtest/results")
def list_results(limit: int = 20, db: Session = Depends(get_db)):
    rows = db.scalars(select(BacktestRun).order_by(BacktestRun.created_at.desc()).limit(limit))
    return [{"run_id": r.id, "kind": r.kind, "status": r.status, "created_at": iso_z(r.created_at),
             "finished_at": iso_z(r.finished_at), "start": str(r.start_date), "end": str(r.end_date),
             "param_version": r.param_version, "n": (r.metrics or {}).get("n"), "error": r.error} for r in rows]


@router.get("/backtest/results/{run_id}")
def get_result(run_id: str, db: Session = Depends(get_db)):
    r = db.get(BacktestRun, run_id)
    if r is None:
        raise HTTPException(404, "unknown run")
    return {"run_id": r.id, "status": r.status, "error": r.error, "config": r.config, "report": r.report_json}


@router.get("/backtest/results/{run_id}/report.md", response_class=PlainTextResponse)
def get_report_md(run_id: str, db: Session = Depends(get_db)):
    r = db.get(BacktestRun, run_id)
    if r is None or r.report_md is None:
        raise HTTPException(404, "report not available")
    return r.report_md


@router.post("/backtest/score-live")
def score_live(db: Session = Depends(get_db)):
    """Attach actual outcomes to matured live predictions and return live track-record metrics."""
    res = runner.score_live_predictions(db)
    return {**res, "live_metrics": runner.live_metrics(db)}
