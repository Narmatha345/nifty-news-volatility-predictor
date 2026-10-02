"""Application-level read endpoints for the web app: overview status, scheduled jobs and the reports
library. Read-only; nothing here changes data. No secret is ever returned (see services.llm)."""
from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.config.settings import PROJECT_ROOT, get_settings
from backend.database.db import get_db
from backend.database.models import (
    ActualResult, BacktestRun, NewsArticle, Prediction, PredictionRun,
)
from backend.services import parameters as params_svc
from backend.services.llm import provider_status, redact
from backend.services.timeutil import iso_z, utcnow
from backend.services.universe import universe_on
from backend.system1_news.providers import enabled_providers

router = APIRouter()
TASKS = ("NiftyNews-PreMarket", "NiftyNews-AfterClose")
LOGS = {"NiftyNews-PreMarket": "daily.log", "NiftyNews-AfterClose": "outcomes.log"}
_REPORT_NAME = re.compile(r"^(bt|exp|audit)-[A-Za-z0-9\-]+\.(md|json)$")


def scheduled_jobs() -> list[dict]:
    """Windows Task Scheduler status of the two pipeline jobs (empty off Windows)."""
    out = []
    for name in TASKS:
        job = {"task": name, "registered": False}
        if os.name == "nt":
            try:
                res = subprocess.run(["schtasks", "/Query", "/TN", name, "/V", "/FO", "CSV"], capture_output=True,
                                     text=True, timeout=10)
                if res.returncode == 0:
                    row = next(csv.DictReader(io.StringIO(res.stdout)))
                    last_run = row.get("Last Run Time") or ""
                    never = last_run.startswith(("30-11-1999", "11/30/1999", "1999")) or row.get("Last Result") == "267011"
                    code = row.get("Last Result")
                    job.update(registered=True, status=row.get("Status"), next_run=row.get("Next Run Time"),
                               last_run="never" if never else last_run,
                               last_result="not run yet" if never else ("success" if code == "0" else f"exit code {code}"),
                               command=redact(row.get("Task To Run")))
            except Exception as exc:  # pragma: no cover - scheduler not reachable
                job["error"] = str(exc)[:200]
        log = PROJECT_ROOT / "data" / LOGS[name]
        if log.exists():
            mtime = datetime.fromtimestamp(log.stat().st_mtime, tz=timezone.utc).replace(tzinfo=None)
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
            job.update(log_file=f"data/{LOGS[name]}", log_updated=iso_z(mtime), log_tail=redact("\n".join(tail)))
        out.append(job)
    return out


STARTUP_REFRESH: dict = {}   # result of the AUTO_REFRESH_ON_START background refresh (filled by app.py)


def schedule_info() -> dict:
    """How this deployment keeps its data fresh. Windows: the two Task Scheduler jobs. Elsewhere (e.g. a
    Render web service) there is no scheduler: data is refreshed on start-up or by the page actions."""
    if os.name == "nt":
        return {"mode": "windows_task_scheduler", "jobs": scheduled_jobs(),
                "note": "Windows Task Scheduler · jobs run only while this Windows user is logged on and the PC is awake."}
    auto = os.getenv("AUTO_REFRESH_ON_START", "").lower() in ("1", "true", "yes")
    note = ("No scheduler on this server. Prices, news and a fresh prediction are refreshed each time the server "
            "starts; use the page actions (admin token) to refresh by hand. The scheduled 08:30 / 16:30 IST jobs "
            "run on the Windows PC (scripts/schedule_windows.ps1).") if auto else \
           "No scheduler on this server. Use the page actions (admin token) or the CLI to refresh data."
    return {"mode": "startup_refresh" if auto else "manual", "jobs": [], "note": note,
            "startup_refresh": dict(STARTUP_REFRESH) or None}


@router.get("/schedule")
def schedule():
    return schedule_info()


@router.get("/status")
def status(db: Session = Depends(get_db)):
    """One call for the Overview page."""
    s = get_settings()
    active, skipped = enabled_providers()
    pver, _ = params_svc.get_active(db)
    snap = universe_on(db)
    total = db.scalar(select(func.count(NewsArticle.id)))
    last_fetch = db.scalar(select(func.max(NewsArticle.fetched_at)))
    verified = db.scalar(select(func.count(NewsArticle.id)).where(NewsArticle.availability_status == "verified_pre_market"))
    by_provider = dict(db.execute(select(NewsArticle.provider, func.count(NewsArticle.id)).group_by(NewsArticle.provider)).all())

    run = db.scalar(select(PredictionRun).where(PredictionRun.is_backtest.is_(False)).order_by(PredictionRun.created_at.desc()).limit(1))
    run_info = None
    if run is not None:
        preds = db.scalars(select(Prediction).where(Prediction.run_id == run.id)).all()
        today = [p for p in preds if p.horizon_type == "today"]
        # latest review per company (reviews are stored per prediction; count the next-session one)
        vals: dict[str, int] = {}
        for p in today:
            latest = max(p.validations, key=lambda v: v.created_at, default=None)
            if latest is not None:
                vals[latest.validation_status] = vals.get(latest.validation_status, 0) + 1
        run_info = {"run_id": run.id, "created_at": iso_z(run.created_at), "model_version": run.model_version,
                    "param_version": run.param_version, "target_session": str(run.target_session) if run.target_session else None,
                    "prediction_cutoff": iso_z(run.prediction_cutoff), "universe_version": run.universe_version,
                    "predictions": len(preds), "ok": sum(p.status == "OK" for p in preds),
                    "next_session_directions": {d: sum(p.predicted_direction == d for p in today) for d in ("UP", "DOWN", "NEUTRAL")},
                    "data_quality_flags": (run.data_quality or {}).get("flags"),
                    "llm_validations": vals}
    live_scored = db.scalar(select(func.count(ActualResult.id)).join(Prediction).join(PredictionRun)
                            .where(PredictionRun.is_backtest.is_(False)))
    live_pending = db.scalar(select(func.count(Prediction.id)).join(PredictionRun)
                             .where(PredictionRun.is_backtest.is_(False), Prediction.status == "OK",
                                    ~Prediction.id.in_(select(ActualResult.prediction_id))))
    bt = db.scalar(select(BacktestRun).where(BacktestRun.status == "completed").order_by(BacktestRun.created_at.desc()).limit(1))
    bt_info = None
    if bt is not None:
        today = ((bt.report_json or {}).get("metrics_by_horizon") or {}).get("today") or {}
        bc = today.get("baseline_comparison") or {}
        base = {k: v.get("mae") for k, v in (bc.get("baselines") or {}).items()}
        bt_info = {"run_id": bt.id, "finished_at": iso_z(bt.finished_at), "period": f"{bt.start_date} → {bt.end_date}",
                   "param_version": bt.param_version, "n": (bt.metrics or {}).get("n"),
                   "next_session_mae": (today.get("mae") or {}).get("value"),
                   "next_session_sign_accuracy": (today.get("sign_accuracy") or {}).get("value"),
                   "baseline_mae": base, "beats_all_baselines": bc.get("beats_all_baselines_mae"),
                   "tuning_reason": ((bt.report_json or {}).get("tuning") or {}).get("reason")}
    return {
        "generated_at": iso_z(utcnow()),
        "system1": {"articles": total, "verified_pre_market": verified, "last_fetched_at": iso_z(last_fetch),
                    "articles_by_provider": by_provider, "providers_active": [p.name for p in active],
                    "providers_skipped": skipped, "sentiment_engine": s.sentiment_engine},
        "system2": {"latest_run": run_info, "live_outcomes_scored": live_scored, "live_outcomes_pending": live_pending,
                    **provider_status()},
        "system3": {"latest_backtest": bt_info},
        "config": {"active_parameters": pver, "universe_version": snap.version,
                   "universe_members": [m.ticker for m in snap.members], "display_timezone": s.display_timezone},
        "schedule": schedule_info(),
    }


@router.get("/reports")
def reports():
    d = get_settings().reports_dir
    if not d.exists():
        return []
    out = []
    for f in sorted(d.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not _REPORT_NAME.match(f.name):
            continue
        kind = {"bt": "backtest", "exp": "experiment", "audit": "news audit"}[f.name.split("-")[0]]
        out.append({"name": f.name, "kind": kind, "format": f.suffix[1:], "size": f.stat().st_size,
                    "modified": iso_z(datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).replace(tzinfo=None)),
                    "url": f"/api/reports/{f.name}"})
    return out


@router.get("/reports/{name}")
def report_file(name: str):
    if not _REPORT_NAME.match(name):          # no path traversal: fixed name pattern, flat directory
        raise HTTPException(404, "unknown report")
    f = get_settings().reports_dir / name
    if not f.exists():
        raise HTTPException(404, "unknown report")
    text = f.read_text(encoding="utf-8")
    if name.endswith(".json"):
        return JSONResponse(json.loads(text))
    return PlainTextResponse(text, media_type="text/markdown; charset=utf-8")
