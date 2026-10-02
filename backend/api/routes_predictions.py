"""System 2 endpoints: /predictions, /predictions/validate."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.api.schemas import parse_date, parse_dt
from backend.database.db import get_db
from backend.database.models import Prediction, PredictionRun
from backend.services import orchestrator
from backend.services.llm import get_llm
from backend.system2_prediction.predictor import prediction_to_dict

router = APIRouter()


class PredictRequest(BaseModel):
    as_of: str | None = None
    validate_with_llm: bool = False


class ValidateRequest(BaseModel):
    run_id: str | None = None      # default: latest live run
    tickers: list[str] | None = None
    provider: str | None = None    # openai | anthropic | ollama (default: LLM_PROVIDER)


@router.post("/predictions")
def create_predictions(req: PredictRequest, db: Session = Depends(get_db)):
    out = orchestrator.predict(db, parse_dt(req.as_of) if req.as_of else None)
    if req.validate_with_llm:
        out["llm_validation"] = orchestrator.validate_safely(db, out["run_id"])
    return out


@router.get("/predictions")
def list_predictions(ticker: str | None = None, horizon_type: str | None = None, run_id: str | None = None,
                     target_date: str | None = None, include_backtest: bool = False, limit: int = 500,
                     db: Session = Depends(get_db)):
    q = select(Prediction).join(PredictionRun)
    if run_id is None and not include_backtest:
        latest = db.scalar(select(PredictionRun.id).where(PredictionRun.is_backtest.is_(False))
                           .order_by(PredictionRun.created_at.desc()).limit(1))
        if latest is None:
            return {"run_id": None, "status": "MISSING", "message": "No predictions yet. POST /predictions.",
                    "items": []}
        run_id = latest
    if run_id:
        q = q.where(Prediction.run_id == run_id)
    if not include_backtest:
        q = q.where(PredictionRun.is_backtest.is_(False))
    if ticker:
        q = q.where(Prediction.ticker == ticker)
    if horizon_type:
        q = q.where(Prediction.horizon_type == horizon_type)
    if target_date:
        q = q.where(Prediction.target_date == parse_date(target_date))
    items = [prediction_to_dict(p) for p in db.scalars(q.order_by(Prediction.id).limit(limit))]
    return {"run_id": run_id, "count": len(items), "items": items,
            "disclaimer": "Model estimates for research; not guaranteed outcomes."}


@router.get("/predictions/runs")
def list_runs(limit: int = 30, db: Session = Depends(get_db)):
    rows = db.scalars(select(PredictionRun).where(PredictionRun.is_backtest.is_(False))
                      .order_by(PredictionRun.created_at.desc()).limit(limit))
    return [{"run_id": r.id, "as_of": r.as_of.isoformat() + "Z", "model_version": r.model_version,
             "param_version": r.param_version, "target_session": str(r.target_session) if r.target_session else None,
             "prediction_cutoff": r.prediction_cutoff.isoformat() + "Z" if r.prediction_cutoff else None,
             "universe_version": r.universe_version} for r in rows]


@router.get("/predictions/live-metrics")
def live_track_record(db: Session = Depends(get_db)):
    """Accuracy of LIVE predictions whose outcome is known (read-only; scoring happens in `outcomes`)."""
    from backend.system3_backtesting.runner import live_metrics
    return live_metrics(db)


@router.post("/predictions/validate")
def validate(req: ValidateRequest, db: Session = Depends(get_db)):
    run_id = req.run_id or db.scalar(select(PredictionRun.id).where(PredictionRun.is_backtest.is_(False))
                                     .order_by(PredictionRun.created_at.desc()).limit(1))
    if run_id is None:
        raise HTTPException(404, "no prediction run to validate")
    llm = get_llm(req.provider)
    if llm is None:
        raise HTTPException(400, "LLM_PROVIDER is 'none'")
    return {"run_id": run_id, "provider": llm.name, "model": llm.model,
            "note": "Validation is advisory and stored separately; predictions are not modified.",
            "results": orchestrator.validate_safely(db, run_id, req.provider, req.tickers)}
