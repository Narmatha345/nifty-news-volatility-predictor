"""OpenAI LLM-validation tests. No network: a fake client records exactly what would be sent.
All data is SYNTHETIC test data; keys below are fake test strings."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import httpx2
import openai
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.config.settings import get_settings
from backend.database.models import LLMValidation, Prediction
from backend.services import llm as llm_mod
from backend.services import orchestrator
from backend.services.llm import OPENAI_NOT_CONFIGURED, OpenAIProvider, get_llm, redact
from backend.services.timeutil import market_close_utc
from backend.system2_prediction import validator
from backend.system2_prediction.validator import assert_no_leakage, build_payload, check_response, validate_run
from backend.system3_backtesting import runner
from tests.test_phase3_data_integrity import _article, _seed_prices

FAKE_KEY = "sk-test-FAKEKEY-0123456789abcdefSECRET"
GOOD = {"validation_status": "weakly_supported", "reason": "One verified contract headline supports a small move.",
        "key_supporting_factors": ["contract win"], "key_contradicting_factors": [],
        "data_quality_issues": ["single source"], "cutoff_check": "passed", "missing_information": ["order size"]}


@pytest.fixture
def openai_env(monkeypatch):
    def set_env(key=FAKE_KEY, model="test-model"):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", key)
        monkeypatch.setenv("OPENAI_MODEL", model)
        get_settings.cache_clear()
    yield set_env
    get_settings.cache_clear()


class FakeClient:
    """Mimics openai.OpenAI().chat.completions.create and records every request."""
    def __init__(self, content=None, exc=None, finish="stop", refusal=None):
        self.calls, self.content, self.exc, self.finish, self.refusal = [], content, exc, finish, refusal
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        msg = SimpleNamespace(content=self.content, refusal=self.refusal)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=self.finish)])

    def sent_text(self) -> str:
        return "\n".join(m["content"] for c in self.calls for m in c["messages"])


def provider(client):
    return OpenAIProvider(client=client)


@pytest.fixture
def world(session):
    _seed_prices(session)
    ids = {
        "verified": _article(session, "HDFC Bank wins large contract", datetime(2026, 9, 28, 10, 0, 5),
                             datetime(2026, 9, 28, 10, 5, 0)).id,
        "late": _article(session, "HDFC Bank faces regulatory action", datetime(2026, 9, 28, 11, 0, 7),
                         datetime(2026, 10, 1, 9, 0, 0)).id,
        "dated": _article(session, "HDFC Bank shares in focus", datetime(2026, 9, 28, 7, 0), datetime(2026, 10, 1),
                          precision="date_only").id,
    }
    orchestrator.analyze_news(session)
    out = orchestrator.predict(session, datetime(2026, 9, 29, 2, 0))
    return {"ids": ids, "run_id": out["run_id"]}


def _snapshot(session, run_id):
    return {p.id: (p.predicted_movement, p.predicted_direction, p.sentiment_score, p.previous_close,
                   json.dumps(p.features, sort_keys=True))
            for p in session.scalars(select(Prediction).where(Prediction.run_id == run_id))}


# 1 ------------------------------------------------------------------------------------------
def test_openai_provider_initializes_with_env_configuration(openai_env):
    openai_env()
    p = get_llm()
    assert isinstance(p, OpenAIProvider) and p.name == "openai" and p.model == "test-model"
    assert isinstance(p.client, openai.OpenAI)
    assert FAKE_KEY not in repr(p) and FAKE_KEY not in repr(get_settings())


# 2 ------------------------------------------------------------------------------------------
def test_missing_key_is_graceful_not_performed(session, world, openai_env):
    openai_env(key="")
    p = get_llm()
    assert p.configured is False and p.name == "openai"
    before = _snapshot(session, world["run_id"])
    res = validate_run(session, world["run_id"], p, tickers=["HDFCBANK"])["HDFCBANK"]
    assert res["validation_status"] == "not_performed" and res["validation_performed"] is False
    assert res["reason"] == OPENAI_NOT_CONFIGURED == "LLM validation not performed: OpenAI API key is not configured."
    assert _snapshot(session, world["run_id"]) == before
    assert llm_mod.provider_status()["llm_configured"] is False


# 3 ------------------------------------------------------------------------------------------
def test_successful_validation_is_stored_separately(session, world, openai_env):
    openai_env()
    client = FakeClient(json.dumps(GOOD))
    before = _snapshot(session, world["run_id"])
    res = validate_run(session, world["run_id"], provider(client), tickers=["HDFCBANK"])["HDFCBANK"]
    assert res["validation_status"] == "weakly_supported" and res["validation_performed"] is True
    call = client.calls[0]
    assert call["model"] == "test-model" and call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["strict"] is True
    v = session.scalars(select(LLMValidation).where(LLMValidation.provider == "openai")).first()
    assert (v.model, v.validation_status, v.cutoff_check, v.request_status) == ("test-model", "weakly_supported",
                                                                               "passed", "ok")
    assert v.supporting_factors == ["contract win"] and v.data_quality_issues == ["single source"]
    assert v.missing_information == ["order size"] and v.reasoning.startswith("One verified")
    assert _snapshot(session, world["run_id"]) == before


# 4 ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("content", ["not json at all", json.dumps({"validation_status": "supported"}),
                                     json.dumps({**GOOD, "validation_status": "bullish"}), json.dumps([1, 2])])
def test_invalid_response_is_a_validation_error_without_crashing(session, world, openai_env, content):
    openai_env()
    before = _snapshot(session, world["run_id"])
    res = validate_run(session, world["run_id"], provider(FakeClient(content)), tickers=["HDFCBANK"])["HDFCBANK"]
    assert res["validation_status"] == "validation_error" and res["request_status"] == "invalid_response"
    assert _snapshot(session, world["run_id"]) == before


# 5 ------------------------------------------------------------------------------------------
def test_api_timeout_keeps_the_prediction(session, monkeypatch, openai_env):
    openai_env()
    _seed_prices(session)
    req = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    failing = FakeClient(exc=openai.APITimeoutError(request=req))
    monkeypatch.setattr(orchestrator, "collect_and_analyze", lambda s, start=None, end=None: {"new": 0})
    monkeypatch.setattr(orchestrator, "refresh_market_data", lambda s: {})
    monkeypatch.setattr(llm_mod, "get_llm", lambda name=None: OpenAIProvider(client=failing))
    clock = {"t": datetime(2026, 9, 29, 1, 0)}

    def tick():
        clock["t"] += timedelta(seconds=30)
        return clock["t"]
    monkeypatch.setattr(orchestrator, "utcnow", tick)
    steps = orchestrator.daily_run(session, validate=True)
    assert steps["predictions"] == 100
    statuses = {r["validation_status"] for r in steps["llm_validation"].values()}
    assert statuses == {"api_error"} and failing.calls
    session.rollback()                         # even a rollback after validation cannot lose the prediction
    assert len(session.scalars(select(Prediction).where(Prediction.run_id == steps["prediction_run"])).all()) == 100


# 6 ------------------------------------------------------------------------------------------
def test_openai_cannot_modify_the_numeric_prediction(session, world, openai_env):
    openai_env()
    sneaky = {**GOOD, "predicted_return_percent": 9.9, "predicted_direction": "UP"}
    before = _snapshot(session, world["run_id"])
    res = validate_run(session, world["run_id"], provider(FakeClient(json.dumps(sneaky))))
    assert {r["validation_status"] for r in res.values()} == {"validation_error"}
    assert check_response(sneaky) and "unexpected fields" in check_response(sneaky)[0]
    validate_run(session, world["run_id"], provider(FakeClient(json.dumps(GOOD))))
    assert _snapshot(session, world["run_id"]) == before


# 7 / 9 -----------------------------------------------------------------------------------------
def test_only_cutoff_eligible_news_is_sent(session, world, openai_env):
    openai_env()
    client = FakeClient(json.dumps(GOOD))
    validate_run(session, world["run_id"], provider(client), tickers=["HDFCBANK"])
    sent = client.sent_text()
    payload = json.loads(sent.split("Data (JSON):\n", 1)[1])
    ids = {n["news_id"] for n in payload["news_evidence"]}
    assert world["ids"]["verified"] in ids
    assert world["ids"]["late"] not in ids and "regulatory action" not in sent          # fetched after cut-off
    assert world["ids"]["dated"] not in ids                                            # date-only, not verified
    assert all(n["eligibility"] == "verified" for n in payload["news_evidence"])
    assert payload["prediction_cutoff"] == "2026-09-29T03:45:00Z" and payload["availability_policy"] == "verified"


def test_date_only_news_follows_the_run_policy(session, world):
    preds = session.scalars(select(Prediction).where(Prediction.run_id == world["run_id"],
                                                     Prediction.ticker == "HDFCBANK")).all()
    p0 = preds[0]
    extra = {"news_id": 999, "entity": "HDFCBANK", "role": "new_company_news", "title": "date-only item",
             "available_at": "2026-09-29T00:00:00Z", "fetched_at": "2026-10-01T00:00:00Z",
             "timestamp_precision": "date_only", "eligibility": "uncertain", "event_type": "other"}
    p0.news_inputs = list(p0.news_inputs) + [extra]
    assert 999 not in {n["news_id"] for n in build_payload(session, preds)["news_evidence"]}   # verified policy
    p0.run.availability_policy = "provider_timestamp"                                          # research replay
    sent = {n["news_id"]: n for n in build_payload(session, preds)["news_evidence"]}
    assert sent[999]["eligibility"] == "uncertain"                   # included, but labelled as uncertain
    future = {**extra, "news_id": 1000, "available_at": "2026-09-29T05:00:00Z"}                # after the cut-off
    p0.news_inputs = list(p0.news_inputs) + [future]
    assert 1000 not in {n["news_id"] for n in build_payload(session, preds)["news_evidence"]}
    session.rollback()


# 8 ------------------------------------------------------------------------------------------
def test_actual_outcome_is_never_sent(session, openai_env):
    openai_env()
    _seed_prices(session)
    out = orchestrator.predict(session, datetime(2026, 10, 1, 14, 0))           # targets Mon 5 Oct
    scored = runner.score_live_predictions(session, now=market_close_utc(date(2026, 10, 5)) + timedelta(hours=1))
    assert scored["scored"] > 0                                                 # outcomes now exist in the DB
    client = FakeClient(json.dumps(GOOD))
    validate_run(session, out["run_id"], provider(client), tickers=["HDFCBANK"])
    sent = client.sent_text()
    p = session.scalars(select(Prediction).where(Prediction.run_id == out["run_id"], Prediction.ticker == "HDFCBANK",
                                                 Prediction.horizon_type == "today")).first()
    assert p.actual is not None
    assert '"actual' not in sent and '"error"' not in sent and '"abs_error"' not in sent
    assert repr(p.actual.actual_close) not in sent and f"{p.actual.actual_movement}" not in sent


def test_leakage_guard_blocks_outcome_fields_and_post_cutoff_news(session, world, monkeypatch, openai_env):
    openai_env()
    cutoff = datetime(2026, 9, 29, 3, 45)
    with pytest.raises(llm_mod.LLMError):
        assert_no_leakage({"predictions": [{"actual_close": 1.0}]}, cutoff, cutoff)
    with pytest.raises(llm_mod.LLMError):
        assert_no_leakage({"news_evidence": [{"news_id": 1, "available_at": "2026-09-29T04:00:00Z"}]}, cutoff, cutoff)
    real = validator.build_payload
    monkeypatch.setattr(validator, "build_payload", lambda s, preds: {**real(s, preds), "actual_return": 1.2})
    client = FakeClient(json.dumps(GOOD))
    res = validate_run(session, world["run_id"], provider(client), tickers=["HDFCBANK"])["HDFCBANK"]
    assert res["request_status"] == "leakage_blocked" and client.calls == []    # nothing was sent


# 10 -----------------------------------------------------------------------------------------
def test_api_key_never_appears_in_logs_storage_or_responses(session, world, openai_env, caplog, monkeypatch):
    openai_env()
    # keep the API on this test's in-memory database (never the real data/nifty_nvp.db)
    monkeypatch.setattr("backend.api.app.init_engine", lambda url=None: None)
    caplog.set_level(logging.DEBUG)
    req = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions",
                         headers={"Authorization": f"Bearer {FAKE_KEY}"})
    resp = httpx2.Response(401, request=req)
    err = openai.AuthenticationError(f"Incorrect API key provided: {FAKE_KEY}. Authorization: Bearer {FAKE_KEY}",
                                     response=resp, body=None)
    res = validate_run(session, world["run_id"], provider(FakeClient(exc=err)), tickers=["HDFCBANK"])
    bad = provider(FakeClient(exc=openai.APIConnectionError(message=f"proxy said Bearer {FAKE_KEY}", request=req)))
    res2 = validate_run(session, world["run_id"], bad, tickers=["TCS"])
    stored = session.scalars(select(LLMValidation)).all()
    blobs = [json.dumps(res, default=str), json.dumps(res2, default=str), caplog.text] + \
            [json.dumps({"e": v.error_message, "r": v.reasoning, "raw": v.raw_response, "p": v.request_payload},
                        default=str) for v in stored]
    assert all(FAKE_KEY not in b for b in blobs)
    assert "[REDACTED]" in res2["TCS"]["reason"]
    assert redact(f"Authorization: Bearer {FAKE_KEY}").count("[REDACTED]") >= 1
    from backend.api.app import app
    session.commit()
    with TestClient(app) as c:
        for path in ("/api/config", "/api/predictions?include_backtest=false"):
            assert FAKE_KEY not in c.get(path).text
