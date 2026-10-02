import pytest
from fastapi.testclient import TestClient

from backend.database import db as dbmod


@pytest.fixture
def client(monkeypatch):
    real_init = dbmod.init_engine
    monkeypatch.setattr("backend.api.app.init_engine", lambda url=None: real_init("sqlite://"))
    from backend.api.app import app
    with TestClient(app) as c:
        yield c


def test_core_endpoints(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    comps = client.get("/api/companies").json()
    assert len(comps) == 10 and comps[0]["ticker"] == "HDFCBANK"
    params = client.get("/api/parameters").json()
    assert params[0]["status"] == "active"
    assert client.get("/api/sentiment").json()["status"] == "MISSING"
    assert client.get("/api/predictions").json()["status"] == "MISSING"
    assert client.get("/api/market-data", params={"ticker": "HDFCBANK.NS"}).json()["status"] == "MISSING"
    cfg = client.get("/api/config").json()
    assert cfg["horizons_today"][0]["horizon_type"] == "today"


def test_adhoc_analysis(client):
    r = client.post("/api/news/analyze", json={"title": "HDFC Bank reports strong quarterly profit growth"}).json()
    assert r["relevant"] and r["entities"][0]["entity"] == "HDFCBANK" and r["sentiment_score"] > 0


def test_company_update_and_parameter_activation_guard(client):
    r = client.put("/api/companies/HDFCBANK", json={"nifty_weight": 12.5, "weight_as_of": "2026-09-30"})
    assert r.json()["nifty_weight"] == 12.5
    assert client.post("/api/parameters/does-not-exist/activate").status_code == 404
    assert client.post("/api/backtest", json={"start": "2025-01-01", "end": "2025-02-01",
                                              "horizon_types": ["weekly"]}).status_code == 422


def test_dashboard_served(client):
    # every page has its own URL and serves the app shell (client-side rendered)
    for path in ("/", "/overview", "/system-1", "/system-2", "/system-3", "/settings"):
        r = client.get(path)
        assert r.status_code == 200 and 'id="view"' in r.text and "/app.js" in r.text, path
    assert client.get("/app.js").status_code == 200 and client.get("/styles.css").status_code == 200


def test_app_status_and_reports_endpoints(client):
    s = client.get("/api/status").json()
    assert {"system1", "system2", "system3", "config", "schedule"} <= set(s)
    assert s["config"]["universe_members"] and "llm_provider" in s["system2"]
    assert client.get("/api/predictions/live-metrics").status_code == 200
    assert isinstance(client.get("/api/reports").json(), list)
    assert client.get("/api/reports/..%2F.env").status_code == 404        # no path traversal
    assert client.get("/api/reports/not-a-report.txt").status_code == 404


def test_schedule_reports_how_this_deployment_refreshes(client):
    sch = client.get("/api/schedule").json()
    assert sch["mode"] in ("windows_task_scheduler", "startup_refresh", "manual") and sch["note"]
    assert client.get("/api/status").json()["schedule"]["mode"] == sch["mode"]
    if sch["mode"] != "windows_task_scheduler":
        assert sch["jobs"] == []          # no fake "not registered" Windows jobs on a Linux server


def test_news_audit_is_cached_until_the_data_changes(client, monkeypatch):
    from datetime import datetime
    from backend.database.db import session_scope
    from backend.database.models import FetchLog
    from backend.system1_news import availability
    calls = []
    real = availability.audit_report
    monkeypatch.setattr(availability, "audit_report", lambda *a, **k: calls.append(1) or real(*a, **k))
    first = client.get("/api/news/audit").json()
    assert client.get("/api/news/audit").json() == first and len(calls) == 1      # served from the cache
    with session_scope() as s:
        s.add(FetchLog(provider="google_news_rss", query="q", started_at=datetime(2026, 10, 1), status="ok", items=3))
    after = client.get("/api/news/audit").json()
    assert len(calls) == 2 and after["by_provider"]["google_news_rss"]["rejected_at_collection"]["fetched"] == 3


def test_admin_token_protects_actions_but_not_reading(client, monkeypatch):
    from backend.config.settings import get_settings
    monkeypatch.setenv("ADMIN_TOKEN", "test-admin-token")
    get_settings.cache_clear()
    try:
        assert client.get("/api/auth").json() == {"actions_require_token": True}
        assert client.get("/api/companies").status_code == 200                       # reading stays public
        assert client.post("/api/parameters/x/activate").status_code == 401           # no token
        assert client.post("/api/parameters/x/activate", headers={"X-Admin-Token": "wrong"}).status_code == 401
        r = client.post("/api/parameters/x/activate", headers={"X-Admin-Token": "test-admin-token"})
        assert r.status_code == 404                                                   # passed the guard
        assert "test-admin-token" not in client.get("/api/config").text
    finally:
        monkeypatch.delenv("ADMIN_TOKEN")
        get_settings.cache_clear()
    assert client.get("/api/auth").json() == {"actions_require_token": False}         # local default: open


def test_seed_database_is_restored_only_when_missing(tmp_path, monkeypatch):
    import gzip
    import sqlite3
    from backend.api.app import restore_seed_database
    from backend.config.settings import get_settings
    src = tmp_path / "seed.sqlite"
    sqlite3.connect(src).execute("create table t (x int)").connection.commit()
    gz = tmp_path / "seed.sqlite.gz"
    gz.write_bytes(gzip.compress(src.read_bytes()))
    target = tmp_path / "live" / "db.sqlite"
    monkeypatch.setenv("SEED_DATABASE", str(gz))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{target.as_posix()}")
    get_settings.cache_clear()
    try:
        assert restore_seed_database() == str(target) and target.exists()
        assert restore_seed_database() is None                                        # never overwrites
    finally:
        get_settings.cache_clear()
