"""FastAPI application. Run: uvicorn backend.api.app:app --reload

Deployment switches (environment variables, all optional; nothing changes locally when unset):
  SEED_DATABASE             path of a gzipped SQLite snapshot restored on first start when the SQLite
                            database file does not exist yet (ephemeral disks, e.g. Render free tier).
  AUTO_REFRESH_ON_START     "true": after start-up, refresh prices, collect news and make a fresh
                            prediction in a background thread (no LLM calls - those stay manual).
"""
from __future__ import annotations

import gzip
import hashlib
import logging
import os
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api import routes_app, routes_backtest, routes_meta, routes_news, routes_predictions
from backend.config.settings import PROJECT_ROOT, get_settings
from backend.database.db import init_engine, session_scope
from backend.services.llm import redact
from backend.services.orchestrator import bootstrap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def restore_seed_database() -> str | None:
    """Unpack SEED_DATABASE into the SQLite file if that file does not exist yet."""
    seed, url = os.getenv("SEED_DATABASE", ""), get_settings().database_url
    if not seed or not url.startswith("sqlite:///"):
        return None
    target = Path(url.replace("sqlite:///", ""))
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    seed_path = Path(seed) if Path(seed).is_absolute() else PROJECT_ROOT / seed
    if target.exists() or not seed_path.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(seed_path, "rb") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)
    log.info("restored seed database %s -> %s", seed_path, target)
    return str(target)


def _refresh_in_background() -> None:
    def work():
        from backend.services import orchestrator
        from backend.services.timeutil import iso_z, utcnow
        steps = routes_app.STARTUP_REFRESH
        steps.clear()
        steps["started_at"] = iso_z(utcnow())
        for name, fn in (("prices", orchestrator.refresh_market_data), ("news", orchestrator.collect_and_analyze),
                         ("predict", orchestrator.predict)):
            try:
                with session_scope() as s:
                    out = fn(s)
                steps[name] = "ok" if name != "predict" else f"ok ({out['run_id']}, target {out['target_session']})"
            except Exception as exc:     # a failing source must not take the web app down
                steps[name] = f"failed: {type(exc).__name__}: {redact(str(exc))[:200]}"
        try:   # warm the availability-audit cache so the first Settings / System 1 visit is fast
            with session_scope() as s:
                routes_news.news_audit(None, None, s)
        except Exception as exc:
            log.warning("audit cache warm-up failed: %s", type(exc).__name__)
        steps["finished_at"] = iso_z(utcnow())
        log.info("start-up refresh finished: %s", steps)
    threading.Thread(target=work, name="startup-refresh", daemon=True).start()


@asynccontextmanager
async def lifespan(_: FastAPI):
    restore_seed_database()
    init_engine()
    with session_scope() as s:
        bootstrap(s)
    if os.getenv("AUTO_REFRESH_ON_START", "").lower() in ("1", "true", "yes"):
        _refresh_in_background()
    yield


app = FastAPI(title="NIFTY News to Volatility Predictor", version="1.0.0", lifespan=lifespan,
              description="Research system: news sentiment -> predicted % movement (model estimates, "
                          "not guaranteed outcomes).")


app.add_middleware(GZipMiddleware, minimum_size=1000)   # large JSON (predictions, reports) compresses ~5-10x


for r in (routes_news.router, routes_predictions.router, routes_backtest.router, routes_meta.router, routes_app.router):
    app.include_router(r, prefix="/api")

# Web-app pages (client-side rendered). Each has its own URL so it can be bookmarked / refreshed.
PAGES = ("overview", "system-1", "system-2", "system-3", "settings")


def _asset_version(name: str) -> str:
    return hashlib.sha256((PROJECT_ROOT / "frontend" / name).read_bytes()).hexdigest()[:10]


def _page():
    """The app shell. styles.css / app.js get a content-hash query string, so after an update a browser
    never combines the new page with a stale cached stylesheet or script."""
    html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    for name in ("styles.css", "app.js"):
        html = html.replace(f'"/{name}"', f'"/{name}?v={_asset_version(name)}"')
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


app.add_api_route("/", _page, include_in_schema=False)
for _p in PAGES:
    app.add_api_route(f"/{_p}", _page, include_in_schema=False)

app.mount("/", StaticFiles(directory=PROJECT_ROOT / "frontend", html=True), name="frontend")
