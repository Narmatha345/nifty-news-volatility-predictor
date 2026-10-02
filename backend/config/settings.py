"""Runtime settings from environment variables (.env supported). Secrets never live in code."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional at runtime
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    database_url: str
    display_timezone: str
    market_timezone: str
    news_providers: list[str]
    newsapi_key: str = field(repr=False)          # secrets are excluded from repr / logs
    gnews_api_key: str = field(repr=False)
    marketaux_api_key: str = field(repr=False)
    sentiment_engine: str
    quarter_convention: str
    llm_provider: str
    anthropic_model: str
    ollama_base_url: str
    ollama_model: str
    openai_api_key: str = field(repr=False)
    openai_model: str
    admin_token: str = field(repr=False)   # protects state-changing API calls when set
    ca_bundle: str
    reports_dir: Path
    http_timeout_s: float = 20.0
    llm_timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "120"))
    http_max_retries: int = 4
    extra: dict = field(default_factory=dict)


@lru_cache
def get_settings() -> Settings:
    default_db = f"sqlite:///{(PROJECT_ROOT / 'data' / 'nifty_nvp.db').as_posix()}"
    ca_bundle = os.getenv("CA_BUNDLE", "")
    if ca_bundle:
        # Corporate proxies / antivirus HTTPS inspection: make every HTTP stack trust the bundle.
        for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            os.environ.setdefault(var, ca_bundle)
    return Settings(
        database_url=os.getenv("DATABASE_URL", default_db),
        display_timezone=os.getenv("DISPLAY_TIMEZONE", "Asia/Kolkata"),
        market_timezone="Asia/Kolkata",
        news_providers=_csv(os.getenv("NEWS_PROVIDERS", "google_news_rss,publisher_rss,newsapi,gnews,marketaux")),
        newsapi_key=os.getenv("NEWSAPI_KEY", ""),
        gnews_api_key=os.getenv("GNEWS_API_KEY", ""),
        marketaux_api_key=os.getenv("MARKETAUX_API_KEY", ""),
        sentiment_engine=os.getenv("SENTIMENT_ENGINE", "lexicon"),
        quarter_convention=os.getenv("QUARTER_CONVENTION", "indian_fy"),
        llm_provider=os.getenv("LLM_PROVIDER", "anthropic"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "").strip(),
        admin_token=os.getenv("ADMIN_TOKEN", "").strip(),
        ca_bundle=ca_bundle,
        reports_dir=Path(os.getenv("REPORTS_DIR", str(PROJECT_ROOT / "reports"))),
    )
