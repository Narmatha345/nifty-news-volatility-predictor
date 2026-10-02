"""Provider registry."""
from __future__ import annotations

from backend.config.settings import get_settings
from backend.system1_news.providers.base import NewsProvider, RawArticle  # noqa: F401
from backend.system1_news.providers.google_news_rss import GoogleNewsRSSProvider
from backend.system1_news.providers.keyed import GNewsProvider, MarketauxProvider, NewsAPIProvider
from backend.system1_news.providers.publisher_rss import PublisherRSSProvider

REGISTRY: dict[str, type[NewsProvider]] = {
    GoogleNewsRSSProvider.name: GoogleNewsRSSProvider,
    PublisherRSSProvider.name: PublisherRSSProvider,
    NewsAPIProvider.name: NewsAPIProvider,
    GNewsProvider.name: GNewsProvider,
    MarketauxProvider.name: MarketauxProvider,
}


def register(provider_cls: type[NewsProvider]) -> None:
    REGISTRY[provider_cls.name] = provider_cls


def enabled_providers() -> tuple[list[NewsProvider], list[str]]:
    """Providers listed in NEWS_PROVIDERS that are configured, plus the names that were skipped."""
    active, skipped = [], []
    for name in get_settings().news_providers:
        cls = REGISTRY.get(name)
        if cls is None:
            skipped.append(f"{name} (unknown)")
            continue
        p = cls()
        (active if p.is_configured() else skipped).append(p if p.is_configured() else f"{name} (no API key)")
    return active, skipped
