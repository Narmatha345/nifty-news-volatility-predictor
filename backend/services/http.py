"""Shared HTTP client: retries with exponential backoff, Retry-After handling, per-host rate limits."""
from __future__ import annotations

import logging
import random
import ssl
import threading
import time
from typing import Any

import httpx

from backend.config.settings import get_settings

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class ProviderError(RuntimeError):
    """Raised when a provider call fails permanently (after retries or on a non-retryable status)."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class RateLimiter:
    """Minimum interval between calls, per key. Thread-safe."""

    def __init__(self) -> None:
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, key: str, min_interval_s: float) -> None:
        if min_interval_s <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait_for = self._last.get(key, 0.0) + min_interval_s - now
            self._last[key] = max(now, now + wait_for)
        if wait_for > 0:
            time.sleep(wait_for)


_rate_limiter = RateLimiter()


def request_with_retry(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    json: Any = None,
    rate_key: str | None = None,
    min_interval_s: float = 0.0,
    max_retries: int | None = None,
    timeout_s: float | None = None,
    sleep=time.sleep,
) -> httpx.Response:
    settings = get_settings()
    max_retries = settings.http_max_retries if max_retries is None else max_retries
    timeout_s = timeout_s or settings.http_timeout_s
    verify: Any = ssl.create_default_context(cafile=settings.ca_bundle) if settings.ca_bundle else True
    attempt = 0
    while True:
        if rate_key:
            _rate_limiter.wait(rate_key, min_interval_s)
        try:
            with httpx.Client(timeout=timeout_s, verify=verify, follow_redirects=True,
                              headers={"User-Agent": "nifty-news-volatility-predictor/1.0"}) as client:
                resp = client.request(method, url, params=params, headers=headers, json=json)
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            if attempt >= max_retries:
                raise ProviderError(f"network error after {attempt + 1} attempts: {exc}") from exc
            delay = _backoff(attempt)
            log.warning("network error on %s (%s); retry in %.1fs", _redact(url), exc, delay)
        else:
            if resp.status_code < 400:
                return resp
            if resp.status_code not in RETRYABLE_STATUS or attempt >= max_retries:
                raise ProviderError(f"HTTP {resp.status_code}: {resp.text[:300]}", resp.status_code)
            delay = _retry_after(resp) or _backoff(attempt)
            log.warning("HTTP %s on %s; retry in %.1fs", resp.status_code, _redact(url), delay)
        attempt += 1
        sleep(delay)


def _backoff(attempt: int) -> float:
    return min(60.0, (2 ** attempt) + random.uniform(0, 0.5))


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after")
    if not value:
        return None
    try:
        return min(120.0, float(value))
    except ValueError:
        return None


def _redact(url: str) -> str:
    return url.split("?")[0]
