"""LLM provider adapters returning schema-constrained JSON. Used only for validation/review.

Providers: openai (OpenAI API, LLM_PROVIDER=openai), anthropic, ollama (local). The provider is chosen
ONLY by LLM_PROVIDER - there is no silent fallback to another provider. A provider that is not
configured is replaced by UnavailableLLM, whose calls fail with an explicit "not performed" reason.

Secrets: API keys come from the environment only, are never logged, and every error message that
can reach storage, logs or API responses passes through `redact()`.
"""
from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod

from backend.config.settings import get_settings
from backend.services.http import request_with_retry

log = logging.getLogger(__name__)

OPENAI_NOT_CONFIGURED = "LLM validation not performed: OpenAI API key is not configured."
OPENAI_MODEL_NOT_CONFIGURED = "LLM validation not performed: OpenAI model is not configured (set OPENAI_MODEL)."


class LLMError(RuntimeError):
    """Any validation-call failure. `kind` is stored as the request status."""
    kind = "api_error"


class LLMNotConfigured(LLMError):
    kind = "not_configured"


class LLMInvalidResponse(LLMError):
    kind = "invalid_response"


_SECRET_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{6,}"), "sk-[REDACTED]"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9_\-\.=]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(authorization['\"]?\s*[:=]\s*['\"]?)[^'\",\s]+(\s+[^'\",\s]+)?"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(x-api-key['\"]?\s*[:=]\s*['\"]?)[^'\",\s]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(api[_-]?key['\"]?\s*[:=]\s*['\"]?)[^'\",\s]+"), r"\1[REDACTED]"),
]


def redact(text: str | None) -> str | None:
    """Remove API keys / authorization headers from any text before it is logged, stored or returned."""
    if not text:
        return text
    out = str(text)
    for key in (get_settings().openai_api_key, os.getenv("ANTHROPIC_API_KEY", ""), os.getenv("OPENAI_API_KEY", "")):
        if key and len(key) >= 8:
            out = out.replace(key, "[REDACTED]")
    for rx, repl in _SECRET_PATTERNS:
        out = rx.sub(repl, out)
    return out


class LLMProvider(ABC):
    name = "base"
    model = ""
    configured = True

    @abstractmethod
    def complete_json(self, system: str, user: str, schema: dict) -> tuple[dict, str]:
        """Returns (parsed_json, raw_text). Raises LLMError (or a subclass) on any failure."""

    def __repr__(self) -> str:                      # never includes credentials
        return f"<{type(self).__name__} {self.name}/{self.model}>"


def _parse(text: str, who: str) -> tuple[dict, str]:
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise LLMInvalidResponse(f"invalid JSON from {who}: {redact((text or '')[:200])}") from exc
    if not isinstance(data, dict):
        raise LLMInvalidResponse(f"{who} returned JSON that is not an object")
    return data, text


class OpenAIProvider(LLMProvider):
    """OpenAI Chat Completions with a strict JSON-schema response format (structured outputs)."""
    name = "openai"

    def __init__(self, model: str | None = None, client=None):
        s = get_settings()
        if not s.openai_api_key:
            raise LLMNotConfigured(OPENAI_NOT_CONFIGURED)
        self.model = model or s.openai_model
        if not self.model:
            raise LLMNotConfigured(OPENAI_MODEL_NOT_CONFIGURED)
        if client is None:
            import openai
            kwargs = {}
            if s.ca_bundle:   # HTTPS interception (antivirus / proxy): trust the configured bundle
                kwargs["http_client"] = openai.DefaultHttpxClient(verify=s.ca_bundle)
            client = openai.OpenAI(api_key=s.openai_api_key, timeout=s.llm_timeout_s, max_retries=2, **kwargs)
        self.client = client

    def complete_json(self, system, user, schema):
        import openai
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "prediction_validation", "schema": schema, "strict": True}},
            )
        except openai.AuthenticationError as exc:
            raise LLMError("OpenAI authentication failed: the configured OPENAI_API_KEY was rejected") from exc
        except openai.RateLimitError as exc:
            raise LLMError("OpenAI rate limit or quota reached; retry later") from exc
        except openai.APITimeoutError as exc:
            raise LLMError("OpenAI request timed out") from exc
        except openai.APIConnectionError as exc:
            raise LLMError(redact(f"OpenAI connection error: {exc}")) from exc
        except openai.APIStatusError as exc:
            raise LLMError(redact(f"OpenAI API error {exc.status_code}: {exc.message}")) from exc
        except openai.OpenAIError as exc:
            raise LLMError(redact(f"OpenAI error: {exc}")) from exc
        choice = resp.choices[0] if getattr(resp, "choices", None) else None
        if choice is None:
            raise LLMInvalidResponse("OpenAI returned no choices")
        msg = choice.message
        if getattr(msg, "refusal", None):
            raise LLMInvalidResponse(redact(f"model declined to validate: {msg.refusal}"))
        if choice.finish_reason == "length":
            raise LLMInvalidResponse("validation response was truncated (finish_reason=length)")
        return _parse(msg.content, "OpenAI")


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str | None = None):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY / ant auth profile
        self.model = model or get_settings().anthropic_model

    def complete_json(self, system, user, schema):
        a = self._anthropic
        try:
            # Server-side fallback: if a safety classifier declines, the API retries on a fallback model.
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema}},
            )
        except a.AuthenticationError as exc:
            raise LLMError("Anthropic authentication failed - set ANTHROPIC_API_KEY") from exc
        except a.RateLimitError as exc:
            raise LLMError("Anthropic rate limit reached; retry later") from exc
        except a.APIStatusError as exc:
            raise LLMError(redact(f"Anthropic API error {exc.status_code}: {exc.message}")) from exc
        except a.APIConnectionError as exc:
            raise LLMError(redact(f"Anthropic connection error: {exc}")) from exc
        if resp.stop_reason == "refusal":
            raise LLMInvalidResponse("model declined to validate this prediction (refusal)")
        if resp.stop_reason == "max_tokens":
            raise LLMInvalidResponse("validation response was truncated (max_tokens)")
        return _parse(next((b.text for b in resp.content if b.type == "text"), ""), "Anthropic")


class OllamaProvider(LLMProvider):
    """Local models via Ollama's /api/chat with a JSON-schema `format`. Labelled 'ollama' - never
    presented as OpenAI or Claude validation."""
    name = "ollama"

    def __init__(self, model: str | None = None):
        s = get_settings()
        self.base_url = s.ollama_base_url.rstrip("/")
        self.model = model or s.ollama_model

    def complete_json(self, system, user, schema):
        try:
            resp = request_with_retry("POST", f"{self.base_url}/api/chat", timeout_s=300, max_retries=1, json={
                "model": self.model, "stream": False, "format": schema,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        except Exception as exc:
            raise LLMError(redact(f"Ollama request failed: {exc}")) from exc
        return _parse(resp.json().get("message", {}).get("content", ""), "Ollama")


class UnavailableLLM(LLMProvider):
    """Stands in when the selected provider is not configured / cannot be constructed. Every call
    fails with an explicit reason, so the stored result says "not performed" - it never pretends
    that a review happened, and it never falls back to another provider."""
    configured = False

    def __init__(self, name: str, model: str, reason: str):
        self.name, self.model, self.reason = name, model or "(not configured)", redact(reason)

    def complete_json(self, system, user, schema):
        raise LLMNotConfigured(self.reason)


def get_llm(name: str | None = None) -> LLMProvider | None:
    name = name or get_settings().llm_provider
    if name == "none":
        return None
    if name == "openai":
        try:
            return OpenAIProvider()
        except LLMNotConfigured as exc:
            return UnavailableLLM("openai", get_settings().openai_model, str(exc))
        except Exception as exc:   # SDK missing, bad CA bundle path, ...
            return UnavailableLLM("openai", get_settings().openai_model,
                                  f"LLM validation not performed: OpenAI client could not be created ({exc})")
    if name == "anthropic":
        try:
            return AnthropicProvider()
        except Exception as exc:   # missing credentials, SDK not installed, ...
            return UnavailableLLM("anthropic", get_settings().anthropic_model, str(exc)[:300])
    if name == "ollama":
        return OllamaProvider()
    raise ValueError(f"unknown LLM_PROVIDER {name!r}")


def provider_status() -> dict:
    """Safe-to-display LLM configuration (never includes a key)."""
    s = get_settings()
    out = {"llm_provider": s.llm_provider}
    if s.llm_provider == "openai":
        configured = bool(s.openai_api_key) and bool(s.openai_model)
        out.update(llm_model=s.openai_model or None, llm_configured=configured,
                   llm_status=None if configured else (OPENAI_NOT_CONFIGURED if not s.openai_api_key
                                                       else OPENAI_MODEL_NOT_CONFIGURED))
    elif s.llm_provider == "anthropic":
        out.update(llm_model=s.anthropic_model, llm_configured=None, llm_status=None)
    elif s.llm_provider == "ollama":
        out.update(llm_model=s.ollama_model, llm_configured=None, llm_status=None)
    else:
        out.update(llm_model=None, llm_configured=False, llm_status="LLM validation disabled (LLM_PROVIDER=none)")
    return out
