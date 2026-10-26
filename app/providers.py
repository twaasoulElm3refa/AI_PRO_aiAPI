from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from sqlalchemy import select

from app.errors import ProviderError, ProviderOutputError
from app.image_prompt_output import strip_hidden_reasoning
from app.prompt_contracts import apply_common_system_guardrails
from app.request_context import get_request_id
from app.billing.context import get_current_db, mark_current_provider_call_started
from app.billing.errors import BillingConfigurationError
from app.models import AIModel
from app.schemas import ChatMessage
from app.settings import get_settings
from app.tasks import MODEL_ROUTES


logger = logging.getLogger("app.openrouter")

# Keep retries conservative. A failed generation can still consume provider capacity,
# and deterministic schema/response_format failures should not be hammered repeatedly.
_MAX_ATTEMPTS = 2
_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass
class ProviderResult:
    content: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    raw_usage: Optional[dict] = None
    trace_id: Optional[str] = None
    generation_id: Optional[str] = None
    requested_model: Optional[str] = None
    returned_model: Optional[str] = None
    provider_name: Optional[str] = None
    finish_reason: Optional[str] = None
    router_metadata: Optional[dict] = None

    def trace_metadata(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "generation_id": self.generation_id,
            "requested_model": self.requested_model,
            "returned_model": self.returned_model,
            "provider": self.provider_name,
            "finish_reason": self.finish_reason,
            "router_metadata": self.router_metadata,
        }


def _clean_ascii_header(value: str) -> str:
    cleaned = str(value or "").encode("ascii", "ignore").decode("ascii").strip()
    return cleaned or "AI Tools API"


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def _preview(value: Any, limit: int) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = repr(value)

    text = text.replace("\x00", "")
    return text if len(text) <= limit else text[:limit] + "..."


def _selected_provider(data: dict[str, Any]) -> str | None:
    provider = data.get("provider")
    if isinstance(provider, str) and provider.strip():
        return provider.strip()

    metadata = data.get("openrouter_metadata")
    if not isinstance(metadata, dict):
        return None

    attempts = metadata.get("attempts")
    if isinstance(attempts, list):
        for attempt in reversed(attempts):
            if isinstance(attempt, dict) and attempt.get("provider"):
                return str(attempt["provider"])

    endpoints = metadata.get("endpoints")
    if isinstance(endpoints, dict):
        available = endpoints.get("available")
        if isinstance(available, list):
            for endpoint in available:
                if (
                    isinstance(endpoint, dict)
                    and endpoint.get("selected")
                    and endpoint.get("provider")
                ):
                    return str(endpoint["provider"])

    return None


def _coerce_status_code(value: Any, default: int | None = None) -> int | None:
    try:
        code = int(value)
    except (TypeError, ValueError):
        return default

    return code if 100 <= code <= 599 else default


def _first_choice(data: dict[str, Any]) -> dict[str, Any] | None:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    choice = choices[0]
    return choice if isinstance(choice, dict) else None


def _embedded_error(
    data: dict[str, Any],
    choice: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    OpenRouter/provider failures are not always represented by a non-2xx HTTP status.

    Some upstream failures arrive as HTTP 200 with either:
      - a top-level ``error`` object, or
      - ``choices[0].error`` and ``finish_reason == 'error'``.

    The Alibaba/Qwen structured-output failure reproduced by this project uses the
    second form, so we must detect it before consuming ``message.content``.
    """
    top_level = data.get("error")
    if isinstance(top_level, dict):
        return top_level

    if choice is None:
        choice = _first_choice(data)

    if isinstance(choice, dict):
        choice_error = choice.get("error")
        if isinstance(choice_error, dict):
            return choice_error

        message = choice.get("message")
        if isinstance(message, dict):
            message_error = message.get("error")
            if isinstance(message_error, dict):
                return message_error

    return None


def _error_type(error: dict[str, Any] | None) -> str | None:
    if not isinstance(error, dict):
        return None

    metadata = error.get("metadata")
    if isinstance(metadata, dict) and metadata.get("error_type"):
        return str(metadata["error_type"])

    return None


def _error_message(error: dict[str, Any] | None) -> str:
    if not isinstance(error, dict):
        return "Unknown upstream provider error"

    message = error.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()

    return _preview(error, 1500)


def _is_retryable_error(error: dict[str, Any] | None) -> bool:
    if not isinstance(error, dict):
        return False

    status_code = _coerce_status_code(error.get("code"))
    if status_code in _RETRYABLE_STATUS_CODES:
        return True

    error_type = (_error_type(error) or "").lower()
    return error_type in {
        "provider_unavailable",
        "rate_limit_exceeded",
        "server_error",
        "timeout",
    }




def _is_structured_output_failure(error: dict[str, Any] | None) -> bool:
    """Return True when the upstream failure is specifically tied to response_format."""
    if not isinstance(error, dict):
        return False

    text = _error_message(error).lower()
    return any(
        marker in text
        for marker in (
            "response_format",
            "json response",
            "structured output",
            "structured-output",
            "requested parameters",
        )
    )


def _without_structured_output(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Build a compatibility retry payload when an upstream provider's structured
    output implementation fails. The tool prompts still request JSON, and the
    project's local validators remain responsible for enforcing the object shape.
    """
    fallback = dict(payload)
    fallback.pop("response_format", None)

    plugins = fallback.get("plugins")
    if isinstance(plugins, list):
        remaining_plugins = [
            plugin
            for plugin in plugins
            if not (isinstance(plugin, dict) and plugin.get("id") == "response-healing")
        ]
        if remaining_plugins:
            fallback["plugins"] = remaining_plugins
        else:
            fallback.pop("plugins", None)

    provider_options = fallback.get("provider")
    if isinstance(provider_options, dict):
        provider_options = dict(provider_options)
        provider_options.pop("require_parameters", None)
        if provider_options:
            fallback["provider"] = provider_options
        else:
            fallback.pop("provider", None)

    # Ask providers not to expose a separate reasoning stream on the compatibility
    # retry. Existing JSON-only system prompts and downstream validators still apply.
    fallback.setdefault("reasoning", {"exclude": True})
    return fallback

def _retry_delay_seconds(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), 5.0)
        except ValueError:
            pass

    # attempt is 1-based. Keep the one retry short and predictable.
    return min(0.75 * attempt, 2.0)


def build_openrouter_headers(*, include_content_type: bool = True) -> dict[str, str]:
    """Build common OpenRouter headers for chat, STT, and TTS requests."""
    settings = get_settings()
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}".strip(),
        "HTTP-Referer": str(settings.SITE_URL),
        "X-Title": _clean_ascii_header(settings.APP_NAME),
    }

    if include_content_type:
        headers["Content-Type"] = "application/json"

    request_id = get_request_id()
    if request_id and settings.OPENROUTER_SESSION_STICKINESS_ENABLED:
        # Sticky routing is useful only when explicitly requested. During provider
        # debugging it can pin retries to the same unhealthy endpoint, so it remains
        # controlled by the existing setting.
        headers["X-Session-ID"] = request_id[:256]

    if settings.OPENROUTER_ROUTER_METADATA_ENABLED:
        headers["X-OpenRouter-Metadata"] = "enabled"

    return headers


def _log_request(
    *,
    trace_id: str,
    request_id: str | None,
    model_key: str,
    model_name: str,
    payload: dict[str, Any],
) -> None:
    settings = get_settings()
    if not settings.OPENROUTER_TRACE_ENABLED:
        return

    response_format = payload.get("response_format")
    schema_name = None
    if isinstance(response_format, dict):
        json_schema = response_format.get("json_schema")
        if isinstance(json_schema, dict):
            schema_name = json_schema.get("name")

    provider_options = payload.get("provider")
    require_parameters = (
        bool(provider_options.get("require_parameters"))
        if isinstance(provider_options, dict)
        else False
    )

    response_healing = any(
        isinstance(plugin, dict) and plugin.get("id") == "response-healing"
        for plugin in (payload.get("plugins") or [])
    )

    logger.info(
        "OPENROUTER request trace_id=%s request_id=%s model_key=%s model=%s "
        "temperature=%s max_completion_tokens=%s response_format=%s schema=%s "
        "require_parameters=%s response_healing=%s web_search=%s",
        trace_id,
        request_id,
        model_key,
        model_name,
        payload.get("temperature"),
        payload.get("max_completion_tokens"),
        response_format.get("type") if isinstance(response_format, dict) else response_format,
        schema_name,
        require_parameters,
        response_healing,
        bool(payload.get("tools")),
    )

    if settings.OPENROUTER_TRACE_CONTENT:
        logger.info(
            "OPENROUTER request-content trace_id=%s payload=%s",
            trace_id,
            _preview(payload, settings.OPENROUTER_TRACE_MAX_CHARS),
        )


def _log_response(
    *,
    trace_id: str,
    request_id: str | None,
    attempt: int,
    status_code: int,
    elapsed_seconds: float,
    generation_id: str | None,
    data: dict[str, Any] | None,
    raw_text: str,
) -> None:
    settings = get_settings()
    if not settings.OPENROUTER_TRACE_ENABLED:
        return

    data = data or {}
    choice = _first_choice(data) or {}
    message = choice.get("message") if isinstance(choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    embedded_error = _embedded_error(data, choice)

    logger.info(
        "OPENROUTER response trace_id=%s request_id=%s attempt=%s status=%s "
        "elapsed=%.3fs generation_id=%s returned_model=%s provider=%s "
        "finish_reason=%s provider_error_code=%s provider_error_type=%s",
        trace_id,
        request_id,
        attempt,
        status_code,
        elapsed_seconds,
        generation_id,
        data.get("model") if isinstance(data, dict) else None,
        _selected_provider(data) if isinstance(data, dict) else None,
        choice.get("finish_reason") if isinstance(choice, dict) else None,
        embedded_error.get("code") if isinstance(embedded_error, dict) else None,
        _error_type(embedded_error),
    )

    if isinstance(data, dict) and data.get("openrouter_metadata"):
        logger.info(
            "OPENROUTER router-metadata trace_id=%s metadata=%s",
            trace_id,
            _preview(
                data.get("openrouter_metadata"),
                settings.OPENROUTER_TRACE_MAX_CHARS,
            ),
        )

    if isinstance(embedded_error, dict):
        logger.warning(
            "OPENROUTER provider-error trace_id=%s attempt=%s code=%s "
            "error_type=%s message=%s",
            trace_id,
            attempt,
            embedded_error.get("code"),
            _error_type(embedded_error),
            _preview(_error_message(embedded_error), settings.OPENROUTER_TRACE_MAX_CHARS),
        )

    if settings.OPENROUTER_TRACE_CONTENT:
        logger.info(
            "OPENROUTER response-content trace_id=%s content=%s raw_response=%s",
            trace_id,
            _preview(content, settings.OPENROUTER_TRACE_MAX_CHARS),
            _preview(raw_text, settings.OPENROUTER_TRACE_MAX_CHARS),
        )


def _raise_embedded_provider_error(
    *,
    data: dict[str, Any],
    choice: dict[str, Any] | None,
    http_status: int,
    generation_id: str | None,
    trace_id: str,
) -> None:
    """Raise ProviderError for provider failures hidden inside HTTP 2xx responses."""
    error = _embedded_error(data, choice)
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None

    # A normal completion has neither an error object nor finish_reason='error'.
    if error is None and finish_reason != "error":
        return

    provider_name = _selected_provider(data) or "unknown"
    returned_model = str(data.get("model") or "unknown")

    if error is not None:
        embedded_status = _coerce_status_code(error.get("code"), default=502)
        error_type = _error_type(error) or "unknown"
        message = _error_message(error)
    else:
        embedded_status = 502
        error_type = "unknown"
        message = "The upstream provider ended generation with finish_reason='error'."

    raise ProviderError(
        "OpenRouter generation failed: "
        f"provider={provider_name}, "
        f"model={returned_model}, "
        f"code={embedded_status}, "
        f"error_type={error_type}, "
        f"finish_reason={finish_reason or 'unknown'}, "
        f"trace_id={trace_id}, "
        f"message={message}",
        provider="openrouter",
        status_code=embedded_status or http_status or 502,
        generation_id=generation_id,
    )


async def send_messages_with_model(
    model_key: str,
    messages: List[ChatMessage],
    temperature_override: Optional[float] = None,
    max_tokens_override: Optional[int] = None,
    enable_web_search: bool = False,
    web_search_max_results: Optional[int] = None,
    web_search_max_total_results: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
    exclude_reasoning: bool = False,
    reasoning_effort: Optional[str] = None,
    model_override: Optional[str] = None,
) -> ProviderResult:
    """
    Send a chat-completions request through OpenRouter.

    Compatibility guarantees for the rest of the project:
      - same function name and arguments
      - same ProviderResult shape
      - legacy model roles resolve from ai_models instead of environment values
      - same response_format / response-healing / web-search behavior

    Important error handling:
      OpenRouter can return HTTP 200 while ``choices[0]`` contains an upstream
      provider error. Such responses are treated as provider failures here and are
      never passed to downstream JSON parsers as if they were successful content.
    """
    settings = get_settings()

    if model_key not in MODEL_ROUTES:
        raise ValueError(f"Unknown model_key: {model_key}")

    route = MODEL_ROUTES[model_key]
    provider = route["provider"]
    if provider != "openrouter":
        raise ValueError(f"Unsupported provider: {provider}")

    model_name = str(model_override or "").strip()
    if not model_name:
        context_db = get_current_db()
        if context_db is not None:
            catalog_model = context_db.execute(select(AIModel).where(
                AIModel.provider == "openrouter",
                AIModel.tool_key == model_key,
                AIModel.operation == "text_generation",
                AIModel.is_active.is_(True),
                AIModel.is_available.is_(True),
                AIModel.deleted_at.is_(None),
            ).order_by(
                AIModel.is_recommended.desc(), AIModel.sort_order, AIModel.id
            )).scalars().first()
            if catalog_model is not None:
                model_name = str(catalog_model.provider_model_id or "").strip()
    if not model_name:
        raise BillingConfigurationError(
            f"No active ai_models row configured for legacy model key {model_key}"
        )

    if not settings.OPENROUTER_API_KEY:
        raise ProviderError(
            "OPENROUTER_API_KEY is missing",
            provider="openrouter",
        )

    temperature = (
        float(temperature_override)
        if temperature_override is not None
        else float(route.get("temperature", settings.DEFAULT_TEMPERATURE))
    )
    max_tokens = (
        int(max_tokens_override)
        if max_tokens_override is not None
        else int(route.get("max_tokens", settings.DEFAULT_MAX_TOKENS))
    )

    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            msg.model_dump()
            for msg in apply_common_system_guardrails(messages)
        ],
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
    }

    if response_format is not None:
        payload["response_format"] = response_format

        # Keep this conditional. Making require_parameters=True unconditionally can
        # cause a 404 when no endpoint supports every requested parameter.
        if settings.OPENROUTER_REQUIRE_PARAMETERS:
            payload["provider"] = {"require_parameters": True}

        if settings.OPENROUTER_RESPONSE_HEALING_ENABLED:
            payload["plugins"] = [{"id": "response-healing"}]

    if exclude_reasoning or reasoning_effort is not None:
        reasoning: Dict[str, Any] = {"exclude": bool(exclude_reasoning)}
        if reasoning_effort is not None:
            normalized_effort = str(reasoning_effort).strip().lower()
            allowed_efforts = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
            if normalized_effort not in allowed_efforts:
                raise ValueError(f"Unsupported reasoning effort: {reasoning_effort}")
            reasoning["effort"] = normalized_effort
        payload["reasoning"] = reasoning

    if enable_web_search:
        max_results = _clamp(
            web_search_max_results or settings.WEB_SEARCH_DEFAULT_MAX_RESULTS,
            1,
            settings.WEB_SEARCH_HARD_MAX_RESULTS,
        )
        max_total_results = _clamp(
            web_search_max_total_results or settings.WEB_SEARCH_DEFAULT_MAX_TOTAL_RESULTS,
            max_results,
            settings.WEB_SEARCH_HARD_MAX_TOTAL_RESULTS,
        )
        payload["tools"] = [
            {
                "type": "openrouter:web_search",
                "parameters": {
                    "max_results": max_results,
                    "max_total_results": max_total_results,
                    "search_context_size": "medium",
                },
            }
        ]

    url = f"{settings.OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
    headers = build_openrouter_headers()
    trace_id = uuid.uuid4().hex[:12]
    request_id = get_request_id()

    _log_request(
        trace_id=trace_id,
        request_id=request_id,
        model_key=model_key,
        model_name=model_name,
        payload=payload,
    )

    last_response: httpx.Response | None = None
    last_data: dict[str, Any] | None = None
    last_generation_id: str | None = None

    timeout = httpx.Timeout(float(settings.REQUEST_TIMEOUT_SECONDS))

    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            started = time.monotonic()

            try:
                mark_current_provider_call_started()
                response = await client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                elapsed = time.monotonic() - started

                if settings.OPENROUTER_TRACE_ENABLED:
                    logger.exception(
                        "OPENROUTER transport-error trace_id=%s request_id=%s "
                        "attempt=%s elapsed=%.3fs",
                        trace_id,
                        request_id,
                        attempt,
                        elapsed,
                    )

                if attempt < _MAX_ATTEMPTS:
                    await asyncio.sleep(0.75 * attempt)
                    continue

                raise ProviderError(
                    f"OpenRouter transport error after {attempt} attempt(s): {exc}",
                    provider="openrouter",
                ) from exc

            elapsed = time.monotonic() - started
            generation_id = response.headers.get("X-Generation-Id")
            raw_text = response.text

            data: dict[str, Any] | None = None
            try:
                parsed = response.json()
                if isinstance(parsed, dict):
                    data = parsed
            except ValueError:
                data = None

            if generation_id is None and isinstance(data, dict):
                response_id = data.get("id")
                generation_id = str(response_id) if response_id else None

            _log_response(
                trace_id=trace_id,
                request_id=request_id,
                attempt=attempt,
                status_code=response.status_code,
                elapsed_seconds=elapsed,
                generation_id=generation_id,
                data=data,
                raw_text=raw_text,
            )

            last_response = response
            last_data = data
            last_generation_id = generation_id

            # Normal HTTP failure (4xx/5xx). Some routing/provider errors are
            # specifically caused by structured-output parameters. In that case,
            # make one compatibility retry without response_format; downstream
            # validators still require the expected JSON object.
            if not response.is_success:
                http_error = _embedded_error(data) if isinstance(data, dict) else None
                if (
                    attempt < _MAX_ATTEMPTS
                    and "response_format" in payload
                    and _is_structured_output_failure(http_error)
                ):
                    if settings.OPENROUTER_TRACE_ENABLED:
                        logger.warning(
                            "OPENROUTER compatibility-retry trace_id=%s attempt=%s "
                            "reason=structured_output_http_error status=%s",
                            trace_id,
                            attempt,
                            response.status_code,
                        )
                    payload = _without_structured_output(payload)
                    await asyncio.sleep(0.25)
                    continue

                if (
                    response.status_code in _RETRYABLE_STATUS_CODES
                    and attempt < _MAX_ATTEMPTS
                ):
                    if settings.OPENROUTER_TRACE_ENABLED:
                        logger.warning(
                            "OPENROUTER retry trace_id=%s attempt=%s reason=http_%s",
                            trace_id,
                            attempt,
                            response.status_code,
                        )
                    await asyncio.sleep(_retry_delay_seconds(response, attempt))
                    continue

                detail = _preview(
                    data if data is not None else raw_text,
                    settings.OPENROUTER_TRACE_MAX_CHARS,
                )
                raise ProviderError(
                    f"OpenRouter HTTP {response.status_code}: {detail}",
                    provider="openrouter",
                    status_code=response.status_code,
                    generation_id=generation_id,
                )

            if data is None:
                raise ProviderOutputError(
                    "OpenRouter returned a non-JSON chat response: "
                    f"{_preview(raw_text, 1000)}",
                    provider="openrouter",
                    status_code=response.status_code,
                    generation_id=generation_id,
                )

            choice = _first_choice(data)
            embedded_error = _embedded_error(data, choice)
            finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None

            # Critical fix for your Alibaba/Qwen case: HTTP can be 200 while the
            # actual generation failed with choices[0].error and content=-1e308.
            if embedded_error is not None or finish_reason == "error":
                # Your reproduced Alibaba/Qwen failure lands here: HTTP 200,
                # finish_reason='error', choices[0].error.code=502, and content
                # containing a huge negative number. Do not pass that content to
                # the JSON parser. If the error is specifically response_format
                # related, retry once without provider-side structured output.
                if (
                    attempt < _MAX_ATTEMPTS
                    and "response_format" in payload
                    and _is_structured_output_failure(embedded_error)
                ):
                    if settings.OPENROUTER_TRACE_ENABLED:
                        logger.warning(
                            "OPENROUTER compatibility-retry trace_id=%s attempt=%s "
                            "reason=structured_output_provider_error code=%s error_type=%s",
                            trace_id,
                            attempt,
                            embedded_error.get("code")
                            if isinstance(embedded_error, dict)
                            else None,
                            _error_type(embedded_error),
                        )
                    payload = _without_structured_output(payload)
                    await asyncio.sleep(0.25)
                    continue

                retryable = _is_retryable_error(embedded_error)
                if retryable and attempt < _MAX_ATTEMPTS:
                    if settings.OPENROUTER_TRACE_ENABLED:
                        logger.warning(
                            "OPENROUTER retry trace_id=%s attempt=%s reason=provider_error "
                            "code=%s error_type=%s",
                            trace_id,
                            attempt,
                            embedded_error.get("code")
                            if isinstance(embedded_error, dict)
                            else None,
                            _error_type(embedded_error),
                        )
                    await asyncio.sleep(_retry_delay_seconds(response, attempt))
                    continue

                _raise_embedded_provider_error(
                    data=data,
                    choice=choice,
                    http_status=response.status_code,
                    generation_id=generation_id,
                    trace_id=trace_id,
                )

            # We have a successful HTTP response and no embedded provider failure.
            break

    # Defensive checks. The loop above should always return a response or raise.
    if last_response is None:
        raise ProviderError(
            "OpenRouter request ended without receiving a response",
            provider="openrouter",
        )

    if last_data is None:
        raise ProviderOutputError(
            "OpenRouter returned no usable JSON response",
            provider="openrouter",
            status_code=last_response.status_code,
            generation_id=last_generation_id,
        )

    data = last_data
    generation_id = last_generation_id

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ProviderOutputError(
            f"OpenRouter response contained no usable choices: {_preview(data, 1500)}",
            provider="openrouter",
            status_code=last_response.status_code,
            generation_id=generation_id,
        )

    choice = choices[0]

    # Defensive second check in case response parsing changes upstream.
    _raise_embedded_provider_error(
        data=data,
        choice=choice,
        http_status=last_response.status_code,
        generation_id=generation_id,
        trace_id=trace_id,
    )

    message = choice.get("message")
    if not isinstance(message, dict):
        raise ProviderOutputError(
            f"OpenRouter response contained no assistant message: {_preview(choice, 1500)}",
            provider="openrouter",
            status_code=last_response.status_code,
            generation_id=generation_id,
        )

    raw_content = message.get("content")
    content = strip_hidden_reasoning(raw_content)
    if not content:
        raise ProviderOutputError(
            "OpenRouter returned an empty assistant response",
            provider="openrouter",
            status_code=last_response.status_code,
            generation_id=generation_id,
        )

    usage = data.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}

    input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
    output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return ProviderResult(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        raw_usage=usage,
        trace_id=trace_id,
        generation_id=generation_id,
        requested_model=model_name,
        returned_model=str(data.get("model") or "") or None,
        provider_name=_selected_provider(data),
        finish_reason=str(choice.get("finish_reason") or "") or None,
        router_metadata=(
            data.get("openrouter_metadata")
            if isinstance(data.get("openrouter_metadata"), dict)
            else None
        ),
    )
