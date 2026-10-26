from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import operator
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy.orm import Session

from app.crud import get_ai_model
from app.models import AIAgentArtifact, AIAgentFile, AIAgentRun, AIAgentStep, AIAgentToolCall
from app.providers import send_messages_with_model
from app.billing import (
    BillingAuthorization, estimate_model_reservation, mark_charge_started,
    mark_reconciliation_required,
)
from app.billing.errors import BillingConfigurationError
from app.schemas import ChatMessage
from app.settings import get_settings


@dataclass(frozen=True)
class AgentToolContext:
    db: Session
    run: AIAgentRun
    step: AIAgentStep
    state: dict[str, Any]
    plan_step_index: int


ToolHandler = Callable[[AgentToolContext, dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class AgentToolSpec:
    key: str
    description: str
    input_schema: dict[str, Any]
    category: str = "custom"
    produces_artifact: bool = False
    billing_mode: str = "unconfigured"

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "description": self.description,
            "input_schema": self.input_schema,
            "category": self.category,
            "produces_artifact": self.produces_artifact,
            "billing_mode": self.billing_mode,
        }


class AgentToolRegistry:
    """Server-side tool adapters. Agent allowlists remain the execution boundary."""

    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {}
        self._specs: dict[str, AgentToolSpec] = {}

    def register(
        self,
        key: str,
        handler: ToolHandler,
        *,
        description: str,
        input_schema: dict[str, Any] | None = None,
        category: str = "custom",
        produces_artifact: bool = False,
        billing_mode: str = "unconfigured",
    ) -> None:
        normalized = str(key or "").strip()
        if not normalized or normalized in self._handlers:
            raise ValueError(f"Invalid or duplicate agent tool: {normalized}")
        if not callable(handler):
            raise ValueError(f"Agent tool handler is not callable: {normalized}")
        self._handlers[normalized] = handler
        self._specs[normalized] = AgentToolSpec(
            key=normalized,
            description=str(description or "").strip(),
            input_schema=input_schema or {"type": "object", "additionalProperties": False},
            category=category,
            produces_artifact=produces_artifact,
            billing_mode=billing_mode,
        )

    def has(self, key: str) -> bool:
        return key in self._handlers

    def specs(self, allowed: list[str] | set[str] | None = None) -> list[dict[str, Any]]:
        allowed_set = set(allowed) if allowed is not None else None
        return [
            spec.as_dict() for key, spec in self._specs.items()
            if allowed_set is None or key in allowed_set
        ]

    def validate_arguments(self, key: str, arguments: dict[str, Any]) -> list[str]:
        """Validate the common JSON-schema subset used by planner tool arguments."""
        spec = self._specs.get(key)
        if not spec:
            return [f"unknown tool: {key}"]
        schema = spec.input_schema or {}
        properties = schema.get("properties") or {}
        errors = []
        for name in schema.get("required") or []:
            if name not in arguments or arguments.get(name) in (None, ""):
                errors.append(f"missing required argument: {name}")
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(arguments) - set(properties))
            if unknown:
                errors.append("unknown arguments: " + ", ".join(unknown))
        expected_types = {
            "string": str, "boolean": bool, "object": dict,
            "array": list, "integer": int, "number": (int, float),
        }
        for name, value in arguments.items():
            definition = properties.get(name)
            if not isinstance(definition, dict) or value is None:
                continue
            expected = expected_types.get(definition.get("type"))
            is_boolean_number = isinstance(value, bool) and definition.get("type") in {
                "integer", "number"
            }
            if expected and (not isinstance(value, expected) or is_boolean_number):
                errors.append(f"argument {name} must be {definition.get('type')}")
        return errors

    async def execute(
        self, context: AgentToolContext, tool_key: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        handler = self._handlers.get(tool_key)
        if not handler:
            raise ValueError(f"Agent tool has no registered adapter: {tool_key}")
        spec = self._specs[tool_key]
        if (
            get_settings().BILLING_ENFORCEMENT_ENABLED
            and spec.billing_mode == "unconfigured"
        ):
            raise BillingConfigurationError(
                f"Agent tool {tool_key} has no pre-execution billing policy"
            )
        return await handler(context, payload)


async def _document_extract(context: AgentToolContext, payload: dict[str, Any]) -> dict[str, Any]:
    items = context.state.get("file_items") or []
    extracted = context.state.get("extracted_context") or ""
    if not items:
        raise ValueError("document_extract requires at least one uploaded document")
    if not str(extracted).strip():
        raise ValueError(
            "No text was extracted from the uploaded documents; OCR may be required"
        )
    return {
        "execution": "completed",
        "file_count": len(items),
        "extracted_chars": len(extracted),
        "sources": [item.get("filename") for item in items],
    }


async def _delegated_text(context: AgentToolContext, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "execution": "delegated_to_generator",
        "instruction": str(payload.get("instruction") or ""),
        "reason": "The primary selected text model executes this step in the generator node.",
    }


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate_number(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_number(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_number(node.left)
        right = _evaluate_number(node.right)
        if isinstance(node.op, ast.Pow) and abs(float(right)) > 100:
            raise ValueError("calculator exponent is too large")
        value = _BINARY_OPERATORS[type(node.op)](left, right)
        if abs(float(value)) > 1e100:
            raise ValueError("calculator result is too large")
        return value
    raise ValueError("calculator accepts numbers, parentheses, and arithmetic operators only")


async def _calculator(context: AgentToolContext, payload: dict[str, Any]) -> dict[str, Any]:
    expression = str(payload.get("expression") or payload.get("instruction") or "").strip()
    if not expression or len(expression) > 500:
        raise ValueError("calculator expression must contain 1 to 500 characters")
    try:
        parsed = ast.parse(expression, mode="eval")
        value = _evaluate_number(parsed.body)
    except (SyntaxError, TypeError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"invalid calculator expression: {exc}") from exc
    return {"execution": "completed", "expression": expression, "result": value}


async def _current_datetime(context: AgentToolContext, payload: dict[str, Any]) -> dict[str, Any]:
    timezone_name = str(payload.get("timezone") or "UTC").strip()
    try:
        zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"unknown IANA timezone: {timezone_name}") from exc
    value = datetime.now(timezone.utc).astimezone(zone)
    return {
        "execution": "completed",
        "timezone": timezone_name,
        "iso8601": value.isoformat(),
        "date": value.date().isoformat(),
        "time": value.strftime("%H:%M:%S"),
    }


async def _web_search(context: AgentToolContext, payload: dict[str, Any]) -> dict[str, Any]:
    query = str(payload.get("query") or payload.get("instruction") or "").strip()
    if not query:
        raise ValueError("web_search query is required")
    selected = (context.state.get("selected_tool_model_ids") or {}).get("web_search")
    model_id = int(selected or context.state.get("selected_model_id") or 0)
    model = get_ai_model(context.db, model_id)
    if model.provider.lower() != "openrouter":
        raise ValueError("web_search requires an OpenRouter model")
    settings = get_settings()
    if settings.BILLING_ENFORCEMENT_ENABLED:
        projected = estimate_model_reservation(
            model,
            input_text=(
                "Search the web for the requested information. Return a compact factual "
                "research note with source URLs.\n" + query
            ),
            max_output_tokens=2500,
            web_search_cost_usd=(
                settings.WEB_SEARCH_COST_PER_1000_RESULTS
                * settings.WEB_SEARCH_DEFAULT_MAX_TOTAL_RESULTS
                / 1000
            ),
        )
        budget = (context.state.get("parameters") or {}).get("max_budget_usd")
        if budget is None or (
            float(context.state.get("total_cost") or 0.0) + float(projected)
            > float(budget)
        ):
            raise ValueError("Agent budget is insufficient for the web search step")
    authorization = BillingAuthorization(
        enabled=bool(context.state.get("billing_reservation_uuid")),
        reservation_uuid=context.state.get("billing_reservation_uuid"),
    )
    provider_started = mark_charge_started(context.db, authorization)
    try:
        result = await send_messages_with_model(
            model_key="writer_pro",
            model_override=model.provider_model_id,
            messages=[
                ChatMessage(
                    role="system",
                    content=(
                        "Search the web for the requested information. Return a compact factual "
                        "research note with source URLs. Do not add unsupported claims."
                    ),
                ),
                ChatMessage(role="user", content=query),
            ],
            temperature_override=0.1,
            max_tokens_override=2500,
            enable_web_search=True,
            exclude_reasoning=True,
            reasoning_effort="low",
        )
    except Exception as exc:
        if provider_started:
            mark_reconciliation_required(
                context.db, authorization,
                reason=f"agent_web_search_result_unknown: {type(exc).__name__}",
            )
        raise
    raw_cost = (result.raw_usage or {}).get("cost")
    return {
        "execution": "completed",
        "query": query,
        "content": result.content,
        "model": result.returned_model or model.provider_model_id,
        "usage": {
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.total_tokens,
        },
        "cost": {"total_cost": float(raw_cost), "currency": "USD"}
        if raw_cost is not None else None,
    }


async def _external_adapter_required(
    context: AgentToolContext, payload: dict[str, Any]
) -> dict[str, Any]:
    tool_key = str(payload.get("tool") or "")
    selected = (context.state.get("selected_tool_model_ids") or {}).get(tool_key)
    requires_artifact = bool(payload.get("requires_artifact"))
    if not selected and not requires_artifact:
        return {
            "execution": "prompt_only",
            "selected_model_id": None,
            "reason": "The plan requests creative instructions only, not a generated file.",
        }
    return {
        "execution": "not_executed",
        "selected_model_id": selected,
        "reason": (
            "A downloadable artifact is required, but no secondary model was selected."
            if not selected else
            "A model was selected, but this external tool adapter is not connected yet."
        ),
    }


def build_default_registry() -> AgentToolRegistry:
    registry = AgentToolRegistry()
    no_args = {"type": "object", "additionalProperties": False}
    instruction = {
        "type": "object",
        "properties": {"instruction": {"type": "string"}},
        "additionalProperties": False,
    }
    registry.register(
        "document_extract", _document_extract,
        description="Read text already extracted from the files uploaded with this run.",
        input_schema=no_args, category="documents", billing_mode="free",
    )
    for key, description in {
        "general_chat": "Reason, write, summarize, or answer with the primary language model.",
        "general_code": "Generate, review, explain, debug, or refactor source code.",
        "general_translation": "Translate text while preserving meaning and formatting.",
    }.items():
        registry.register(
            key, _delegated_text, description=description,
            input_schema=instruction, category="model", billing_mode="delegated",
        )
    registry.register(
        "calculator", _calculator,
        description="Evaluate a deterministic arithmetic expression without using the model.",
        input_schema={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"], "additionalProperties": False,
        }, category="utility",
        billing_mode="free",
    )
    registry.register(
        "current_datetime", _current_datetime,
        description="Get the current date and time in an IANA timezone such as Africa/Cairo.",
        input_schema={
            "type": "object",
            "properties": {"timezone": {"type": "string", "default": "UTC"}},
            "additionalProperties": False,
        }, category="utility",
        billing_mode="free",
    )
    registry.register(
        "web_search", _web_search,
        description="Search the live web and return a sourced research note.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"], "additionalProperties": False,
        }, category="research",
        billing_mode="guarded",
    )
    for key, description, category in (
        ("general_media", "Generate or edit images/video using the configured media adapter.", "media"),
        ("general_audio", "Transcribe speech or synthesize audio using the configured audio adapter.", "audio"),
        ("trends", "Analyze or create trend-related media using the configured adapter.", "research"),
    ):
        registry.register(
            key, _external_adapter_required, description=description,
            input_schema=instruction, category=category, produces_artifact=True,
            billing_mode="not_executed",
        )
    _load_external_tool_modules(registry)
    return registry


def _load_external_tool_modules(registry: AgentToolRegistry) -> None:
    """Load trusted, server-configured adapters without coupling them to site tools."""
    configured = str(get_settings().AGENT_TOOL_MODULES or "")
    for module_name in [value.strip() for value in configured.split(",") if value.strip()]:
        module = importlib.import_module(module_name)
        hook = getattr(module, "register_agent_tools", None)
        if not callable(hook):
            raise ValueError(
                f"Agent tool module {module_name} must export register_agent_tools(registry)"
            )
        result = hook(registry)
        if inspect.isawaitable(result):
            raise ValueError(
                f"Agent tool module {module_name} registration must be synchronous"
            )


def create_tool_call(
    context: AgentToolContext, tool_key: str, payload: dict[str, Any]
) -> AIAgentToolCall:
    selected = (context.state.get("selected_tool_model_ids") or {}).get(tool_key)
    call = AIAgentToolCall(
        call_uuid=str(uuid.uuid4()), run_id=context.run.id, step_id=context.step.id,
        plan_step_index=context.plan_step_index, tool_key=tool_key,
        operation=str(payload.get("operation") or tool_key),
        selected_model_id=int(selected) if selected is not None else None,
        status="running", request_data=payload,
        started_at=datetime.utcnow(), created_at=datetime.utcnow(),
    )
    context.db.add(call)
    context.db.flush()
    return call


def register_existing_file_artifact(
    db: Session, run: AIAgentRun, source: AIAgentFile, *, role: str = "input",
    step_id: int | None = None, tool_call_id: int | None = None,
    tool_key: str | None = None,
) -> AIAgentArtifact:
    path = Path(source.storage_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise ValueError(f"Artifact file does not exist: {source.filename}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    content_type = source.content_type or "application/octet-stream"
    if content_type.startswith("image/"):
        artifact_type = "image"
    elif content_type.startswith("audio/"):
        artifact_type = "audio"
    elif content_type.startswith("video/"):
        artifact_type = "video"
    elif content_type in {"application/pdf"} or source.filename.lower().endswith((".pdf", ".docx")):
        artifact_type = "document"
    else:
        artifact_type = "file"
    artifact = AIAgentArtifact(
        artifact_uuid=str(uuid.uuid4()), run_id=run.id, step_id=step_id,
        tool_call_id=tool_call_id, tool_key=tool_key, artifact_type=artifact_type,
        role=role, filename=source.filename, storage_path=str(path),
        content_type=content_type, size_bytes=path.stat().st_size,
        checksum_sha256=digest,
        artifact_metadata={"source_agent_file_id": source.id},
        created_at=datetime.utcnow(),
    )
    db.add(artifact)
    db.flush()
    return artifact
