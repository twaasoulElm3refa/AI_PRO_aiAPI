from __future__ import annotations

from sqlalchemy.orm import Session

from app.billing import (
    BillingAuthorization, authorize_charge, capture_charge,
    estimate_model_reservation, mark_charge_started,
    mark_reconciliation_required, release_charge,
)
from app.crud import get_ai_model, get_existing_conversation_for_model
from app.providers import send_messages_with_model
from app.schemas import ChatMessage, CostUsage, TokenUsage
from app.settings import get_settings

from .catalog import _text_parameters
from .schemas import GeneralTextRequest, GeneralToolResponse


SYSTEM_PROMPTS = {
    "general_chat": "You are a capable general chat and writing assistant. Follow the user's requested language and format. Return only the useful final answer; never expose private reasoning.",
    "general_code": "You are a senior software engineer. Produce correct, secure, maintainable code. Respect the requested mode and programming language. Clearly separate code and explanation when explanation is requested. Never expose private reasoning.",
    "general_translation": "You are a professional translator. Preserve meaning, names, numbers, formatting, and tone. Return only the translation unless a note is essential. Never expose private reasoning.",
}


def _actual_openrouter_cost(raw_usage: dict | None) -> CostUsage | None:
    """Use the amount OpenRouter actually charged, never a local price estimate."""
    usage = raw_usage if isinstance(raw_usage, dict) else {}
    value = usage.get("cost")
    if value is None:
        return None
    try:
        total = float(value)
    except (TypeError, ValueError):
        return None
    return CostUsage(total_cost=round(total, 10), currency="USD")


def _instruction(tool: str, req: GeneralTextRequest) -> str:
    p = req.state.parameters
    parts = []
    if p.language:
        parts.append(f"Response language: {p.language}")
    if tool == "general_code":
        parts.extend([f"Task mode: {p.task_mode}", f"Programming language: {p.programming_language or 'infer'}", f"Include explanation: {p.include_explanation}", f"Include tests: {p.include_tests}"])
    if tool == "general_translation":
        if not p.target_language:
            raise ValueError("target_language is required for translation")
        parts.extend([f"Source language: {p.source_language}", f"Target language: {p.target_language}", f"Preserve formatting: {p.preserve_formatting}"])
    return "\n".join(parts)


def _effective_text_parameters(tool: str, db_model, requested) -> dict:
    """Resolve request overrides -> per-model DB defaults -> tool defaults."""
    schema = {
        **_text_parameters(tool),
        **(db_model.parameter_schema or {}),
    }

    def resolve(name: str, fallback):
        supplied = getattr(requested, name, None)
        if supplied is not None:
            return supplied
        definition = schema.get(name)
        if isinstance(definition, dict) and definition.get("default") is not None:
            return definition["default"]
        return fallback

    return {
        "temperature": float(resolve("temperature", 0.5)),
        "max_tokens": int(resolve("max_tokens", 4000)),
        "quality_mode": str(resolve("quality_mode", "balanced")),
        "reasoning_effort": str(resolve("reasoning_effort", "none")),
    }


async def run_general_text(db: Session, tool: str, req: GeneralTextRequest, request_id: str) -> GeneralToolResponse:
    settings = get_settings()
    if not settings.GENERAL_TOOLS_ENABLED:
        raise ValueError("General tools are disabled")
    if len(req.user_message) > settings.MAX_USER_MESSAGE_LENGTH:
        raise ValueError(f"user_message exceeds max length of {settings.MAX_USER_MESSAGE_LENGTH}")

    conversation = get_existing_conversation_for_model(
        db, req.user_id, req.model_id, req.conversation_uuid
    )
    db_model = get_ai_model(
        db, req.selected_model_id, tool_key=tool, operation="text_generation"
    )
    if db_model.provider.lower() != "openrouter":
        raise ValueError(f"Unsupported provider for {tool}: {db_model.provider}")
    model = db_model.provider_model_id
    tier = db_model.tier
    if req.state.model and req.state.model != model:
        raise ValueError("state.model does not match selected_model_id; omit state.model")
    p = req.state.parameters
    effective = _effective_text_parameters(tool, db_model, p)
    max_tokens = min(effective["max_tokens"], settings.GENERAL_TOOLS_MAX_OUTPUT_TOKENS)
    reasoning_effort = None if effective["reasoning_effort"] == "none" else effective["reasoning_effort"]
    if reasoning_effort is None:
        reasoning_effort = {
            "fast": None,
            "balanced": "low",
            "high": "medium",
        }[effective["quality_mode"]]

    system = SYSTEM_PROMPTS[tool]
    if p.system_prompt and tool != "general_translation":
        system += f"\nAdditional user configuration:\n{p.system_prompt[:4000]}"
    messages = [ChatMessage(role="system", content=system)]
    # Laravel owns model-conversation messages. Add model-message history here
    # only after mapping its actual table; do not read the legacy messages table.
    instruction = _instruction(tool, req)
    user_content = f"{instruction}\n\n{req.user_message}" if instruction else req.user_message
    messages.append(ChatMessage(role="user", content=user_content))

    web_search_cost = 0.0
    if p.web_search and tool == "general_chat":
        web_search_cost = (
            settings.WEB_SEARCH_COST_PER_1000_RESULTS
            * settings.WEB_SEARCH_DEFAULT_MAX_TOTAL_RESULTS
            / 1000
        )
    authorization = BillingAuthorization(enabled=False)
    if settings.BILLING_ENFORCEMENT_ENABLED:
        estimated_cost = estimate_model_reservation(
            db_model,
            input_text=f"{system}\n{user_content}",
            max_output_tokens=max_tokens,
            web_search_cost_usd=web_search_cost,
        )
        authorization = authorize_charge(
            db,
            user_id=req.user_id,
            request_id=request_id,
            selected_model_id=req.selected_model_id,
            tool_key=tool,
            operation="text_generation",
            estimated_provider_cost_usd=estimated_cost,
            pricing_snapshot={
                "pricing": db_model.pricing or {},
                "billing_config": db_model.billing_config or {},
                "provider_model_id": db_model.provider_model_id,
            },
            reference_type="conversation",
            reference_uuid=req.conversation_uuid,
        )

    provider_started = False
    try:
        provider_started = mark_charge_started(db, authorization)
        result = await send_messages_with_model(
            model_key="writer_pro",
            model_override=model,
            messages=messages,
            temperature_override=effective["temperature"],
            max_tokens_override=max_tokens,
            enable_web_search=bool(p.web_search and tool == "general_chat"),
            exclude_reasoning=True,
            # Some advanced models reject effort="none". Reasoning is still hidden.
            reasoning_effort=reasoning_effort,
        )
    except Exception as exc:
        if provider_started:
            mark_reconciliation_required(
                db, authorization,
                reason=f"provider_call_failed: {type(exc).__name__}",
            )
        else:
            release_charge(
                db, authorization,
                reason=f"failed_before_provider_call: {type(exc).__name__}",
            )
        raise
    actual_cost = _actual_openrouter_cost(result.raw_usage)
    billing = capture_charge(
        db,
        authorization,
        actual_provider_cost_usd=actual_cost.total_cost if actual_cost else None,
    )
    usage = TokenUsage(
        input_tokens=result.input_tokens, output_tokens=result.output_tokens, total_tokens=result.total_tokens
    )
    return GeneralToolResponse(
        tool=tool, provider="openrouter", model=result.returned_model or model, model_tier=tier,
        user_id=req.user_id, model_id=req.model_id,
        selected_model_id=req.selected_model_id,
        conversation_uuid=req.conversation_uuid,
        content=result.content, request_id=request_id, usage=usage,
        cost=actual_cost,
        metadata={
            **result.trace_metadata(),
            "billing_source": "openrouter_usage",
            "provider_usage": result.raw_usage or {},
            "billing": billing,
            "effective_parameters": {
                "temperature": effective["temperature"],
                "max_tokens": max_tokens,
                "quality_mode": effective["quality_mode"],
                "reasoning_effort": reasoning_effort or "none",
            },
        },
    )
