from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AIModel
from app.settings import get_settings

from .catalog import _text_parameters


def serialize_ai_model(model: AIModel) -> dict[str, Any]:
    default_schema = (
        _text_parameters(model.tool_key)
        if model.operation == "text_generation"
        else {}
    )
    parameter_schema = {**default_schema, **(model.parameter_schema or {})}
    recommended_parameters = {
        key: definition["default"]
        for key, definition in parameter_schema.items()
        if isinstance(definition, dict) and "default" in definition
    }
    return {
        "id": model.id,
        "provider": model.provider,
        "provider_model_id": model.provider_model_id,
        "name": model.name,
        "description": model.description,
        "tool_key": model.tool_key,
        "operation": model.operation,
        "tier": model.tier,
        "is_free": model.tier == "free",
        "capabilities": model.capabilities or [],
        "parameter_schema": parameter_schema,
        "recommended_parameters": recommended_parameters,
        "pricing": model.pricing,
        "is_available": bool(model.is_available),
        "is_recommended": bool(model.is_recommended),
        "sort_order": model.sort_order,
        "pricing_updated_at": model.pricing_updated_at.isoformat() if model.pricing_updated_at else None,
        "provider_updated_at": model.provider_updated_at.isoformat() if model.provider_updated_at else None,
    }


def _openrouter_price(pricing: dict[str, Any]) -> dict[str, Any]:
    def per_million(key: str) -> float:
        try:
            return float(pricing.get(key) or 0) * 1_000_000
        except (TypeError, ValueError):
            return 0.0
    return {
        "currency": "USD",
        "unit": "per_1m_tokens",
        "input": per_million("prompt"),
        "output": per_million("completion"),
        "request": float(pricing.get("request") or 0),
        "image": float(pricing.get("image") or 0),
        "estimated": True,
        "source": "openrouter_models_api",
    }


def _openrouter_tts_price(provider_pricing: dict[str, Any], existing: dict | None) -> dict[str, Any]:
    existing = existing if isinstance(existing, dict) else {}
    if any(key in existing for key in (
        "per_million_characters", "per_million_utf8_bytes", "per_character"
    )):
        return existing
    try:
        per_character = float(provider_pricing.get("completion") or 0.0)
    except (TypeError, ValueError):
        per_character = 0.0
    return {
        "currency": "USD",
        "per_million_characters": per_character * 1_000_000,
        "estimated": True,
        "source": "openrouter_models_api",
    }


async def sync_openrouter_models(db: Session) -> dict[str, int]:
    """Refresh metadata/pricing only for OpenRouter rows already approved in DB."""
    settings = get_settings()
    headers = {"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"}
    url = f"{settings.OPENROUTER_BASE_URL.rstrip('/')}/models?output_modalities=all"
    async with httpx.AsyncClient(timeout=45) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        payload = response.json()
    provider_rows = {
        str(item.get("id") or ""): item
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    }
    rows = list(db.execute(select(AIModel).where(
        AIModel.provider == "openrouter", AIModel.deleted_at.is_(None)
    )).scalars().all())
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    updated = unavailable = 0
    for model in rows:
        item = provider_rows.get(model.provider_model_id)
        if item is None:
            model.is_available = False
            model.provider_updated_at = now
            unavailable += 1
            continue
        if model.operation == "text_to_speech" and not (model.parameter_schema or {}):
            # OpenRouter requires a voice, and voice identifiers are model-specific.
            # Do not re-enable catalog rows that would fall back to an incompatible
            # global voice after a provider sync.
            model.is_available = False
            model.provider_updated_at = now
            unavailable += 1
            continue
        model.is_available = True
        model.name = str(item.get("name") or model.name)
        model.description = str(item.get("description") or model.description or "") or None
        if model.operation == "text_to_speech":
            model.pricing = _openrouter_tts_price(
                item.get("pricing") or {}, model.pricing
            )
        else:
            refreshed_pricing = _openrouter_price(item.get("pricing") or {})
            existing_pricing = model.pricing if isinstance(model.pricing, dict) else {}
            # Pipeline ceilings are operator policy, not provider catalog data.
            # Preserve them when refreshing volatile provider rates.
            if isinstance(existing_pricing.get("billing"), dict):
                refreshed_pricing["billing"] = existing_pricing["billing"]
            if existing_pricing.get("max_output_tokens") is not None:
                refreshed_pricing["max_output_tokens"] = existing_pricing["max_output_tokens"]
            model.pricing = refreshed_pricing
        model.capabilities = item.get("architecture") or model.capabilities
        model.provider_metadata = {
            "context_length": item.get("context_length"),
            "supported_parameters": item.get("supported_parameters"),
            "top_provider": item.get("top_provider"),
        }
        model.pricing_updated_at = now
        model.provider_updated_at = now
        updated += 1
    db.commit()
    return {"updated": updated, "unavailable": unavailable, "total": len(rows)}
