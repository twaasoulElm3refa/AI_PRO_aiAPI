from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.crud import get_ai_model, get_existing_conversation_for_task, list_ai_models
from app.database import get_db
from app.general_tools.model_catalog import serialize_ai_model
from app.media_tools import run_general_runware_media_operation
from app.models import AITrend
from app.schemas import MediaToolResponse
from app.security import verify_internal_api_key
from app.billing import (
    BillingAuthorization, authorize_charge, capture_charge,
    estimate_model_reservation, mark_charge_started,
    mark_reconciliation_required, release_charge,
)
from app.billing.errors import BillingError
from app.settings import get_settings

from .schemas import TrendRequest


router = APIRouter(prefix="/tasks/trends", tags=["trends"])


def _error(exc: Exception) -> HTTPException:
    if isinstance(exc, BillingError):
        return HTTPException(status_code=exc.status_code, detail=exc.as_detail())
    return HTTPException(status_code=400 if isinstance(exc, ValueError) else 502, detail=str(exc))


def _get_trend(db: Session, slug: str) -> AITrend:
    trend = db.execute(select(AITrend).where(
        AITrend.slug == slug,
        AITrend.is_active.is_(True),
        AITrend.deleted_at.is_(None),
    )).scalar_one_or_none()
    if not trend:
        raise ValueError("Trend not found or disabled")
    return trend


def _serialize_trend(trend: AITrend) -> dict:
    return {
        "id": trend.id,
        "slug": trend.slug,
        "name_ar": trend.name_ar,
        "name_en": trend.name_en,
        "description": trend.description,
        "thumbnail_url": trend.thumbnail_url,
        "preview_url": trend.preview_url,
        "endpoint": f"/tasks/trends/{trend.slug}",
        "input_schema": trend.input_schema or {},
        "output_schema": trend.output_schema or {},
        "allowed_model_ids": trend.allowed_model_ids or [],
        "sort_order": trend.sort_order,
    }


def _model_parameters(model, submitted) -> dict:
    schema = model.parameter_schema or {}
    values = submitted.model_dump(exclude_none=True)
    unknown = sorted(set(values) - set(schema)) if schema else []
    if unknown:
        raise ValueError(f"Parameters not supported by selected model: {', '.join(unknown)}")
    for name, definition in schema.items():
        if isinstance(definition, dict) and name not in values and "default" in definition:
            values[name] = definition["default"]
    return values


@router.get("")
def list_trends(
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
):
    rows = db.execute(select(AITrend).where(
        AITrend.is_active.is_(True), AITrend.deleted_at.is_(None)
    ).order_by(AITrend.sort_order, AITrend.id)).scalars().all()
    return {"success": True, "items": [_serialize_trend(row) for row in rows]}


@router.get("/{trend_slug}")
def get_trend(
    trend_slug: str,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
):
    try:
        return {"success": True, **_serialize_trend(_get_trend(db, trend_slug))}
    except Exception as exc:
        raise _error(exc) from exc


@router.get("/{trend_slug}/models")
def trend_models(
    trend_slug: str,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
):
    try:
        trend = _get_trend(db, trend_slug)
        rows, _ = list_ai_models(
            db, tool_key="general_media", operation="image_edit",
            provider="runware", page=1, per_page=100,
        )
        allowed = {int(value) for value in (trend.allowed_model_ids or [])}
        if allowed:
            rows = [row for row in rows if int(row.id) in allowed]
        return {
            "success": True, "trend": _serialize_trend(trend),
            "items": [serialize_ai_model(row) for row in rows],
        }
    except Exception as exc:
        raise _error(exc) from exc


@router.post("/{trend_slug}", response_model=MediaToolResponse)
async def run_trend(
    trend_slug: str,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> MediaToolResponse:
    authorization = BillingAuthorization(enabled=False)
    provider_started = False
    provider_completed = False
    try:
        content_type = (request.headers.get("content-type") or "").lower()
        if not content_type.startswith("multipart/form-data"):
            raise ValueError("Trend requests require multipart/form-data")
        form = await request.form()
        raw_payload = form.get("payload")
        if not raw_payload:
            raise ValueError("payload is required")
        req = TrendRequest.model_validate_json(str(raw_payload))
        uploaded_file = form.get("file")
        reference_file = form.get("reference_file")
        if uploaded_file is None or not getattr(uploaded_file, "filename", None):
            raise ValueError("file is required")

        trend = _get_trend(db, trend_slug)
        get_existing_conversation_for_task(
            db, req.user_id, req.sub_tool_id, req.conversation_uuid
        )
        model = get_ai_model(
            db, req.selected_model_id,
            tool_key="general_media", operation="image_edit",
        )
        if model.provider.lower() != "runware":
            raise ValueError("Trend models must use Runware")
        allowed = {int(value) for value in (trend.allowed_model_ids or [])}
        if allowed and int(model.id) not in allowed:
            raise ValueError("Selected model is not allowed for this trend")

        user_request = req.user_message.strip() or "Keep the default trend composition."
        prompt = trend.prompt_template.replace("{user_request}", user_request)
        parameters = _model_parameters(model, req.state.parameters)
        if get_settings().BILLING_ENFORCEMENT_ENABLED:
            estimate = estimate_model_reservation(
                model, input_text=prompt, units=1
            )
            authorization = authorize_charge(
                db, user_id=req.user_id,
                request_id=request.state.request_id,
                selected_model_id=req.selected_model_id,
                tool_key="trends", operation="image_edit",
                estimated_provider_cost_usd=estimate,
                pricing_snapshot={
                    "pricing": model.pricing or {},
                    "billing_config": model.billing_config or {},
                    "provider_model_id": model.provider_model_id,
                    "parameters": parameters,
                    "trend": trend.slug,
                },
                reference_type="conversation",
                reference_uuid=req.conversation_uuid,
            )
        provider_started = mark_charge_started(db, authorization)
        result = await run_general_runware_media_operation(
            db, user_id=req.user_id, sub_tool_id=req.sub_tool_id,
            conversation_uuid=req.conversation_uuid, operation="image_edit",
            prompt=prompt, file=uploaded_file, mask_file=None,
            model=model.provider_model_id, parameters=parameters,
            provider_metadata=model.provider_metadata or {},
            request_id=request.state.request_id,
            additional_files=[reference_file] if reference_file else None,
        )
        provider_completed = True
        billing = capture_charge(
            db, authorization,
            actual_provider_cost_usd=(
                result.cost.total_cost if result.cost is not None else None
            ),
        )
        return MediaToolResponse(
            tool=f"trend_{trend.slug}", provider=result.provider, model=result.model,
            user_id=req.user_id, sub_tool_id=req.sub_tool_id,
            conversation_uuid=req.conversation_uuid, message=result.message,
            files=result.files, count=result.count, request_id=result.request_id,
            usage=result.usage, cost=result.cost,
            metadata={
                "trend": trend.slug, "trend_name": trend.name_ar,
                "selected_model_id": req.selected_model_id,
                "model_tier": model.tier, "billing": billing, **result.metadata,
            },
            debug=result.debug,
        )
    except Exception as exc:
        if authorization.enabled and not provider_completed:
            if provider_started:
                mark_reconciliation_required(
                    db, authorization,
                    reason=f"trend_provider_result_unknown: {type(exc).__name__}",
                )
            else:
                release_charge(
                    db, authorization,
                    reason=f"trend_failed_before_provider_call: {type(exc).__name__}",
                )
        raise _error(exc) from exc
