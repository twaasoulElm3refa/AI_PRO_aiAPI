from __future__ import annotations

from typing import Awaitable, Callable, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.billing.context import (
    reset_current_authorization, reset_current_db,
    set_current_authorization, set_current_db,
)
from app.billing.errors import BillingConfigurationError
from app.billing.service import (
    BillingAuthorization,
    authorize_charge,
    capture_charge,
    estimate_model_reservation,
    mark_reconciliation_required,
    release_charge,
)
from app.models import AIModel, SubTool
from app.settings import get_settings


T = TypeVar("T")


def _legacy_pricing_model(db: Session, tool_key: str, operation: str) -> AIModel:
    rows = list(db.execute(select(AIModel).where(
        AIModel.tool_key == tool_key,
        AIModel.operation == operation,
        AIModel.is_active.is_(True),
        AIModel.is_available.is_(True),
        AIModel.deleted_at.is_(None),
    ).order_by(AIModel.is_recommended.desc(), AIModel.sort_order, AIModel.id)).scalars())
    if not rows:
        raise BillingConfigurationError(
            f"Legacy tool {tool_key}/{operation} has no active billing model"
        )
    return rows[0]


async def run_legacy_billable(
    db: Session,
    *,
    user_id: int,
    sub_tool_id: int,
    request_id: str,
    tool_key: str,
    operation: str,
    call: Callable[[], Awaitable[T]],
    request_payload=None,
) -> T:
    """Credits gate for legacy routes whose internal pipeline may call AI more than once."""
    authorization = BillingAuthorization(enabled=False)
    token = None
    db_token = None
    try:
        model = _legacy_pricing_model(db, tool_key, operation)
        if request_payload is not None:
            if isinstance(request_payload, dict):
                if operation in {
                    "image_generation", "background_remove", "image_upscale",
                    "speech_to_text", "text_to_speech",
                }:
                    if not request_payload.get("model_override"):
                        request_payload["model_override"] = model.provider_model_id
                    if not request_payload.get("provider_override"):
                        request_payload["provider_override"] = model.provider
            else:
                if hasattr(request_payload, "model"):
                    request_payload.model = model.provider_model_id
                if hasattr(request_payload, "provider"):
                    request_payload.provider = model.provider
        if get_settings().BILLING_ENFORCEMENT_ENABLED:
            sub_tool = db.execute(select(SubTool).where(
                SubTool.id == int(sub_tool_id),
                SubTool.is_active.is_(True),
                SubTool.deleted_at.is_(None),
            )).scalar_one_or_none()
            if sub_tool is None:
                raise ValueError("Sub-tool not found or disabled")
            estimate = estimate_model_reservation(model, units=1)
            authorization = authorize_charge(
                db,
                user_id=user_id,
                request_id=request_id,
                selected_model_id=model.id,
                tool_key=tool_key,
                operation=operation,
                estimated_provider_cost_usd=estimate,
                pricing_snapshot={
                    "pricing": model.pricing or {},
                    "provider_model_id": model.provider_model_id,
                    "legacy_pipeline": True,
                },
                reference_type="sub_tool",
                reference_uuid=str(sub_tool_id),
            )
        token = set_current_authorization(authorization)
        db_token = set_current_db(db)
        # Provider modules call mark_current_provider_call_started() at the
        # actual network boundary. Do not mark validation/file work as paid.
        result = await call()
        actual_cost = getattr(getattr(result, "cost", None), "total_cost", None)
        billing = capture_charge(
            db, authorization, actual_provider_cost_usd=actual_cost
        )
        metadata = getattr(result, "metadata", None)
        if isinstance(metadata, dict):
            metadata["billing"] = billing
        return result
    except Exception as exc:
        if authorization.enabled:
            if authorization.provider_started:
                mark_reconciliation_required(
                    db, authorization,
                    reason=f"legacy_provider_result_unknown: {type(exc).__name__}",
                )
            else:
                release_charge(
                    db, authorization,
                    reason=f"legacy_failed_before_provider: {type(exc).__name__}",
                )
        raise
    finally:
        if token is not None:
            reset_current_authorization(token)
        if db_token is not None:
            reset_current_db(db_token)
