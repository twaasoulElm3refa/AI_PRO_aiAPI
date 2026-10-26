from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.billing.context import set_current_authorization
from app.billing.errors import BillingConfigurationError, InsufficientBalanceError
from app.models import Wallet
from app.settings import get_settings


def _decimal(value, *, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise BillingConfigurationError(f"Invalid model pricing field: {field}") from exc
    if not result.is_finite() or result < 0:
        raise BillingConfigurationError(f"Invalid model pricing field: {field}")
    return result


@dataclass
class BillingAuthorization:
    enabled: bool = False
    reservation_uuid: str | None = None  # Backward-compatible field name.
    request_id: str | None = None
    user_id: int | None = None
    estimated_cost_usd: Decimal = Decimal("0")
    required_credits: int = 0
    available_credits: int = 0
    status: str = "disabled"
    provider_started: bool = False


def _estimated_input_tokens(text: str) -> int:
    # UTF-8 bytes are deliberately conservative for Arabic and mixed content.
    # The final debit still uses the provider's actual usage.
    return max(1, len(str(text or "").encode("utf-8")) + 64)


def estimate_model_reservation(
    model,
    *,
    input_text: str = "",
    max_output_tokens: int | None = None,
    units: int = 1,
    web_search_cost_usd: float | Decimal = 0,
) -> Decimal:
    """Return a conservative pre-request provider cost in USD.

    Despite the historical function name, this only estimates; it never reserves
    or debits credits.
    """
    if str(getattr(model, "tier", "standard")).lower() == "free":
        return Decimal("0")
    pricing = model.pricing if isinstance(getattr(model, "pricing", None), dict) else {}
    billing = pricing.get("billing") if isinstance(pricing.get("billing"), dict) else {}
    maximum = billing.get("max_request_usd", pricing.get("max_request_usd"))
    if maximum is not None:
        return _decimal(maximum, field="max_request_usd")

    operation = str(getattr(model, "operation", "") or "")
    request_fee = _decimal(pricing.get("request", 0), field="request")
    extra = _decimal(web_search_cost_usd, field="web_search_cost_usd")

    if operation in {"text_generation", "summarization", "document_generation"}:
        if pricing.get("input") is None or pricing.get("output") is None:
            raise BillingConfigurationError(
                f"Paid model {getattr(model, 'id', '?')} has no input/output pricing"
            )
        output_cap = max_output_tokens
        if output_cap is None:
            output_cap = billing.get("max_output_tokens", pricing.get("max_output_tokens"))
        if output_cap is None or int(output_cap) <= 0:
            raise BillingConfigurationError(
                f"Paid model {getattr(model, 'id', '?')} has no safe max_output_tokens"
            )
        input_cost = Decimal(_estimated_input_tokens(input_text)) * _decimal(
            pricing["input"], field="input"
        ) / Decimal(1_000_000)
        output_cost = Decimal(int(output_cap)) * _decimal(
            pricing["output"], field="output"
        ) / Decimal(1_000_000)
        return input_cost + output_cost + request_fee + extra

    unit_candidates = (
        ("per_million_characters", Decimal(max(0, units)) / Decimal(1_000_000)),
        ("per_million_utf8_bytes", Decimal(max(0, units)) / Decimal(1_000_000)),
        ("per_character", Decimal(max(0, units))),
        ("per_second", Decimal(max(0, units))),
        ("per_output", Decimal(max(1, units))),
        ("per_request", Decimal(1)),
    )
    for key, quantity in unit_candidates:
        if pricing.get(key) is not None:
            return quantity * _decimal(pricing[key], field=key) + request_fee + extra
    raise BillingConfigurationError(
        f"Paid model {getattr(model, 'id', '?')} has no safe pre-request price"
    )


def _required_credits(estimated_cost_usd: Decimal) -> int:
    settings = get_settings()
    multiplier = _decimal(settings.BILLING_MARKUP_MULTIPLIER, field="markup")
    margin = Decimal("1") + _decimal(
        settings.BILLING_SAFETY_MARGIN_PERCENT, field="safety_margin"
    ) / Decimal("100")
    raw = estimated_cost_usd * multiplier * margin * Decimal(
        settings.BILLING_CREDITS_PER_USD
    )
    return max(int(settings.BILLING_MIN_REQUIRED_CREDITS), int(math.ceil(raw)))


def authorize_charge(
    db: Session,
    *,
    user_id: int,
    request_id: str,
    selected_model_id: int | None,
    tool_key: str,
    operation: str | None,
    estimated_provider_cost_usd,
    pricing_snapshot: dict | None = None,
    reference_type: str | None = None,
    reference_uuid: str | None = None,
) -> BillingAuthorization:
    settings = get_settings()
    if not settings.BILLING_ENFORCEMENT_ENABLED:
        return BillingAuthorization(enabled=False)
    estimate = _decimal(estimated_provider_cost_usd, field="estimated_provider_cost_usd")
    required = 0 if estimate == 0 else _required_credits(estimate)

    db.expire_all()
    wallet = db.execute(select(Wallet).where(
        Wallet.user_id == int(user_id),
        Wallet.is_active.is_(True),
        Wallet.deleted_at.is_(None),
    )).scalar_one_or_none()
    available = int(wallet.balance) if wallet is not None else 0
    if available < required:
        raise InsufficientBalanceError(
            "رصيدك غير كافٍ لاستخدام هذا النموذج.",
            available_credits=available,
            required_credits=required,
            shortfall_credits=required - available,
            credits_per_usd=int(settings.BILLING_CREDITS_PER_USD),
            selected_model_id=selected_model_id,
            tool_key=tool_key,
            operation=operation,
        )
    authorization = BillingAuthorization(
        enabled=True,
        reservation_uuid=str(uuid.uuid4()),
        request_id=request_id,
        user_id=int(user_id),
        estimated_cost_usd=estimate,
        required_credits=required,
        available_credits=available,
        status="checked",
    )
    return authorization


def mark_charge_started(db: Session, authorization: BillingAuthorization) -> bool:
    if not authorization.enabled:
        return False
    authorization.status = "executing"
    authorization.provider_started = True
    set_current_authorization(authorization)
    return True


def capture_charge(
    db: Session,
    authorization: BillingAuthorization,
    *,
    actual_provider_cost_usd=None,
) -> dict:
    if not authorization.enabled:
        return {"billing_enforced": False}
    actual = (
        _decimal(actual_provider_cost_usd, field="actual_provider_cost_usd")
        if actual_provider_cost_usd is not None else None
    )
    actual_credits = (
        int(math.ceil(
            actual
            * Decimal(get_settings().BILLING_CREDITS_PER_USD)
            * _decimal(get_settings().BILLING_MARKUP_MULTIPLIER, field="markup")
        ))
        if actual is not None else None
    )
    authorization.status = "completed"
    return {
        "billing_enforced": True,
        "mode": "check_only",
        "check_uuid": authorization.reservation_uuid,
        "available_credits": authorization.available_credits,
        "required_credits": authorization.required_credits,
        "estimated_provider_cost_usd": float(authorization.estimated_cost_usd),
        "actual_provider_cost_usd": float(actual) if actual is not None else None,
        "actual_credits": actual_credits,
        "status": "completed",
        "debit_owner": "trusted_backend",
    }


def release_charge(db: Session, authorization: BillingAuthorization, *, reason: str) -> bool:
    if not authorization.enabled:
        return False
    authorization.status = "failed"
    return True


def mark_reconciliation_required(
    db: Session, authorization: BillingAuthorization, *, reason: str
) -> bool:
    if not authorization.enabled:
        return False
    authorization.status = "provider_result_unknown"
    return True


def authorization_status(db: Session, authorization: BillingAuthorization) -> str:
    return authorization.status
