from __future__ import annotations

from contextvars import ContextVar, Token
from typing import TYPE_CHECKING

from app.billing.errors import BillingConfigurationError
from app.settings import get_settings

if TYPE_CHECKING:
    from app.billing.service import BillingAuthorization
    from sqlalchemy.orm import Session


_current_authorization: ContextVar[BillingAuthorization | None] = ContextVar(
    "billing_authorization", default=None
)
_current_db: ContextVar[Session | None] = ContextVar("billing_db", default=None)


def set_current_authorization(value: BillingAuthorization | None) -> Token:
    return _current_authorization.set(value)


def reset_current_authorization(token: Token) -> None:
    _current_authorization.reset(token)


def set_current_db(value: Session | None) -> Token:
    return _current_db.set(value)


def reset_current_db(token: Token) -> None:
    _current_db.reset(token)


def get_current_db() -> Session | None:
    return _current_db.get()


def mark_current_provider_call_started() -> None:
    """Last fail-closed boundary immediately before an external paid request."""
    authorization = _current_authorization.get()
    if authorization is None:
        if get_settings().BILLING_ENFORCEMENT_ENABLED:
            raise BillingConfigurationError(
                "Provider call was blocked because no credits check is active"
            )
        return
    if authorization.enabled:
        authorization.status = "executing"
        authorization.provider_started = True
