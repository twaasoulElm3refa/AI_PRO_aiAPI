from __future__ import annotations


class BillingError(RuntimeError):
    status_code = 503
    code = "BILLING_ERROR"

    def __init__(self, message: str, **details):
        super().__init__(message)
        self.message = message
        self.details = details

    def as_detail(self) -> dict:
        return {"code": self.code, "message": self.message, **self.details}


class InsufficientBalanceError(BillingError):
    status_code = 402
    code = "INSUFFICIENT_CREDITS"


class BillingConfigurationError(BillingError):
    status_code = 503
    code = "BILLING_CONFIGURATION_ERROR"

