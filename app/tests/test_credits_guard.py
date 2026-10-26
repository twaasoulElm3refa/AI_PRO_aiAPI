from __future__ import annotations

import os
import unittest
from datetime import datetime
from decimal import Decimal


os.environ["APP_ENV"] = "test"
os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"
os.environ["BILLING_ENFORCEMENT_ENABLED"] = "true"
os.environ["BILLING_CREDITS_PER_USD"] = "1000000"
os.environ["BILLING_SAFETY_MARGIN_PERCENT"] = "10"
os.environ["BILLING_MARKUP_MULTIPLIER"] = "1"

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.billing import (
    InsufficientBalanceError,
    authorize_charge,
    capture_charge,
    estimate_model_reservation,
)
from app.database import Base
from app.models import AIModel, User, Wallet


class CreditsGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
        Base.metadata.create_all(self.engine)
        self.db = Session(self.engine)
        now = datetime.utcnow()
        self.db.add(User(id=1, name="Test", is_active=True, created_at=now))
        self.model = AIModel(
            id=1,
            provider="openrouter", provider_model_id="test/model", name="Test",
            tool_key="general_chat", operation="text_generation", tier="standard",
            pricing={"input": 1.0, "output": 2.0, "currency": "USD"},
            is_active=True, is_available=True, created_at=now,
        )
        self.db.add(self.model)
        self.db.flush()
        self.db.add(Wallet(
            id=1, user_id=1, balance=10_000, is_active=True,
            deleted_at=None,
        ))
        self.db.commit()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def test_estimate_uses_input_and_max_output_prices(self) -> None:
        estimate = estimate_model_reservation(
            self.model, input_text="hello", max_output_tokens=1000
        )
        self.assertGreater(estimate, Decimal("0.002"))

    def test_insufficient_credits_blocks_before_provider(self) -> None:
        with self.assertRaises(InsufficientBalanceError) as caught:
            authorize_charge(
                self.db, user_id=1, request_id="req-low",
                selected_model_id=self.model.id, tool_key="general_chat",
                operation="text_generation", estimated_provider_cost_usd=Decimal("1"),
            )
        detail = caught.exception.as_detail()
        self.assertEqual(detail["code"], "INSUFFICIENT_CREDITS")
        self.assertGreater(detail["required_credits"], detail["available_credits"])

    def test_check_only_never_changes_wallet(self) -> None:
        authorization = authorize_charge(
            self.db, user_id=1, request_id="req-ok",
            selected_model_id=self.model.id, tool_key="general_chat",
            operation="text_generation", estimated_provider_cost_usd=Decimal("0.001"),
        )
        result = capture_charge(
            self.db, authorization, actual_provider_cost_usd=Decimal("0.0005")
        )
        self.db.expire_all()
        self.assertEqual(self.db.get(Wallet, 1).balance, 10_000)
        self.assertEqual(result["mode"], "check_only")
        self.assertEqual(result["actual_credits"], 500)


if __name__ == "__main__":
    unittest.main()
