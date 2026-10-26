from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas import CostUsage, TokenUsage


class AgentParameters(BaseModel):
    model_config = ConfigDict(extra="allow")

    language: str | None = None
    quality_mode: Literal["fast", "balanced", "high"] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_steps: int | None = None
    max_budget_usd: float | None = None
    task_mode: str | None = None
    include_tests: bool | None = None
    platforms: list[str] | None = None
    content_count: int | None = None
    variants_count: int | None = None
    campaign_goal: str | None = None
    include_images: bool | None = None

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value):
        if value is not None and not 0 <= value <= 2:
            raise ValueError("temperature must be between 0 and 2")
        return value

    @field_validator("max_tokens", "max_steps", "content_count", "variants_count")
    @classmethod
    def positive_integer(cls, value):
        if value is not None and value < 1:
            raise ValueError("value must be positive")
        return value

    @field_validator("max_budget_usd")
    @classmethod
    def positive_budget(cls, value):
        if value is not None and value <= 0:
            raise ValueError("max_budget_usd must be greater than zero")
        return value


class AgentState(BaseModel):
    model_selection: Literal["user"] = "user"
    # Reserved for secondary tools (for example a Runware image model). The MVP
    # validates and executes selected_model_id as the primary reasoning model.
    selected_tool_model_ids: dict[str, int] = Field(default_factory=dict)
    parameters: AgentParameters = Field(default_factory=AgentParameters)


class AgentRunRequest(BaseModel):
    # Let clients create this UUID before starting the request so they can poll
    # the run status while the POST request is still executing.
    run_uuid: str | None = None
    user_id: int
    model_id: int
    selected_model_id: int
    conversation_uuid: str
    user_message: str
    state: AgentState = Field(default_factory=AgentState)
    debug: bool = False
    client_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("conversation_uuid", "user_message")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("value cannot be empty")
        return value

    @field_validator("run_uuid")
    @classmethod
    def valid_run_uuid(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return str(UUID(str(value)))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("run_uuid must be a valid UUID") from exc


class AgentEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    score: float = Field(ge=0, le=1)
    missing_requirements: list[str] = Field(default_factory=list)
    factual_issues: list[str] = Field(default_factory=list)
    formatting_issues: list[str] = Field(default_factory=list)
    feedback: str = ""


class AgentRunResponse(BaseModel):
    success: bool = True
    type: str = "agent_result"
    tool: str = "agents"
    agent: str
    run_uuid: str
    status: str
    provider: str
    model: str
    model_tier: str | None = None
    user_id: int
    model_id: int
    selected_model_id: int
    conversation_uuid: str
    progress: dict[str, Any] = Field(default_factory=dict)
    poll_url: str | None = None
    content: str | None = None
    usage: TokenUsage | None = None
    cost: CostUsage | None = None
    steps: list[dict[str, Any]] = Field(default_factory=list)
    files: list[dict[str, Any]] = Field(default_factory=list)
    request_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    debug: dict[str, Any] | None = None
