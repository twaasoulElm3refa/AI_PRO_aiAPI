from datetime import datetime
from sqlalchemy import (
    BigInteger, String, Text, DateTime, Boolean, ForeignKey, JSON, Numeric,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.settings import get_settings


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role: Mapped[str | None] = mapped_column(String(50), nullable=True)
    is_active: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class SubTool(Base):
    __tablename__ = "sub_tools"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    main_tool_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    website: Mapped[str | None] = mapped_column(String(50), nullable=True)
    image: Mapped[str | None] = mapped_column(String(255), nullable=True)
    slug: Mapped[str | None] = mapped_column(String(255), nullable=True)
    prompt_placeholder: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Existing Laravel JSON configuration. Billing pipeline metadata may live
    # under config["billing"], without adding a parallel column.
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_free: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    endpoint: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    sort_order: Mapped[int | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    sub_tool_id: Mapped[int] = mapped_column(ForeignKey("sub_tools.id"), index=True)
    uuid: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    is_pinned: Mapped[bool | None] = mapped_column(Boolean, default=False)
    is_archived: Mapped[bool | None] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        order_by="Message.id",
    )


class ModelsConversation(Base):
    """Conversation table used only by the new general-model tools.

    The table name is configurable and defaults to ``models_conversations``.
    """

    __tablename__ = get_settings().MODELS_CONVERSATIONS_TABLE

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    model_id: Mapped[int] = mapped_column(BigInteger, index=True)
    uuid: Mapped[str] = mapped_column(String(255), unique=True, index=True)


class AIModel(Base):
    """Unified provider-model catalog for text, image, audio and video tools."""
    __tablename__ = "ai_models"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(50), index=True)
    provider_model_id: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_key: Mapped[str] = mapped_column(String(100), index=True)
    operation: Mapped[str] = mapped_column(String(100), index=True)
    tier: Mapped[str] = mapped_column(String(30), default="standard", index=True)
    capabilities: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    parameter_schema: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    pricing: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    provider_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    is_available: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    is_recommended: Mapped[bool] = mapped_column(Boolean, default=False)
    sort_order: Mapped[int] = mapped_column(default=0)
    pricing_updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    provider_updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    @property
    def billing_config(self) -> dict:
        """Compatibility view: billing controls are stored inside pricing JSON."""
        value = self.pricing if isinstance(self.pricing, dict) else {}
        nested = value.get("billing")
        return nested if isinstance(nested, dict) else {}


class Wallet(Base):
    """Read-only view of the Laravel credits wallet used by the AI guard."""
    __tablename__ = "wallets"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    balance: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AITrend(Base):
    """Database-driven visual trend; adding a row creates a new trend API slug."""
    __tablename__ = "ai_trends"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    name_ar: Mapped[str] = mapped_column(String(255))
    name_en: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    thumbnail_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    preview_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    prompt_template: Mapped[str] = mapped_column(Text)
    negative_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_schema: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    output_schema: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    allowed_model_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    sort_order: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AIAgent(Base):
    __tablename__ = "ai_agents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    name_ar: Mapped[str] = mapped_column(String(255))
    name_en: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    system_prompt: Mapped[str] = mapped_column(Text)
    primary_model_tool: Mapped[str] = mapped_column(String(100), default="general_chat")
    allowed_tools: Mapped[list | None] = mapped_column(JSON, nullable=True)
    allowed_model_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    input_schema: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    output_schema: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    default_parameters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    max_steps: Mapped[int] = mapped_column(default=8)
    max_files: Mapped[int] = mapped_column(default=5)
    max_execution_seconds: Mapped[int] = mapped_column(default=300)
    default_budget_usd: Mapped[float | None] = mapped_column(Numeric(12, 6), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    is_recommended: Mapped[bool] = mapped_column(Boolean, default=False)
    sort_order: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AIAgentRun(Base):
    __tablename__ = "ai_agent_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_uuid: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    agent_id: Mapped[int] = mapped_column(BigInteger, index=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    model_id: Mapped[int] = mapped_column(BigInteger, index=True)
    selected_model_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    conversation_uuid: Mapped[str] = mapped_column(String(255), index=True)
    user_message: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(30), default="queued", index=True)
    input_state: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    final_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_files: Mapped[list | None] = mapped_column(JSON, nullable=True)
    usage_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    cost_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    state_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    current_node: Mapped[str | None] = mapped_column(String(100), nullable=True)
    iteration: Mapped[int] = mapped_column(default=0)
    max_iterations: Mapped[int] = mapped_column(default=2)
    total_steps: Mapped[int] = mapped_column(default=0)
    completed_steps: Mapped[int] = mapped_column(default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AIAgentStep(Base):
    __tablename__ = "ai_agent_steps"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(BigInteger, index=True)
    step_number: Mapped[int] = mapped_column()
    step_key: Mapped[str | None] = mapped_column(String(100), nullable=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    tool_key: Mapped[str | None] = mapped_column(String(100), nullable=True)
    operation: Mapped[str | None] = mapped_column(String(100), nullable=True)
    selected_model_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="pending", index=True)
    tool_input: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    tool_output: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    output_files: Mapped[list | None] = mapped_column(JSON, nullable=True)
    usage_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    cost_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AIAgentFile(Base):
    __tablename__ = "ai_agent_files"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(BigInteger, index=True)
    step_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    role: Mapped[str] = mapped_column(String(30))
    filename: Mapped[str] = mapped_column(String(255))
    storage_path: Mapped[str] = mapped_column(Text)
    public_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    file_metadata: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AIAgentCheckpoint(Base):
    __tablename__ = "ai_agent_checkpoints"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    checkpoint_uuid: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    run_id: Mapped[int] = mapped_column(BigInteger, index=True)
    step_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    node_name: Mapped[str] = mapped_column(String(100), index=True)
    iteration: Mapped[int] = mapped_column(default=0)
    status: Mapped[str] = mapped_column(String(30), default="completed", index=True)
    state_data: Mapped[dict] = mapped_column(JSON)
    checkpoint_metadata: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AIAgentToolCall(Base):
    __tablename__ = "ai_agent_tool_calls"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    call_uuid: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    run_id: Mapped[int] = mapped_column(BigInteger, index=True)
    step_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    plan_step_index: Mapped[int | None] = mapped_column(nullable=True)
    tool_key: Mapped[str] = mapped_column(String(100), index=True)
    operation: Mapped[str | None] = mapped_column(String(100), nullable=True)
    selected_model_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="pending", index=True)
    request_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    response_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    usage_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    cost_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AIAgentArtifact(Base):
    __tablename__ = "ai_agent_artifacts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    artifact_uuid: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    run_id: Mapped[int] = mapped_column(BigInteger, index=True)
    step_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    tool_call_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    tool_key: Mapped[str | None] = mapped_column(String(100), nullable=True)
    artifact_type: Mapped[str] = mapped_column(String(50), index=True)
    role: Mapped[str] = mapped_column(String(30), default="output")
    filename: Mapped[str] = mapped_column(String(255))
    storage_path: Mapped[str] = mapped_column(Text)
    content_type: Mapped[str | None] = mapped_column(String(150), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    artifact_metadata: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), index=True)
    content: Mapped[str] = mapped_column(Text)
    is_error: Mapped[bool | None] = mapped_column(Boolean, default=False)
    role: Mapped[str] = mapped_column(String(20), index=True)  # user / system / assistant
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")
