from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas import CostUsage, GeneratedMediaFileInfo, TokenUsage


class GeneralParameters(BaseModel):
    # Optional in the request. Missing values are resolved from the selected
    # ai_models.parameter_schema and then from safe tool defaults.
    temperature: float | None = None
    max_tokens: int | None = None
    quality_mode: Literal["fast", "balanced", "high"] | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None
    language: str | None = None
    system_prompt: str | None = None
    web_search: bool = False
    task_mode: Literal["generate", "explain", "debug", "review", "refactor", "test"] = "generate"
    programming_language: str | None = None
    include_explanation: bool = True
    include_tests: bool = False
    source_language: str = "auto"
    target_language: str | None = None
    preserve_formatting: bool = True

    @field_validator("temperature")
    @classmethod
    def temperature_range(cls, value: float) -> float:
        if value is None:
            return value
        if not 0 <= value <= 2:
            raise ValueError("temperature must be between 0 and 2")
        return value

    @field_validator("max_tokens")
    @classmethod
    def tokens_range(cls, value: int) -> int:
        if value is None:
            return value
        if not 64 <= value <= 12000:
            raise ValueError("max_tokens must be between 64 and 12000")
        return value


class GeneralState(BaseModel):
    # Deprecated: AI model selection is made with selected_model_id.
    model: str | None = None
    parameters: GeneralParameters = Field(default_factory=GeneralParameters)


class GeneralTextRequest(BaseModel):
    user_id: int
    # Existing application tool/model id stored on models_conversations.
    model_id: int
    # User-selected executable AI model from ai_models.
    selected_model_id: int
    conversation_uuid: str
    user_message: str
    state: GeneralState = Field(default_factory=GeneralState)
    debug: bool = False
    client_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("conversation_uuid", "user_message")
    @classmethod
    def required_text(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("value cannot be empty")
        return value


class GeneralMediaParameters(BaseModel):
    model_config = ConfigDict(extra="allow")

    size: str | None = None
    quality: Literal["low", "medium", "high"] | None = None
    count: int | None = None
    output_format: Literal["png", "jpg", "jpeg", "webp"] | None = None
    negative_prompt: str | None = None
    enhance_prompt: bool | None = None
    seed: int | None = None
    scale: Literal[2, 3, 4] | None = None
    face_enhance: bool | None = None
    quality_mode: Literal["faithful", "balanced", "quality"] | None = None
    enhance_details: bool | None = None
    realism: bool | None = None
    width: int | None = None
    height: int | None = None
    duration: int | None = None
    outpaint_top: int | None = None
    outpaint_right: int | None = None
    outpaint_bottom: int | None = None
    outpaint_left: int | None = None
    auto_crop: bool | None = None

    @field_validator("count")
    @classmethod
    def count_range(cls, value: int) -> int:
        if value is None:
            return value
        if not 1 <= value <= 4:
            raise ValueError("count must be between 1 and 4")
        return value


class GeneralMediaState(BaseModel):
    operation: Literal[
        "auto",
        "image_generation",
        "background_remove",
        "image_upscale",
        "image_edit",
        "remove_element",
        "restore",
        "outpaint",
        "resize",
        "video_generation",
    ] = "auto"
    model: str | None = None
    parameters: GeneralMediaParameters = Field(default_factory=GeneralMediaParameters)


class GeneralMediaRequest(BaseModel):
    user_id: int
    model_id: int
    selected_model_id: int
    conversation_uuid: str
    user_message: str
    state: GeneralMediaState = Field(default_factory=GeneralMediaState)
    debug: bool = False
    client_metadata: dict[str, Any] = Field(default_factory=dict)


class GeneralAudioParameters(BaseModel):
    # Provider/model-specific fields declared in ai_models.parameter_schema are
    # preserved here and validated dynamically in the router.
    model_config = ConfigDict(extra="allow")

    language: str | None = None
    include_segments: bool | None = None
    voice: str | None = None
    response_format: str | None = None
    speed: float | None = None
    provider_options: dict[str, Any] | None = None

    @field_validator("speed")
    @classmethod
    def speed_range(cls, value: float) -> float:
        if value is None:
            return value
        if not 0.25 <= value <= 4:
            raise ValueError("speed must be between 0.25 and 4")
        return value


class GeneralAudioState(BaseModel):
    operation: Literal["speech_to_text", "text_to_speech"]
    model: str | None = None
    provider: str | None = None
    parameters: GeneralAudioParameters = Field(default_factory=GeneralAudioParameters)


class GeneralAudioRequest(BaseModel):
    user_id: int
    model_id: int
    selected_model_id: int
    conversation_uuid: str
    user_message: str
    state: GeneralAudioState
    debug: bool = False
    client_metadata: dict[str, Any] = Field(default_factory=dict)


class GeneralToolResponse(BaseModel):
    success: bool = True
    type: str = "result"
    tool: str
    provider: str
    model: str
    model_tier: str | None = None
    user_id: int
    model_id: int
    selected_model_id: int
    conversation_uuid: str
    content: str | None = None
    files: list[GeneratedMediaFileInfo] = Field(default_factory=list)
    request_id: str
    usage: TokenUsage | None = None
    cost: CostUsage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    debug: dict[str, Any] | None = None
