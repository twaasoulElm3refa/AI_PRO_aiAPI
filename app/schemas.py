from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


RoleType = Literal["system", "user", "assistant"]
SearchMode = Literal["auto", "on", "off"]


class ChatMessage(BaseModel):
    role: RoleType
    content: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message content cannot be empty")
        return v


class TokenUsage(BaseModel):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class CostUsage(BaseModel):
    input_cost: Optional[float] = None
    output_cost: Optional[float] = None
    web_search_cost: Optional[float] = None
    total_cost: Optional[float] = None
    currency: str = "USD"


class TaskOptions(BaseModel):
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    history_limit: Optional[int] = None

    search_mode: SearchMode = "auto"
    enable_web_search: Optional[bool] = None
    force_no_search: bool = False

    web_search_max_results: Optional[int] = None
    web_search_max_total_results: Optional[int] = None

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if v < 0 or v > 2:
            raise ValueError("temperature must be between 0 and 2")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if v < 16:
            raise ValueError("max_tokens must be at least 16")
        if v > 12000:
            raise ValueError("max_tokens is too high")
        return v

    @model_validator(mode="after")
    def normalize_legacy_search_flags(self):
        if self.force_no_search:
            self.search_mode = "off"
        elif self.enable_web_search is True:
            self.search_mode = "on"
        elif self.enable_web_search is False and self.search_mode == "auto":
            self.search_mode = "off"
        return self


class TaskRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: str
    custom_prompt: Optional[str] = None
    response_language: Optional[str] = None
    debug: bool = False
    task_options: Optional[TaskOptions] = None
    client_metadata: Dict[str, Any] = Field(default_factory=dict)
    # Optional structured state for legacy one-shot tools: writer, summarizer, paraphraser.
    # This keeps old requests valid while allowing the frontend to send/receive variables.
    state: Optional[Dict[str, Any]] = None

    @field_validator("conversation_uuid")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("conversation_uuid cannot be empty")
        return v

    @field_validator("user_message")
    @classmethod
    def validate_user_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user_message cannot be empty")
        return v


class TaskResponse(BaseModel):
    reply: str
    task_key: str
    model_key: str
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    request_id: str
    debug: Optional[dict] = None
    usage: Optional[TokenUsage] = None
    cost: Optional[CostUsage] = None
    state: Optional[Dict[str, Any]] = None


# These are examples for the UI/prompt, not strict backend enums.
# Keep them as normal strings so the chat can accept any language, tone, goal, etc.
HeadlineContentType = str
HeadlineGoal = str
HeadlineLanguage = str
HeadlineTone = str
HeadlineLength = str

HeadlineChatType = Literal[
    "question",
    "result",
]


class HeadlineState(BaseModel):
    content: Optional[str] = None
    content_type: Optional[HeadlineContentType] = None
    goal: Optional[HeadlineGoal] = None
    language: Optional[HeadlineLanguage] = None
    tone: Optional[HeadlineTone] = None
    number_of_headlines: Optional[int] = None
    headline_length: Optional[HeadlineLength] = None
    extra_options: List[str] = Field(default_factory=list)

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = str(v).strip()
        return v or None

    @field_validator("content_type", "goal", "language", "tone", "headline_length", mode="before")
    @classmethod
    def normalize_optional_text_fields(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    @field_validator("number_of_headlines", mode="before")
    @classmethod
    def validate_number_of_headlines(cls, v) -> Optional[int]:
        if v is None or v == "":
            return None

        try:
            value = int(v)
        except (TypeError, ValueError):
            raise ValueError("number_of_headlines must be a positive integer")

        if value < 1:
            raise ValueError("number_of_headlines must be at least 1")

        # Not restricted to fixed choices. This cap only protects cost/performance.
        # Increase it if you want to allow larger batches.
        if value > 100:
            raise ValueError("number_of_headlines is too high; maximum is 100")

        return value

    @field_validator("extra_options", mode="before")
    @classmethod
    def normalize_extra_options(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = []
            for item in v:
                item = str(item).strip()
                if item and item not in cleaned:
                    cleaned.append(item)
            return cleaned
        item = str(v).strip()
        return [item] if item else []


class HeadlineChatRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: str
    state: Optional[HeadlineState] = None
    debug: bool = False
    client_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("conversation_uuid")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("conversation_uuid cannot be empty")
        return v

    @field_validator("user_message")
    @classmethod
    def validate_user_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user_message cannot be empty")
        return v


class HeadlineItem(BaseModel):
    id: int
    text: str
    subheadline: Optional[str] = None


class HeadlineChatResponse(BaseModel):
    success: bool = True
    type: HeadlineChatType
    tool: str = "ai_headline_generator"
    provider: str = "openrouter"
    model_key: str = "headline_fast"
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    message: str
    state: HeadlineState
    headlines: List[HeadlineItem] = Field(default_factory=list)
    count: int = 0
    request_id: str
    debug: Optional[dict] = None
    usage: Optional[TokenUsage] = None
    cost: Optional[CostUsage] = None


# These are examples for the UI/prompt, not strict backend enums.
# Keep them as normal strings so the chat can accept any language, tone, mode, etc.
ParaphraserLanguage = str
ParaphraserTone = str
ParaphraserRewriteMode = str
ParaphraserChangeLevel = str

ParaphraserChatType = Literal[
    "question",
    "result",
]


class ParaphraserState(BaseModel):
    # Keep all fields nullable so frontend can always send the same schema:
    # {"content": null, "language": null, ...}
    # Defaults are applied later in app/paraphraser_chat.py by normalize_paraphraser_state().
    content: Optional[str] = None
    language: Optional[str] = None
    tone: Optional[str] = None
    rewrite_mode: Optional[str] = None
    change_level: Optional[str] = None
    results_count: Optional[int] = None
    extra_options: List[str] = Field(default_factory=list)

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    @field_validator("language", "tone", "rewrite_mode", "change_level", mode="before")
    @classmethod
    def normalize_optional_text_fields(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    @field_validator("results_count", mode="before")
    @classmethod
    def validate_results_count(cls, v) -> Optional[int]:
        if v is None or v == "":
            return None

        try:
            value = int(v)
        except (TypeError, ValueError):
            raise ValueError("results_count must be a positive integer")

        if value < 1:
            raise ValueError("results_count must be at least 1")

        # Not restricted to fixed choices. This cap only protects cost/performance.
        if value > 20:
            raise ValueError("results_count is too high; maximum is 20")

        return value

    @field_validator("extra_options", mode="before")
    @classmethod
    def normalize_extra_options(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = []
            for item in v:
                item = str(item).strip()
                if item and item not in cleaned:
                    cleaned.append(item)
            return cleaned
        item = str(v).strip()
        return [item] if item else []


class ParaphraserChatRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: str
    state: Optional[ParaphraserState] = None
    debug: bool = False
    client_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("conversation_uuid")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("conversation_uuid cannot be empty")
        return v

    @field_validator("user_message")
    @classmethod
    def validate_user_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user_message cannot be empty")
        return v


class ParaphraserResultItem(BaseModel):
    id: int
    text: str


class ParaphraserChatResponse(BaseModel):
    success: bool = True
    type: ParaphraserChatType
    tool: str = "ai_paraphraser"
    provider: str = "openrouter"
    model_key: str = "paraphraser_fast"
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    message: str
    state: ParaphraserState
    results: List[ParaphraserResultItem] = Field(default_factory=list)
    count: int = 0
    request_id: str
    debug: Optional[dict] = None
    usage: Optional[TokenUsage] = None
    cost: Optional[CostUsage] = None

# Generic content chat tool responses.
# These tools intentionally keep fields nullable so the frontend can send a stable
# state object with null values. Backend defaults are applied in generic_content_chat.py.
ContentToolChatType = Literal[
    "question",
    "result",
]


class ContentToolResultItem(BaseModel):
    id: int
    text: str
    title: Optional[str] = None
    subject: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class SocialPostState(BaseModel):
    content: Optional[str] = None
    platform: Optional[str] = None
    language: Optional[str] = None
    tone: Optional[str] = None
    audience: Optional[str] = None
    goal: Optional[str] = None
    length: Optional[str] = None
    hashtag_count: Optional[int] = None
    include_emojis: Optional[bool] = None
    results_count: Optional[int] = None
    extra_options: List[str] = Field(default_factory=list)
    last_output: Optional[str] = None

    @field_validator("content", "platform", "language", "tone", "audience", "goal", "length", "last_output", mode="before")
    @classmethod
    def normalize_optional_text_fields(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    @field_validator("hashtag_count", "results_count", mode="before")
    @classmethod
    def validate_positive_ints(cls, v) -> Optional[int]:
        if v is None or v == "":
            return None
        try:
            value = int(v)
        except (TypeError, ValueError):
            raise ValueError("count fields must be positive integers")
        if value < 1:
            raise ValueError("count fields must be at least 1")
        if value > 100:
            raise ValueError("count fields are too high; maximum is 100")
        return value

    @field_validator("extra_options", mode="before")
    @classmethod
    def normalize_extra_options(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = []
            for item in v:
                item = str(item).strip()
                if item and item not in cleaned:
                    cleaned.append(item)
            return cleaned
        item = str(v).strip()
        return [item] if item else []


class SocialPostChatRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: str
    state: Optional[SocialPostState] = None
    debug: bool = False
    client_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("conversation_uuid")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("conversation_uuid cannot be empty")
        return v

    @field_validator("user_message")
    @classmethod
    def validate_user_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user_message cannot be empty")
        return v


class SocialPostChatResponse(BaseModel):
    success: bool = True
    type: ContentToolChatType
    tool: str = "ai_social_post_generator"
    provider: str = "openrouter"
    model_key: str = "social_post_generator"
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    message: str
    state: SocialPostState
    results: List[ContentToolResultItem] = Field(default_factory=list)
    count: int = 0
    request_id: str
    debug: Optional[dict] = None
    usage: Optional[TokenUsage] = None
    cost: Optional[CostUsage] = None


class EmailWriterState(BaseModel):
    purpose: Optional[str] = None
    email_type: Optional[str] = None
    recipient: Optional[str] = None
    sender_name: Optional[str] = None
    language: Optional[str] = None
    tone: Optional[str] = None
    length: Optional[str] = None
    subject_line: Optional[str] = None
    call_to_action: Optional[str] = None
    include_subject: Optional[bool] = None
    extra_options: List[str] = Field(default_factory=list)
    last_output: Optional[str] = None

    @field_validator("purpose", "email_type", "recipient", "sender_name", "language", "tone", "length", "subject_line", "call_to_action", "last_output", mode="before")
    @classmethod
    def normalize_optional_text_fields(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    @field_validator("extra_options", mode="before")
    @classmethod
    def normalize_extra_options(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = []
            for item in v:
                item = str(item).strip()
                if item and item not in cleaned:
                    cleaned.append(item)
            return cleaned
        item = str(v).strip()
        return [item] if item else []


class EmailWriterChatRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: str
    state: Optional[EmailWriterState] = None
    debug: bool = False
    client_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("conversation_uuid")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("conversation_uuid cannot be empty")
        return v

    @field_validator("user_message")
    @classmethod
    def validate_user_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user_message cannot be empty")
        return v


class EmailWriterChatResponse(BaseModel):
    success: bool = True
    type: ContentToolChatType
    tool: str = "ai_email_writer"
    provider: str = "openrouter"
    model_key: str = "email_writer"
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    message: str
    state: EmailWriterState
    results: List[ContentToolResultItem] = Field(default_factory=list)
    count: int = 0
    request_id: str
    debug: Optional[dict] = None
    usage: Optional[TokenUsage] = None
    cost: Optional[CostUsage] = None


class ScriptGeneratorState(BaseModel):
    topic: Optional[str] = None
    platform: Optional[str] = None
    duration: Optional[str] = None
    script_type: Optional[str] = None
    target_audience: Optional[str] = None
    tone: Optional[str] = None
    language: Optional[str] = None
    include_visual_details: Optional[bool] = None
    include_effects: Optional[bool] = None
    include_sound_effects: Optional[bool] = None
    include_camera_movements: Optional[bool] = None
    include_on_screen_text: Optional[bool] = None

    # Legacy/backward-compatible fields kept so old clients do not break.
    audience: Optional[str] = None
    format: Optional[str] = None
    include_scene_notes: Optional[bool] = None

    results_count: Optional[int] = None
    extra_options: List[str] = Field(default_factory=list)
    last_output: Optional[str] = None

    @field_validator(
        "topic",
        "platform",
        "duration",
        "script_type",
        "target_audience",
        "tone",
        "language",
        "audience",
        "format",
        "last_output",
        mode="before",
    )
    @classmethod
    def normalize_optional_text_fields(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    @model_validator(mode="after")
    def sync_legacy_fields(self):
        # target_audience is the new public field. audience is kept for older frontends.
        if self.target_audience is None and self.audience is not None:
            self.target_audience = self.audience
        if self.audience is None and self.target_audience is not None:
            self.audience = self.target_audience

        # include_visual_details replaces include_scene_notes but both remain accepted.
        if self.include_visual_details is None and self.include_scene_notes is not None:
            self.include_visual_details = self.include_scene_notes
        if self.include_scene_notes is None and self.include_visual_details is not None:
            self.include_scene_notes = self.include_visual_details
        return self

    @field_validator("results_count", mode="before")
    @classmethod
    def validate_results_count(cls, v) -> Optional[int]:
        if v is None or v == "":
            return None
        try:
            value = int(v)
        except (TypeError, ValueError):
            raise ValueError("results_count must be a positive integer")
        if value < 1:
            raise ValueError("results_count must be at least 1")
        if value > 20:
            raise ValueError("results_count is too high; maximum is 20")
        return value

    @field_validator("extra_options", mode="before")
    @classmethod
    def normalize_extra_options(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = []
            for item in v:
                item = str(item).strip()
                if item and item not in cleaned:
                    cleaned.append(item)
            return cleaned
        item = str(v).strip()
        return [item] if item else []

class ScriptGeneratorChatRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: str
    state: Optional[ScriptGeneratorState] = None
    debug: bool = False
    client_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("conversation_uuid")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("conversation_uuid cannot be empty")
        return v

    @field_validator("user_message")
    @classmethod
    def validate_user_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user_message cannot be empty")
        return v


class ScriptGeneratorChatResponse(BaseModel):
    success: bool = True
    type: ContentToolChatType
    tool: str = "ai_script_generator"
    provider: str = "openrouter"
    model_key: str = "script_generator"
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    message: str
    state: ScriptGeneratorState
    results: List[ContentToolResultItem] = Field(default_factory=list)
    count: int = 0
    request_id: str
    debug: Optional[dict] = None
    usage: Optional[TokenUsage] = None
    cost: Optional[CostUsage] = None


class ProductDescriptionState(BaseModel):
    product: Optional[str] = None
    brand_name: Optional[str] = None
    product_features: Optional[str] = None
    target_audience: Optional[str] = None
    language: Optional[str] = None
    tone: Optional[str] = None
    length: Optional[str] = None
    platform: Optional[str] = None
    include_bullets: Optional[bool] = None
    include_seo_keywords: Optional[bool] = None
    extra_options: List[str] = Field(default_factory=list)
    last_output: Optional[str] = None

    @field_validator("product", "brand_name", "product_features", "target_audience", "language", "tone", "length", "platform", "last_output", mode="before")
    @classmethod
    def normalize_optional_text_fields(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    @field_validator("extra_options", mode="before")
    @classmethod
    def normalize_extra_options(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = []
            for item in v:
                item = str(item).strip()
                if item and item not in cleaned:
                    cleaned.append(item)
            return cleaned
        item = str(v).strip()
        return [item] if item else []


class ProductDescriptionChatRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: str
    state: Optional[ProductDescriptionState] = None
    debug: bool = False
    client_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("conversation_uuid")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("conversation_uuid cannot be empty")
        return v

    @field_validator("user_message")
    @classmethod
    def validate_user_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user_message cannot be empty")
        return v


class ProductDescriptionChatResponse(BaseModel):
    success: bool = True
    type: ContentToolChatType
    tool: str = "ai_product_description_generator"
    provider: str = "openrouter"
    model_key: str = "product_description_generator"
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    message: str
    state: ProductDescriptionState
    results: List[ContentToolResultItem] = Field(default_factory=list)
    count: int = 0
    request_id: str
    debug: Optional[dict] = None
    usage: Optional[TokenUsage] = None
    cost: Optional[CostUsage] = None


class PromptGeneratorState(BaseModel):
    task: Optional[str] = None
    target_ai_tool: Optional[str] = None
    output_type: Optional[str] = None
    language: Optional[str] = None
    tone: Optional[str] = None
    audience: Optional[str] = None
    prompt_style: Optional[str] = None
    detail_level: Optional[str] = None
    include_constraints: Optional[bool] = None
    results_count: Optional[int] = None
    extra_options: List[str] = Field(default_factory=list)
    last_output: Optional[str] = None

    @field_validator("task", "target_ai_tool", "output_type", "language", "tone", "audience", "prompt_style", "detail_level", "last_output", mode="before")
    @classmethod
    def normalize_optional_text_fields(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    @field_validator("results_count", mode="before")
    @classmethod
    def validate_results_count(cls, v) -> Optional[int]:
        if v is None or v == "":
            return None
        try:
            value = int(v)
        except (TypeError, ValueError):
            raise ValueError("results_count must be a positive integer")
        if value < 1:
            raise ValueError("results_count must be at least 1")
        if value > 20:
            raise ValueError("results_count is too high; maximum is 20")
        return value

    @field_validator("extra_options", mode="before")
    @classmethod
    def normalize_extra_options(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = []
            for item in v:
                item = str(item).strip()
                if item and item not in cleaned:
                    cleaned.append(item)
            return cleaned
        item = str(v).strip()
        return [item] if item else []


class PromptGeneratorChatRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: str
    state: Optional[PromptGeneratorState] = None
    debug: bool = False
    client_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("conversation_uuid")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("conversation_uuid cannot be empty")
        return v

    @field_validator("user_message")
    @classmethod
    def validate_user_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user_message cannot be empty")
        return v


class PromptGeneratorChatResponse(BaseModel):
    success: bool = True
    type: ContentToolChatType
    tool: str = "ai_prompt_generator"
    provider: str = "openrouter"
    model_key: str = "prompt_generator"
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    message: str
    state: PromptGeneratorState
    results: List[ContentToolResultItem] = Field(default_factory=list)
    count: int = 0
    request_id: str
    debug: Optional[dict] = None
    usage: Optional[TokenUsage] = None
    cost: Optional[CostUsage] = None


class PromptEnhancerState(BaseModel):
    original_prompt: Optional[str] = None
    target_ai_tool: Optional[str] = None
    language: Optional[str] = None
    enhancement_goal: Optional[str] = None
    tone: Optional[str] = None
    output_format: Optional[str] = None
    detail_level: Optional[str] = None
    preserve_intent: Optional[bool] = None
    results_count: Optional[int] = None
    extra_options: List[str] = Field(default_factory=list)
    last_output: Optional[str] = None

    @field_validator("original_prompt", "target_ai_tool", "language", "enhancement_goal", "tone", "output_format", "detail_level", "last_output", mode="before")
    @classmethod
    def normalize_optional_text_fields(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    @field_validator("results_count", mode="before")
    @classmethod
    def validate_results_count(cls, v) -> Optional[int]:
        if v is None or v == "":
            return None
        try:
            value = int(v)
        except (TypeError, ValueError):
            raise ValueError("results_count must be a positive integer")
        if value < 1:
            raise ValueError("results_count must be at least 1")
        if value > 20:
            raise ValueError("results_count is too high; maximum is 20")
        return value

    @field_validator("extra_options", mode="before")
    @classmethod
    def normalize_extra_options(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = []
            for item in v:
                item = str(item).strip()
                if item and item not in cleaned:
                    cleaned.append(item)
            return cleaned
        item = str(v).strip()
        return [item] if item else []


class PromptEnhancerChatRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: str
    state: Optional[PromptEnhancerState] = None
    debug: bool = False
    client_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("conversation_uuid")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("conversation_uuid cannot be empty")
        return v

    @field_validator("user_message")
    @classmethod
    def validate_user_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user_message cannot be empty")
        return v


class PromptEnhancerChatResponse(BaseModel):
    success: bool = True
    type: ContentToolChatType
    tool: str = "ai_prompt_enhancer"
    provider: str = "openrouter"
    model_key: str = "prompt_enhancer"
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    message: str
    state: PromptEnhancerState
    results: List[ContentToolResultItem] = Field(default_factory=list)
    count: int = 0
    request_id: str
    debug: Optional[dict] = None
    usage: Optional[TokenUsage] = None
    cost: Optional[CostUsage] = None


class IdeaGeneratorState(BaseModel):
    topic: Optional[str] = None
    idea_type: Optional[str] = None
    industry: Optional[str] = None
    audience: Optional[str] = None
    language: Optional[str] = None
    tone: Optional[str] = None
    creativity_level: Optional[str] = None
    results_count: Optional[int] = None
    include_titles: Optional[bool] = None
    include_descriptions: Optional[bool] = None
    extra_options: List[str] = Field(default_factory=list)
    last_output: Optional[str] = None

    @field_validator("topic", "idea_type", "industry", "audience", "language", "tone", "creativity_level", "last_output", mode="before")
    @classmethod
    def normalize_optional_text_fields(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    @field_validator("results_count", mode="before")
    @classmethod
    def validate_results_count(cls, v) -> Optional[int]:
        if v is None or v == "":
            return None
        try:
            value = int(v)
        except (TypeError, ValueError):
            raise ValueError("results_count must be a positive integer")
        if value < 1:
            raise ValueError("results_count must be at least 1")
        if value > 50:
            raise ValueError("results_count is too high; maximum is 50")
        return value

    @field_validator("extra_options", mode="before")
    @classmethod
    def normalize_extra_options(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = []
            for item in v:
                item = str(item).strip()
                if item and item not in cleaned:
                    cleaned.append(item)
            return cleaned
        item = str(v).strip()
        return [item] if item else []


class IdeaGeneratorChatRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: str
    state: Optional[IdeaGeneratorState] = None
    debug: bool = False
    client_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("conversation_uuid")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("conversation_uuid cannot be empty")
        return v

    @field_validator("user_message")
    @classmethod
    def validate_user_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user_message cannot be empty")
        return v


class IdeaGeneratorChatResponse(BaseModel):
    success: bool = True
    type: ContentToolChatType
    tool: str = "ai_idea_generator"
    provider: str = "openrouter"
    model_key: str = "idea_generator"
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    message: str
    state: IdeaGeneratorState
    results: List[ContentToolResultItem] = Field(default_factory=list)
    count: int = 0
    request_id: str
    debug: Optional[dict] = None
    usage: Optional[TokenUsage] = None
    cost: Optional[CostUsage] = None


class HookGeneratorState(BaseModel):
    topic: Optional[str] = None
    platform: Optional[str] = None
    content_type: Optional[str] = None
    language: Optional[str] = None
    tone: Optional[str] = None
    audience: Optional[str] = None
    hook_style: Optional[str] = None
    length: Optional[str] = None
    results_count: Optional[int] = None
    extra_options: List[str] = Field(default_factory=list)
    last_output: Optional[str] = None

    @field_validator("topic", "platform", "content_type", "language", "tone", "audience", "hook_style", "length", "last_output", mode="before")
    @classmethod
    def normalize_optional_text_fields(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    @field_validator("results_count", mode="before")
    @classmethod
    def validate_results_count(cls, v) -> Optional[int]:
        if v is None or v == "":
            return None
        try:
            value = int(v)
        except (TypeError, ValueError):
            raise ValueError("results_count must be a positive integer")
        if value < 1:
            raise ValueError("results_count must be at least 1")
        if value > 50:
            raise ValueError("results_count is too high; maximum is 50")
        return value

    @field_validator("extra_options", mode="before")
    @classmethod
    def normalize_extra_options(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = []
            for item in v:
                item = str(item).strip()
                if item and item not in cleaned:
                    cleaned.append(item)
            return cleaned
        item = str(v).strip()
        return [item] if item else []


class HookGeneratorChatRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: str
    state: Optional[HookGeneratorState] = None
    debug: bool = False
    client_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("conversation_uuid")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("conversation_uuid cannot be empty")
        return v

    @field_validator("user_message")
    @classmethod
    def validate_user_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user_message cannot be empty")
        return v


class HookGeneratorChatResponse(BaseModel):
    success: bool = True
    type: ContentToolChatType
    tool: str = "ai_hook_generator"
    provider: str = "openrouter"
    model_key: str = "hook_generator"
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    message: str
    state: HookGeneratorState
    results: List[ContentToolResultItem] = Field(default_factory=list)
    count: int = 0
    request_id: str
    debug: Optional[dict] = None
    usage: Optional[TokenUsage] = None
    cost: Optional[CostUsage] = None


class KeywordGeneratorState(BaseModel):
    topic: Optional[str] = None
    industry: Optional[str] = None
    target_audience: Optional[str] = None
    language: Optional[str] = None
    keyword_type: Optional[str] = None
    search_intent: Optional[str] = None
    location: Optional[str] = None
    results_count: Optional[int] = None
    include_long_tail: Optional[bool] = None
    include_clusters: Optional[bool] = None
    extra_options: List[str] = Field(default_factory=list)
    last_output: Optional[str] = None

    @field_validator("topic", "industry", "target_audience", "language", "keyword_type", "search_intent", "location", "last_output", mode="before")
    @classmethod
    def normalize_optional_text_fields(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    @field_validator("results_count", mode="before")
    @classmethod
    def validate_results_count(cls, v) -> Optional[int]:
        if v is None or v == "":
            return None
        try:
            value = int(v)
        except (TypeError, ValueError):
            raise ValueError("results_count must be a positive integer")
        if value < 1:
            raise ValueError("results_count must be at least 1")
        if value > 100:
            raise ValueError("results_count is too high; maximum is 100")
        return value

    @field_validator("extra_options", mode="before")
    @classmethod
    def normalize_extra_options(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = []
            for item in v:
                item = str(item).strip()
                if item and item not in cleaned:
                    cleaned.append(item)
            return cleaned
        item = str(v).strip()
        return [item] if item else []


class KeywordGeneratorChatRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: str
    state: Optional[KeywordGeneratorState] = None
    debug: bool = False
    client_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("conversation_uuid")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("conversation_uuid cannot be empty")
        return v

    @field_validator("user_message")
    @classmethod
    def validate_user_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user_message cannot be empty")
        return v


class KeywordGeneratorChatResponse(BaseModel):
    success: bool = True
    type: ContentToolChatType
    tool: str = "ai_keyword_generator"
    provider: str = "openrouter"
    model_key: str = "keyword_generator"
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    message: str
    state: KeywordGeneratorState
    results: List[ContentToolResultItem] = Field(default_factory=list)
    count: int = 0
    request_id: str
    debug: Optional[dict] = None
    usage: Optional[TokenUsage] = None
    cost: Optional[CostUsage] = None


class MetaDescriptionState(BaseModel):
    content: Optional[str] = None
    page_title: Optional[str] = None
    primary_keyword: Optional[str] = None
    language: Optional[str] = None
    tone: Optional[str] = None
    length: Optional[str] = None
    max_characters: Optional[int] = None
    include_cta: Optional[bool] = None
    results_count: Optional[int] = None
    extra_options: List[str] = Field(default_factory=list)
    last_output: Optional[str] = None

    @field_validator("content", "page_title", "primary_keyword", "language", "tone", "length", "last_output", mode="before")
    @classmethod
    def normalize_optional_text_fields(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    @field_validator("results_count", "max_characters", mode="before")
    @classmethod
    def validate_positive_ints(cls, v) -> Optional[int]:
        if v is None or v == "":
            return None
        try:
            value = int(v)
        except (TypeError, ValueError):
            raise ValueError("count fields must be positive integers")
        if value < 1:
            raise ValueError("count fields must be at least 1")
        if value > 1000:
            raise ValueError("count fields are too high; maximum is 1000")
        return value

    @field_validator("extra_options", mode="before")
    @classmethod
    def normalize_extra_options(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = []
            for item in v:
                item = str(item).strip()
                if item and item not in cleaned:
                    cleaned.append(item)
            return cleaned
        item = str(v).strip()
        return [item] if item else []


class MetaDescriptionChatRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: str
    state: Optional[MetaDescriptionState] = None
    debug: bool = False
    client_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("conversation_uuid")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("conversation_uuid cannot be empty")
        return v

    @field_validator("user_message")
    @classmethod
    def validate_user_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user_message cannot be empty")
        return v


class MetaDescriptionChatResponse(BaseModel):
    success: bool = True
    type: ContentToolChatType
    tool: str = "ai_meta_description_generator"
    provider: str = "openrouter"
    model_key: str = "meta_description_generator"
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    message: str
    state: MetaDescriptionState
    results: List[ContentToolResultItem] = Field(default_factory=list)
    count: int = 0
    request_id: str
    debug: Optional[dict] = None
    usage: Optional[TokenUsage] = None
    cost: Optional[CostUsage] = None


class ContentAnalyzerState(BaseModel):
    content: Optional[str] = None
    analysis_goal: Optional[str] = None
    language: Optional[str] = None
    target_keyword: Optional[str] = None
    content_type: Optional[str] = None
    audience: Optional[str] = None
    checks: List[str] = Field(default_factory=list)
    detail_level: Optional[str] = None
    include_recommendations: Optional[bool] = None
    extra_options: List[str] = Field(default_factory=list)
    last_output: Optional[str] = None

    @field_validator("content", "analysis_goal", "language", "target_keyword", "content_type", "audience", "detail_level", "last_output", mode="before")
    @classmethod
    def normalize_optional_text_fields(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    @field_validator("checks", "extra_options", mode="before")
    @classmethod
    def normalize_text_list(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = []
            for item in v:
                item = str(item).strip()
                if item and item not in cleaned:
                    cleaned.append(item)
            return cleaned
        item = str(v).strip()
        return [item] if item else []


class ContentAnalyzerChatRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: str
    state: Optional[ContentAnalyzerState] = None
    debug: bool = False
    client_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("conversation_uuid")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("conversation_uuid cannot be empty")
        return v

    @field_validator("user_message")
    @classmethod
    def validate_user_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user_message cannot be empty")
        return v


class ContentAnalyzerChatResponse(BaseModel):
    success: bool = True
    type: ContentToolChatType
    tool: str = "ai_content_analyzer"
    provider: str = "openrouter"
    model_key: str = "content_analyzer"
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    message: str
    state: ContentAnalyzerState
    results: List[ContentToolResultItem] = Field(default_factory=list)
    count: int = 0
    request_id: str
    debug: Optional[dict] = None
    usage: Optional[TokenUsage] = None
    cost: Optional[CostUsage] = None


class ContentOptimizerState(BaseModel):
    content: Optional[str] = None
    optimization_goal: Optional[str] = None
    primary_keyword: Optional[str] = None
    secondary_keywords: List[str] = Field(default_factory=list)
    language: Optional[str] = None
    tone: Optional[str] = None
    content_type: Optional[str] = None
    audience: Optional[str] = None
    seo_level: Optional[str] = None
    preserve_meaning: Optional[bool] = None
    include_explanation: Optional[bool] = None
    extra_options: List[str] = Field(default_factory=list)
    last_output: Optional[str] = None

    @field_validator("content", "optimization_goal", "primary_keyword", "language", "tone", "content_type", "audience", "seo_level", "last_output", mode="before")
    @classmethod
    def normalize_optional_text_fields(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    @field_validator("secondary_keywords", "extra_options", mode="before")
    @classmethod
    def normalize_text_list(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = []
            for item in v:
                item = str(item).strip()
                if item and item not in cleaned:
                    cleaned.append(item)
            return cleaned
        item = str(v).strip()
        return [item] if item else []


class ContentOptimizerChatRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: str
    state: Optional[ContentOptimizerState] = None
    debug: bool = False
    client_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("conversation_uuid")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("conversation_uuid cannot be empty")
        return v

    @field_validator("user_message")
    @classmethod
    def validate_user_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user_message cannot be empty")
        return v


class ContentOptimizerChatResponse(BaseModel):
    success: bool = True
    type: ContentToolChatType
    tool: str = "ai_content_optimizer"
    provider: str = "openrouter"
    model_key: str = "content_optimizer"
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    message: str
    state: ContentOptimizerState
    results: List[ContentToolResultItem] = Field(default_factory=list)
    count: int = 0
    request_id: str
    debug: Optional[dict] = None
    usage: Optional[TokenUsage] = None
    cost: Optional[CostUsage] = None



class AIDetectorState(BaseModel):
    content: Optional[str] = None
    language: Optional[str] = None
    analysis_depth: Optional[str] = None
    detection_focus: Optional[str] = None
    include_score: Optional[bool] = None
    include_evidence: Optional[bool] = None
    include_rewrite_tips: Optional[bool] = None
    extra_options: List[str] = Field(default_factory=list)
    last_output: Optional[str] = None

    @field_validator("content", "language", "analysis_depth", "detection_focus", "last_output", mode="before")
    @classmethod
    def normalize_optional_text_fields(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    @field_validator("extra_options", mode="before")
    @classmethod
    def normalize_extra_options(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = []
            for item in v:
                item = str(item).strip()
                if item and item not in cleaned:
                    cleaned.append(item)
            return cleaned
        item = str(v).strip()
        return [item] if item else []


class AIDetectorChatRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: str
    state: Optional[AIDetectorState] = None
    debug: bool = False
    client_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("conversation_uuid")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("conversation_uuid cannot be empty")
        return v

    @field_validator("user_message")
    @classmethod
    def validate_user_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user_message cannot be empty")
        return v


class AIDetectorChatResponse(BaseModel):
    success: bool = True
    type: ContentToolChatType
    tool: str = "ai_detector"
    provider: str = "openrouter"
    model_key: str = "ai_detector"
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    message: str
    state: AIDetectorState
    results: List[ContentToolResultItem] = Field(default_factory=list)
    count: int = 0
    request_id: str
    debug: Optional[dict] = None
    usage: Optional[TokenUsage] = None
    cost: Optional[CostUsage] = None


class AIHumanizerState(BaseModel):
    content: Optional[str] = None
    language: Optional[str] = None
    tone: Optional[str] = None
    audience: Optional[str] = None
    humanize_level: Optional[str] = None
    preserve_meaning: Optional[bool] = None
    preserve_keywords: Optional[bool] = None
    results_count: Optional[int] = None
    extra_options: List[str] = Field(default_factory=list)
    last_output: Optional[str] = None

    @field_validator("content", "language", "tone", "audience", "humanize_level", "last_output", mode="before")
    @classmethod
    def normalize_optional_text_fields(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    @field_validator("results_count", mode="before")
    @classmethod
    def validate_results_count(cls, v) -> Optional[int]:
        if v is None or v == "":
            return None
        try:
            value = int(v)
        except (TypeError, ValueError):
            raise ValueError("results_count must be a positive integer")
        if value < 1:
            raise ValueError("results_count must be at least 1")
        if value > 20:
            raise ValueError("results_count is too high; maximum is 20")
        return value

    @field_validator("extra_options", mode="before")
    @classmethod
    def normalize_extra_options(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = []
            for item in v:
                item = str(item).strip()
                if item and item not in cleaned:
                    cleaned.append(item)
            return cleaned
        item = str(v).strip()
        return [item] if item else []


class AIHumanizerChatRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: str
    state: Optional[AIHumanizerState] = None
    debug: bool = False
    client_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("conversation_uuid")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("conversation_uuid cannot be empty")
        return v

    @field_validator("user_message")
    @classmethod
    def validate_user_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user_message cannot be empty")
        return v


class AIHumanizerChatResponse(BaseModel):
    success: bool = True
    type: ContentToolChatType
    tool: str = "ai_humanizer"
    provider: str = "openrouter"
    model_key: str = "ai_humanizer"
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    message: str
    state: AIHumanizerState
    results: List[ContentToolResultItem] = Field(default_factory=list)
    count: int = 0
    request_id: str
    debug: Optional[dict] = None
    usage: Optional[TokenUsage] = None
    cost: Optional[CostUsage] = None


class BusinessNameState(BaseModel):
    business_idea: Optional[str] = None
    industry: Optional[str] = None
    target_audience: Optional[str] = None
    language: Optional[str] = None
    tone: Optional[str] = None
    name_style: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    avoid_words: List[str] = Field(default_factory=list)
    results_count: Optional[int] = None
    include_slogans: Optional[bool] = None
    include_domain_ideas: Optional[bool] = None
    extra_options: List[str] = Field(default_factory=list)
    last_output: Optional[str] = None

    @field_validator("business_idea", "industry", "target_audience", "language", "tone", "name_style", "last_output", mode="before")
    @classmethod
    def normalize_optional_text_fields(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    @field_validator("keywords", "avoid_words", "extra_options", mode="before")
    @classmethod
    def normalize_text_list(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = []
            for item in v:
                item = str(item).strip()
                if item and item not in cleaned:
                    cleaned.append(item)
            return cleaned
        item = str(v).strip()
        return [item] if item else []

    @field_validator("results_count", mode="before")
    @classmethod
    def validate_results_count(cls, v) -> Optional[int]:
        if v is None or v == "":
            return None
        try:
            value = int(v)
        except (TypeError, ValueError):
            raise ValueError("results_count must be a positive integer")
        if value < 1:
            raise ValueError("results_count must be at least 1")
        if value > 100:
            raise ValueError("results_count is too high; maximum is 100")
        return value


class BusinessNameChatRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: str
    state: Optional[BusinessNameState] = None
    debug: bool = False
    client_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("conversation_uuid")
    @classmethod
    def validate_uuid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("conversation_uuid cannot be empty")
        return v

    @field_validator("user_message")
    @classmethod
    def validate_user_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("user_message cannot be empty")
        return v


class BusinessNameChatResponse(BaseModel):
    success: bool = True
    type: ContentToolChatType
    tool: str = "business_name_generator"
    provider: str = "openrouter"
    model_key: str = "business_name_generator"
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    message: str
    state: BusinessNameState
    results: List[ContentToolResultItem] = Field(default_factory=list)
    count: int = 0
    request_id: str
    debug: Optional[dict] = None
    usage: Optional[TokenUsage] = None
    cost: Optional[CostUsage] = None


class ResumeBuilderState(BaseModel):
    target_role: Optional[str] = None
    candidate_name: Optional[str] = None
    language: Optional[str] = None
    tone: Optional[str] = None
    experience_level: Optional[str] = None
    resume_style: Optional[str] = None
    output_format: Optional[str] = None
    sections_to_include: List[str] = Field(default_factory=list)
    extra_options: List[str] = Field(default_factory=list)
    last_output: Optional[str] = None

    @field_validator("target_role", "candidate_name", "language", "tone", "experience_level", "resume_style", "output_format", "last_output", mode="before")
    @classmethod
    def normalize_optional_text_fields(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    @field_validator("sections_to_include", "extra_options", mode="before")
    @classmethod
    def normalize_text_list(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            cleaned = []
            for item in v:
                item = str(item).strip()
                if item and item not in cleaned:
                    cleaned.append(item)
            return cleaned
        item = str(v).strip()
        return [item] if item else []


class GeneratedFileInfo(BaseModel):
    file_id: str
    filename: str
    content_type: str = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    download_url: str


class ResumeBuilderChatResponse(BaseModel):
    success: bool = True
    type: ContentToolChatType = "result"
    tool: str = "resume_builder_ai"
    provider: str = "openrouter"
    model_key: str = "resume_builder"
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    message: str
    state: ResumeBuilderState
    results: List[ContentToolResultItem] = Field(default_factory=list)
    count: int = 0
    file: Optional[GeneratedFileInfo] = None
    request_id: str
    debug: Optional[dict] = None
    usage: Optional[TokenUsage] = None
    cost: Optional[CostUsage] = None



class GeneratedMediaFileInfo(BaseModel):
    file_id: str
    filename: str
    content_type: str
    download_url: str
    size_bytes: int


class MediaToolResponse(BaseModel):
    success: bool = True
    type: str = "result"
    tool: str
    provider: str
    model: str
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    message: str
    files: List[GeneratedMediaFileInfo] = Field(default_factory=list)
    count: int = 0
    request_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    debug: Optional[dict] = None
    usage: Optional[TokenUsage] = None
    cost: Optional[CostUsage] = None


class ImageGeneratorState(BaseModel):
    provider: Optional[str] = None
    negative_prompt: Optional[str] = None
    enhance_prompt: bool = True
    size: str = "1024x1024"
    quality: str = "medium"
    results_count: int = 1
    output_format: str = "png"
    seed: Optional[int] = None
    extra_options: List[str] = Field(default_factory=list)
    last_output: Optional[str] = None


class ImageGeneratorRequest(BaseModel):
    """
    Preferred request shape:
      user_id, sub_tool_id, conversation_uuid, user_message, state, debug

    Legacy request fields (prompt/provider/size/...) are still accepted so
    existing callers keep working without changes.
    """
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: Optional[str] = None
    state: ImageGeneratorState = Field(default_factory=ImageGeneratorState)
    debug: bool = False
    client_metadata: Dict[str, Any] = Field(default_factory=dict)

    # Backward-compatible legacy fields used internally by media_tools.py.
    prompt: Optional[str] = None
    provider: Optional[str] = None
    # Optional provider model override. General-media validates this against its
    # server-side allowlist before passing it here.
    model: Optional[str] = None
    # Trusted server-side model capabilities, populated from ai_models.
    provider_metadata: Dict[str, Any] = Field(default_factory=dict)
    negative_prompt: Optional[str] = None
    enhance_prompt: Optional[bool] = None
    size: Optional[str] = None
    quality: Optional[str] = None
    count: Optional[int] = None
    output_format: Optional[str] = None
    seed: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_generic_or_legacy_shape(cls, raw):
        data = dict(raw or {})
        state = dict(data.get("state") or {})

        prompt = data.get("prompt") or data.get("user_message")
        if prompt is not None:
            data["prompt"] = prompt
            data["user_message"] = data.get("user_message") or prompt

        mappings = {
            "provider": "provider",
            "negative_prompt": "negative_prompt",
            "enhance_prompt": "enhance_prompt",
            "size": "size",
            "quality": "quality",
            "output_format": "output_format",
            "seed": "seed",
        }
        for target, state_key in mappings.items():
            if data.get(target) is None and state.get(state_key) is not None:
                data[target] = state.get(state_key)

        if data.get("count") is None:
            data["count"] = state.get("results_count", state.get("count", 1))

        data.setdefault("enhance_prompt", True)
        data.setdefault("size", "1024x1024")
        data.setdefault("quality", "medium")
        data.setdefault("count", 1)
        data.setdefault("output_format", "png")
        return data

    @field_validator("conversation_uuid", "prompt")
    @classmethod
    def validate_required_text(cls, v: Optional[str]) -> str:
        value = str(v or "").strip()
        if not value:
            raise ValueError("value cannot be empty")
        return value

    @field_validator("provider", "negative_prompt", mode="before")
    @classmethod
    def normalize_optional_text(cls, v):
        if v is None:
            return None
        value = str(v).strip()
        return value or None

    @field_validator("size")
    @classmethod
    def validate_size(cls, v: Optional[str]) -> str:
        import re
        value = str(v or "1024x1024").strip().lower()
        if value != "auto" and not re.fullmatch(r"\d{2,5}x\d{2,5}", value):
            raise ValueError("size must be auto or WIDTHxHEIGHT, for example 1024x1024")
        return value

    @field_validator("count")
    @classmethod
    def validate_count(cls, v: Optional[int]) -> int:
        value = int(v or 1)
        if value < 1 or value > 4:
            raise ValueError("count must be between 1 and 4")
        return value

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: Optional[str]) -> str:
        value = str(v or "png").strip().lower()
        if value not in {"png", "jpg", "jpeg", "webp"}:
            raise ValueError("output_format must be png, jpg, jpeg, or webp")
        return value


class MediaUploadRequest(BaseModel):
    """
    JSON envelope sent inside the multipart/form-data field named `payload`.
    The binary file is sent separately in the `file` field.
    """
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: str
    state: Dict[str, Any] = Field(default_factory=dict)
    debug: bool = False

    @field_validator("conversation_uuid", "user_message")
    @classmethod
    def validate_required_text(cls, v: str) -> str:
        value = str(v).strip()
        if not value:
            raise ValueError("value cannot be empty")
        return value


class YouTubeSummarizerState(BaseModel):
    transcript_languages: List[str] = Field(default_factory=lambda: ["ar", "en"])
    summary_language: Optional[str] = None
    summary_style: str = "concise structured summary"
    max_summary_words: int = 500
    extra_options: List[str] = Field(default_factory=list)
    last_output: Optional[str] = None


class YouTubeSummarizerRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: Optional[str] = None
    state: YouTubeSummarizerState = Field(default_factory=YouTubeSummarizerState)
    debug: bool = False
    client_metadata: Dict[str, Any] = Field(default_factory=dict)

    # Legacy request fields retained for backward compatibility.
    youtube_url: Optional[str] = None
    transcript_languages: List[str] = Field(default_factory=list)
    summary_language: Optional[str] = None
    summary_style: Optional[str] = None
    max_summary_words: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_generic_or_legacy_shape(cls, raw):
        data = dict(raw or {})
        state = dict(data.get("state") or {})

        url = data.get("youtube_url") or data.get("user_message")
        if url is not None:
            data["youtube_url"] = url
            data["user_message"] = data.get("user_message") or url

        if not data.get("transcript_languages"):
            data["transcript_languages"] = state.get("transcript_languages") or ["ar", "en"]
        if data.get("summary_language") is None:
            data["summary_language"] = state.get("summary_language")
        if data.get("summary_style") is None:
            data["summary_style"] = state.get("summary_style") or "concise structured summary"
        if data.get("max_summary_words") is None:
            data["max_summary_words"] = state.get("max_summary_words", 500)
        return data

    @field_validator("conversation_uuid", "youtube_url", "summary_style")
    @classmethod
    def validate_required_text(cls, v: Optional[str]) -> str:
        value = str(v or "").strip()
        if not value:
            raise ValueError("value cannot be empty")
        return value

    @field_validator("transcript_languages", mode="before")
    @classmethod
    def normalize_languages(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = v.split(",")
        return [str(item).strip() for item in v if str(item).strip()]

    @field_validator("summary_language", mode="before")
    @classmethod
    def normalize_summary_language(cls, v):
        if v is None:
            return None
        value = str(v).strip()
        return value or None

    @field_validator("max_summary_words")
    @classmethod
    def validate_max_summary_words(cls, v: Optional[int]) -> int:
        value = int(v or 500)
        if value < 50 or value > 2000:
            raise ValueError("max_summary_words must be between 50 and 2000")
        return value


class YouTubeSummarizerResponse(BaseModel):
    success: bool = True
    type: str = "result"
    tool: str = "youtube_summarizer"
    provider: str = "youtube-transcript-api+openrouter"
    model_key: str = "youtube_summarizer"
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    summary: str
    video_id: str
    transcript_language: Optional[str] = None
    transcript_chars: int
    transcript_segments: int
    transcript_is_generated: bool = False
    request_id: str
    debug: Optional[dict] = None
    usage: Optional[TokenUsage] = None
    cost: Optional[CostUsage] = None


class SpeechToTextResponse(BaseModel):
    success: bool = True
    type: str = "result"
    tool: str = "speech_to_text"
    provider: str
    model: str
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    transcript: str
    detected_language: Optional[str] = None
    duration_seconds: Optional[float] = None
    segments: Optional[List[Dict[str, Any]]] = None
    request_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TextToSpeechState(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    voice: Optional[str] = None
    response_format: Optional[str] = None
    speed: float = 1.0
    provider_options: Dict[str, Any] = Field(default_factory=dict)
    extra_options: List[str] = Field(default_factory=list)
    last_output: Optional[str] = None


class TextToSpeechRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    user_message: Optional[str] = None
    state: TextToSpeechState = Field(default_factory=TextToSpeechState)
    debug: bool = False
    client_metadata: Dict[str, Any] = Field(default_factory=dict)

    # Legacy fields retained for backward compatibility.
    text: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    voice: Optional[str] = None
    response_format: Optional[str] = None
    speed: Optional[float] = None
    provider_options: Optional[Dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_generic_or_legacy_shape(cls, raw):
        data = dict(raw or {})
        state = dict(data.get("state") or {})

        text = data.get("text") or data.get("user_message")
        if text is not None:
            data["text"] = text
            data["user_message"] = data.get("user_message") or text

        for field_name in ("provider", "model", "voice", "response_format", "speed", "provider_options"):
            if data.get(field_name) is None and state.get(field_name) is not None:
                data[field_name] = state.get(field_name)

        data.setdefault("speed", 1.0)
        return data

    @field_validator("conversation_uuid", "text")
    @classmethod
    def validate_required_text(cls, v: Optional[str]) -> str:
        value = str(v or "").strip()
        if not value:
            raise ValueError("value cannot be empty")
        return value

    @field_validator("provider", "model", "voice", "response_format", mode="before")
    @classmethod
    def normalize_optional_text(cls, v):
        if v is None:
            return None
        value = str(v).strip()
        return value or None

    @field_validator("speed")
    @classmethod
    def validate_speed(cls, v: Optional[float]) -> float:
        value = float(v or 1.0)
        if value < 0.25 or value > 4.0:
            raise ValueError("speed must be between 0.25 and 4.0")
        return value


class ConversationCreateRequest(BaseModel):
    user_id: int
    sub_tool_id: int
    conversation_uuid: str
    is_pinned: bool = False
    is_archived: bool = False


class ConversationResponse(BaseModel):
    id: int
    user_id: int
    sub_tool_id: int
    uuid: str
    is_pinned: bool
    is_archived: bool
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class MessageResponse(BaseModel):
    id: int
    conversation_id: int
    role: str
    content: str
    is_error: bool
    created_at: Optional[str] = None
