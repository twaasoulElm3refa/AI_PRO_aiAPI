from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class TrendParameters(BaseModel):
    model_config = ConfigDict(extra="allow")

    width: int | None = None
    height: int | None = None
    output_format: str | None = None
    seed: int | None = None


class TrendState(BaseModel):
    parameters: TrendParameters = Field(default_factory=TrendParameters)


class TrendRequest(BaseModel):
    user_id: int
    # Legacy-tool conversation key. model_id is accepted temporarily so old
    # clients receive a smooth migration, but new clients must send sub_tool_id.
    sub_tool_id: int = Field(validation_alias=AliasChoices("sub_tool_id", "model_id"))
    selected_model_id: int
    conversation_uuid: str
    user_message: str = ""
    state: TrendState = Field(default_factory=TrendState)
    debug: bool = False
    client_metadata: dict[str, Any] = Field(default_factory=dict)
