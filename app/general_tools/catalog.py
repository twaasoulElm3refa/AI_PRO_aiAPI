from __future__ import annotations

from typing import Any

from app.settings import get_settings


def _text_parameters(tool: str) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "temperature": {"type": "number", "minimum": 0, "maximum": 2, "default": 0.5, "required": False},
        "max_tokens": {"type": "integer", "minimum": 64, "maximum": 12000, "default": 4000, "required": False},
        "quality_mode": {"type": "enum", "values": ["fast", "balanced", "high"], "default": "balanced"},
        "reasoning_effort": {"type": "enum", "values": ["none", "minimal", "low", "medium", "high"], "default": "none"},
        "language": {"type": "string", "nullable": True},
    }
    if tool == "general_code":
        parameters.update({
            "task_mode": {"type": "enum", "values": ["generate", "explain", "debug", "review", "refactor", "test"], "default": "generate"},
            "programming_language": {"type": "string", "nullable": True},
            "include_explanation": {"type": "boolean", "default": True},
            "include_tests": {"type": "boolean", "default": False},
        })
    if tool == "general_translation":
        parameters.update({
            "source_language": {"type": "string", "default": "auto"},
            "target_language": {"type": "string", "required": True},
            "preserve_formatting": {"type": "boolean", "default": True},
        })
    return parameters


def build_catalog() -> dict[str, Any]:
    return {
        "enabled": get_settings().GENERAL_TOOLS_ENABLED,
        "pricing_notice": "الأسعار المعروضة تقديرية؛ الخصم يعتمد على التكلفة الفعلية من المزود.",
        "tiers": [
            {"id": "free", "label": "مجاني", "description": "قد يخضع لحدود أو ازدحام"},
            {"id": "standard", "label": "قياسي", "description": "أفضل توازن بين الجودة والتكلفة"},
            {"id": "advanced", "label": "متقدم", "description": "جودة أعلى وتكلفة أكبر"},
        ],
        "tools": [
            {"key": "general_chat", "label": "محادثة وكتابة", "operations": ["text_generation"], "models": [], "parameters": _text_parameters("general_chat")},
            {"key": "general_code", "label": "برمجة وتقنية", "operations": ["text_generation"], "models": [], "parameters": _text_parameters("general_code")},
            {"key": "general_translation", "label": "ترجمة", "operations": ["text_generation"], "models": [], "parameters": _text_parameters("general_translation")},
            {
                "key": "general_media", "label": "صور وفيديو",
                "task_selection": "prompt",
                "task_detection_endpoint": "/tasks/general-media/detect-task",
                "execution_endpoint": "/tasks/general-media",
                "provider": "runware", "canonical_content_type": "multipart/form-data",
                "file_optional": True,
                "operations": [
                    "image_generation", "background_remove", "image_upscale",
                    "image_edit", "remove_element", "restore", "outpaint",
                    "resize", "video_generation",
                ],
                "models": [],
                "requires_file": {
                    "image_generation": False, "background_remove": True,
                    "image_upscale": True, "image_edit": True,
                    "remove_element": True, "restore": True, "outpaint": True,
                    "resize": True, "video_generation": False,
                },
                "requires_mask": {"remove_element": True},
                "parameters_source": "selected_model.parameter_schema",
                "parameters": {},
            },
            {"key": "general_audio", "label": "صوت وصوتيات", "operations": ["speech_to_text", "text_to_speech"], "models": [], "parameters": {"language": {"type": "string", "nullable": True}, "include_segments": {"type": "boolean"}, "voice": {"type": "string"}, "response_format": {"type": "enum", "values": ["mp3", "pcm"]}, "speed": {"type": "number", "minimum": 0.25, "maximum": 4}}},
        ],
    }
