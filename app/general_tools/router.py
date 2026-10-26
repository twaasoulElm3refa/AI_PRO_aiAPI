from __future__ import annotations

import json
import math

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from sqlalchemy.orm import Session

from app.billing import (
    BillingAuthorization, authorize_charge, capture_charge,
    estimate_model_reservation, mark_charge_started,
    mark_reconciliation_required, release_charge,
)
from app.billing.errors import BillingError
from app.database import get_db
from app.crud import get_ai_model, get_existing_conversation_for_model, list_ai_models
from app.media_tools import (
    run_background_remover,
    run_general_runware_media_operation,
    run_image_generator,
    run_image_upscaler,
    run_speech_to_text,
    run_text_to_speech,
)
from app.schemas import CostUsage, ImageGeneratorRequest, TextToSpeechRequest
from app.security import verify_internal_api_key
from app.settings import get_settings

from .catalog import build_catalog
from .model_catalog import serialize_ai_model, sync_openrouter_models
from .schemas import (
    GeneralAudioRequest,
    GeneralMediaRequest,
    GeneralTextRequest,
    GeneralToolResponse,
)
from .service import run_general_text


router = APIRouter(prefix="/tasks", tags=["general tools"])


def _error(exc: Exception) -> HTTPException:
    if isinstance(exc, BillingError):
        return HTTPException(status_code=exc.status_code, detail=exc.as_detail())
    return HTTPException(status_code=400 if isinstance(exc, ValueError) else 502, detail=str(exc))


def _detect_media_operation(prompt: str, *, has_file: bool) -> str:
    """Route the currently implemented media tasks from the user's prompt."""
    text = str(prompt or "").strip().casefold()
    background_terms = (
        "remove background", "background remove", "transparent background",
        "delete background", "احذف الخلفية", "حذف الخلفية", "ازالة الخلفية",
        "إزالة الخلفية", "شيل الخلفية", "خلفية شفافة",
    )
    upscale_terms = (
        "upscale", "increase resolution", "enhance quality", "improve quality",
        "sharpen", "restore photo", "high resolution", "تحسين الجودة",
        "حسن الجودة", "حسّن الجودة", "زيادة الدقة", "كبر الصورة",
        "تكبير الصورة", "وضح الصورة", "توضيح الصورة", "استعادة الصورة",
    )
    generation_terms = (
        "generate image", "create image", "make an image", "draw", "text to image",
        "أنشئ صورة", "انشئ صورة", "إنشاء صورة", "انشاء صورة", "اصنع صورة",
        "ارسم", "ولد صورة", "ولّد صورة", "توليد صورة", "صمم صورة", "صمّم صورة",
    )
    video_terms = (
        "generate video", "create video", "text to video", "image to video",
        "animate image", "أنشئ فيديو", "انشئ فيديو", "توليد فيديو",
        "حول الصورة لفيديو", "حوّل الصورة لفيديو", "حرك الصورة", "حرّك الصورة",
    )
    remove_element_terms = (
        "remove object", "remove element", "erase object", "delete object",
        "احذف العنصر", "احذف الشخص", "امسح العنصر", "إزالة عنصر", "ازالة عنصر",
    )
    outpaint_terms = (
        "outpaint", "extend image", "expand image", "extend canvas",
        "وسع الصورة", "وسّع الصورة", "مدد الصورة", "مدّد الصورة", "توسيع الصورة",
    )
    resize_terms = (
        "resize image", "change dimensions", "change aspect ratio",
        "غير ابعاد", "غيّر أبعاد", "تغيير الأبعاد", "غير مقاس", "غيّر مقاس",
    )
    restore_terms = (
        "restore old photo", "repair photo", "fix damaged photo", "denoise photo",
        "رمم الصورة", "رمّم الصورة", "ترميم الصورة", "اصلح الصورة القديمة",
        "إصلاح الصورة القديمة", "ازالة التشويش", "إزالة التشويش",
    )
    edit_terms = (
        "edit image", "change image", "replace in image", "recolor",
        "عدل الصورة", "عدّل الصورة", "تعديل الصورة", "غير لون", "غيّر لون",
        "استبدل", "غير الخلفية", "غيّر الخلفية",
    )
    if any(term in text for term in video_terms):
        return "video_generation"
    if any(term in text for term in background_terms):
        return "background_remove"
    if any(term in text for term in remove_element_terms):
        return "remove_element"
    if any(term in text for term in outpaint_terms):
        return "outpaint"
    if any(term in text for term in resize_terms):
        return "resize"
    if any(term in text for term in restore_terms):
        return "restore"
    if any(term in text for term in upscale_terms):
        return "image_upscale"
    if any(term in text for term in generation_terms):
        return "image_generation"
    if has_file and any(term in text for term in edit_terms):
        return "image_edit"
    if not has_file:
        return "image_generation"
    raise ValueError(
        "Could not infer the media task from the prompt. Describe whether you "
        "want generation, editing, object removal, restoration, outpainting, "
        "resizing, upscaling, background removal, or video generation."
    )


def _is_real_upload(value) -> bool:
    """Postman may send an empty file part; do not treat it as an upload."""
    if value is None or not hasattr(value, "read"):
        return False
    return bool(str(getattr(value, "filename", "") or "").strip())


@router.post("/general-media/detect-task")
async def detect_general_media_task(
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
):
    try:
        content_type = (request.headers.get("content-type") or "").lower()
        uploaded_file = None
        mask_file = None
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            raw_payload = form.get("payload")
            if not raw_payload:
                raise ValueError("payload is required")
            payload = json.loads(str(raw_payload))
            uploaded_file = form.get("file")
        else:
            payload = await request.json()
        prompt = str(payload.get("user_message") or "").strip()
        if not prompt:
            raise ValueError("user_message is required")
        has_file = _is_real_upload(uploaded_file) or bool(
            (payload.get("client_metadata") or {}).get("has_file")
        )
        operation = _detect_media_operation(prompt, has_file=has_file)
        rows, _ = list_ai_models(
            db, tool_key="general_media", operation=operation,
            provider="runware", page=1, per_page=100,
        )
        return {
            "success": True,
            "detected_task": operation,
            "requires_file": operation not in {"image_generation", "video_generation"},
            "requires_mask": operation == "remove_element",
            "models": [serialize_ai_model(row) for row in rows],
        }
    except Exception as exc:
        raise _error(exc) from exc


def _resolve_model_parameters(db_model, submitted, *, operation: str) -> dict:
    """Apply per-model DB defaults and validate model-specific controls."""
    schema = db_model.parameter_schema or {}
    values = submitted.model_dump(exclude_none=True)
    if operation == "speech_to_text":
        common = {"language", "include_segments"}
    elif operation == "text_to_speech":
        common = {"voice", "response_format", "speed", "provider_options"}
    else:
        common = {
            "size", "quality", "count", "output_format", "negative_prompt",
            "enhance_prompt", "seed", "scale", "face_enhance", "quality_mode",
            "enhance_details", "realism", "width", "height", "duration",
            "outpaint_top", "outpaint_right", "outpaint_bottom", "outpaint_left",
            "auto_crop",
        }
    allowed = set(schema) if schema else common
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(
            f"Parameters not supported by selected model: {', '.join(unknown)}"
        )

    for name, definition in schema.items():
        if not isinstance(definition, dict):
            continue
        choices = definition.get("values") or definition.get("enum")
        if name not in values and "default" in definition:
            values[name] = definition["default"]
        elif (
            name not in values
            and name in {"voice", "response_format"}
            and isinstance(choices, list)
            and choices
        ):
            values[name] = choices[0]
        if definition.get("required") and values.get(name) is None:
            raise ValueError(f"{name} is required by the selected model")
        if name not in values:
            continue
        value = values[name]
        if isinstance(choices, list) and value not in choices:
            raise ValueError(f"{name} must be one of: {choices}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            minimum = definition.get("minimum")
            maximum = definition.get("maximum")
            if minimum is not None and value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
            if maximum is not None and value > maximum:
                raise ValueError(f"{name} must be at most {maximum}")
    return values


def _estimate_tts_catalog_cost(db_model, text_value: str) -> tuple[CostUsage | None, dict]:
    """Fallback for TTS generations whose OpenRouter usage record stays 404."""
    pricing = db_model.pricing if isinstance(db_model.pricing, dict) else {}
    text = str(text_value or "")
    candidates = (
        ("per_million_characters", len(text), 1_000_000),
        ("per_million_utf8_bytes", len(text.encode("utf-8")), 1_000_000),
        ("per_character", len(text), 1),
    )
    for key, units, divisor in candidates:
        if key not in pricing or pricing.get(key) is None:
            continue
        try:
            rate = float(pricing[key])
        except (TypeError, ValueError):
            continue
        total = (float(units) / float(divisor)) * rate
        return CostUsage(total_cost=round(total, 10), currency=str(pricing.get("currency") or "USD")), {
            "billing_source": "ai_models_catalog_estimate",
            "cost_is_estimate": True,
            "pricing_key": key,
            "pricing_rate": rate,
            "billable_units": units,
        }
    if db_model.tier == "free":
        return CostUsage(total_cost=0.0, currency=str(pricing.get("currency") or "USD")), {
            "billing_source": "ai_models_free_tier",
            "cost_is_estimate": False,
            "billable_units": len(text),
        }
    return None, {
        "billing_source": "unavailable",
        "cost_is_estimate": None,
        "billing_warning": "No provider generation cost and no supported TTS catalog price",
    }


@router.get("/general-tools/catalog")
async def general_tools_catalog(
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
):
    catalog = build_catalog()
    for tool in catalog["tools"]:
        rows, _ = list_ai_models(
            db, tool_key=tool["key"],
            provider="runware" if tool["key"] == "general_media" else None,
            page=1, per_page=100,
        )
        tool["models"] = [serialize_ai_model(row) for row in rows]
    return catalog


@router.get("/general-tools/{tool_key}/models")
async def general_tool_models(
    tool_key: str,
    operation: str | None = None,
    provider: str | None = None,
    tier: str | None = None,
    search: str | None = None,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
):
    rows, total = list_ai_models(
        db, tool_key=tool_key, operation=operation, provider=provider,
        tier=tier, search=search, page=page, per_page=per_page,
    )
    return {
        "tool": tool_key,
        "items": [serialize_ai_model(row) for row in rows],
        "pagination": {
            "page": page, "per_page": per_page, "total": total,
            "last_page": max(1, math.ceil(total / per_page)),
        },
    }


@router.post("/general-tools/models/sync/openrouter")
async def sync_openrouter_model_catalog(
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
):
    try:
        return {"success": True, **(await sync_openrouter_models(db))}
    except Exception as exc:
        db.rollback()
        raise _error(exc) from exc


def _text_endpoint(tool: str):
    async def endpoint(
        req: GeneralTextRequest,
        request: Request,
        db: Session = Depends(get_db),
        _: None = Depends(verify_internal_api_key),
    ) -> GeneralToolResponse:
        try:
            return await run_general_text(db, tool, req, request.state.request_id)
        except Exception as exc:
            raise _error(exc) from exc
    return endpoint


router.post("/general-chat", response_model=GeneralToolResponse)(_text_endpoint("general_chat"))
router.post("/general-code", response_model=GeneralToolResponse)(_text_endpoint("general_code"))
router.post("/general-translation", response_model=GeneralToolResponse)(_text_endpoint("general_translation"))


@router.post("/general-media", response_model=GeneralToolResponse)
async def general_media(
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> GeneralToolResponse:
    authorization = BillingAuthorization(enabled=False)
    provider_started = False
    provider_completed = False
    try:
        content_type = (request.headers.get("content-type") or "").lower()
        uploaded_file = None
        mask_file = None
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            payload = form.get("payload")
            if not payload:
                raise ValueError("payload is required for multipart general-media requests")
            req = GeneralMediaRequest.model_validate_json(str(payload))
            uploaded_file = form.get("file")
            mask_file = form.get("mask")
        elif content_type.startswith("application/json"):
            req = GeneralMediaRequest.model_validate(await request.json())
        else:
            raise ValueError(
                "general-media requires application/json for image_generation or "
                "multipart/form-data with payload and file for image processing"
            )

        get_existing_conversation_for_model(
            db, req.user_id, req.model_id, req.conversation_uuid
        )
        requested_operation = req.state.operation
        operation = (
            _detect_media_operation(
                req.user_message,
                has_file=_is_real_upload(uploaded_file),
            )
            if requested_operation == "auto"
            else requested_operation
        )
        db_model = get_ai_model(
            db, req.selected_model_id, tool_key="general_media"
        )
        if db_model.operation != operation:
            raise ValueError(
                f"Prompt was detected as {operation}, but selected_model_id "
                f"supports {db_model.operation}. Choose a model whose operation is {operation}."
            )
        if db_model.provider.lower() != "runware":
            raise ValueError(
                f"Selected provider is not supported for general-media {operation}; use runware"
            )
        if req.state.model and req.state.model != db_model.provider_model_id:
            raise ValueError("state.model does not match selected_model_id; omit state.model")
        params = _resolve_model_parameters(
            db_model, req.state.parameters, operation=operation
        )

        if get_settings().BILLING_ENFORCEMENT_ENABLED:
            estimated_cost = estimate_model_reservation(
                db_model,
                input_text=req.user_message,
                units=int(params.get("count") or 1),
            )
            authorization = authorize_charge(
                db, user_id=req.user_id, request_id=request.state.request_id,
                selected_model_id=req.selected_model_id,
                tool_key="general_media", operation=operation,
                estimated_provider_cost_usd=estimated_cost,
                pricing_snapshot={
                    "pricing": db_model.pricing or {},
                    "billing_config": db_model.billing_config or {},
                    "provider_model_id": db_model.provider_model_id,
                    "parameters": params,
                },
                reference_type="conversation",
                reference_uuid=req.conversation_uuid,
            )

        if operation not in {"image_generation", "video_generation"} and not _is_real_upload(uploaded_file):
            raise ValueError(f"file is required for {operation}")
        if operation == "remove_element" and not _is_real_upload(mask_file):
            raise ValueError("mask is required for remove_element")
        provider_started = mark_charge_started(db, authorization)
        if operation == "image_generation":
            legacy = ImageGeneratorRequest(
                user_id=req.user_id, sub_tool_id=req.model_id,
                conversation_uuid=req.conversation_uuid, user_message=req.user_message,
                provider="runware", model=db_model.provider_model_id,
                provider_metadata=db_model.provider_metadata or {},
                negative_prompt=params.get("negative_prompt"),
                enhance_prompt=bool(params.get("enhance_prompt", True)),
                size=str(params.get("size") or "1024x1024"),
                quality=str(params.get("quality") or "medium"),
                count=int(params.get("count") or 1),
                output_format=str(params.get("output_format") or "png"),
                seed=params.get("seed"), debug=req.debug,
            )
            result = await run_image_generator(
                db, legacy, request.state.request_id, validate_context=False
            )
        elif operation == "video_generation":
            result = await run_general_runware_media_operation(
                db, user_id=req.user_id, sub_tool_id=req.model_id,
                conversation_uuid=req.conversation_uuid, operation=operation,
                prompt=req.user_message, file=uploaded_file,
                mask_file=mask_file, model=db_model.provider_model_id,
                parameters=params,
                provider_metadata=db_model.provider_metadata or {},
                request_id=request.state.request_id,
            )
        else:
            if operation == "background_remove":
                result = await run_background_remover(
                    db, user_id=req.user_id, sub_tool_id=req.model_id,
                    conversation_uuid=req.conversation_uuid, file=uploaded_file,
                    provider_override="runware", request_id=request.state.request_id,
                    model_override=db_model.provider_model_id,
                    validate_context=False,
                )
            elif operation == "image_upscale":
                result = await run_image_upscaler(
                    db, user_id=req.user_id, sub_tool_id=req.model_id,
                    conversation_uuid=req.conversation_uuid, file=uploaded_file,
                    provider_override="runware", scale=int(params.get("scale") or 4),
                    face_enhance=bool(params.get("face_enhance", False)),
                    quality_mode=str(params.get("quality_mode") or "balanced"),
                    enhance_details=params.get("enhance_details"),
                    realism=params.get("realism"),
                    request_id=request.state.request_id,
                    model_override=db_model.provider_model_id,
                    validate_context=False,
                )
            elif operation in {
                "image_edit", "remove_element", "restore", "outpaint", "resize"
            }:
                result = await run_general_runware_media_operation(
                    db, user_id=req.user_id, sub_tool_id=req.model_id,
                    conversation_uuid=req.conversation_uuid, operation=operation,
                    prompt=req.user_message, file=uploaded_file,
                    mask_file=mask_file, model=db_model.provider_model_id,
                    parameters=params,
                    provider_metadata=db_model.provider_metadata or {},
                    request_id=request.state.request_id,
                )
            else:
                raise ValueError(f"Unsupported general-media operation: {operation}")

        provider_completed = True
        billing = capture_charge(
            db, authorization,
            actual_provider_cost_usd=(
                result.cost.total_cost if result.cost is not None else None
            ),
        )

        return GeneralToolResponse(
            tool="general_media", provider=result.provider, model=result.model,
            model_tier=db_model.tier, user_id=req.user_id, model_id=req.model_id,
            selected_model_id=req.selected_model_id,
            conversation_uuid=req.conversation_uuid, content=result.message,
            files=result.files, request_id=result.request_id, usage=result.usage,
            cost=result.cost, metadata={
                "operation": operation,
                "operation_source": "prompt" if requested_operation == "auto" else "explicit",
                "billing": billing,
                **result.metadata,
            },
            debug=result.debug,
        )
    except Exception as exc:
        if authorization.enabled and not provider_completed:
            if provider_started:
                mark_reconciliation_required(
                    db, authorization,
                    reason=f"provider_call_failed: {type(exc).__name__}",
                )
            else:
                release_charge(
                    db, authorization,
                    reason=f"failed_before_provider_call: {type(exc).__name__}",
                )
        raise _error(exc) from exc


@router.post("/general-audio", response_model=GeneralToolResponse)
async def general_audio(
    request: Request,
    payload: str = Form(...),
    file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> GeneralToolResponse:
    authorization = BillingAuthorization(enabled=False)
    provider_started = False
    provider_completed = False
    try:
        req = GeneralAudioRequest.model_validate_json(payload)
        settings = get_settings()
        get_existing_conversation_for_model(
            db, req.user_id, req.model_id, req.conversation_uuid
        )
        operation = req.state.operation
        db_model = get_ai_model(
            db, req.selected_model_id, tool_key="general_audio", operation=operation
        )
        params = _resolve_model_parameters(
            db_model, req.state.parameters, operation=operation
        )
        provider = (req.state.provider or db_model.provider).strip().lower()
        if provider != db_model.provider.lower():
            raise ValueError("Requested provider does not match selected_model_id")
        configured_model = db_model.provider_model_id
        if req.state.model and req.state.model != configured_model:
            raise ValueError("state.model does not match selected_model_id; omit state.model")

        if settings.BILLING_ENFORCEMENT_ENABLED:
            if operation == "text_to_speech":
                pricing = db_model.pricing if isinstance(db_model.pricing, dict) else {}
                units = (
                    len(req.user_message.encode("utf-8"))
                    if pricing.get("per_million_utf8_bytes") is not None
                    else len(req.user_message)
                )
            else:
                units = 0
            estimated_cost = estimate_model_reservation(
                db_model, input_text=req.user_message, units=units
            )
            authorization = authorize_charge(
                db, user_id=req.user_id, request_id=request.state.request_id,
                selected_model_id=req.selected_model_id,
                tool_key="general_audio", operation=operation,
                estimated_provider_cost_usd=estimated_cost,
                pricing_snapshot={
                    "pricing": db_model.pricing or {},
                    "billing_config": db_model.billing_config or {},
                    "provider_model_id": db_model.provider_model_id,
                    "parameters": params,
                },
                reference_type="conversation",
                reference_uuid=req.conversation_uuid,
            )

        if operation == "speech_to_text" and file is None:
            raise ValueError("file is required for speech_to_text")
        provider_started = mark_charge_started(db, authorization)
        if operation == "speech_to_text":
            result = await run_speech_to_text(
                db, user_id=req.user_id, sub_tool_id=req.model_id,
                conversation_uuid=req.conversation_uuid, file=file,
                provider_override=provider, language=params.get("language"),
                include_segments=bool(params.get("include_segments", False)),
                request_id=request.state.request_id,
                model_override=configured_model,
                validate_context=False,
            )
            provider_completed = True
            raw_cost = result.metadata.get("provider_cost_usd")
            billing = capture_charge(
                db, authorization, actual_provider_cost_usd=raw_cost
            )
            return GeneralToolResponse(
                tool="general_audio", provider=result.provider, model=result.model,
                model_tier=db_model.tier, user_id=req.user_id, model_id=req.model_id,
                selected_model_id=req.selected_model_id,
                conversation_uuid=req.conversation_uuid, content=result.transcript,
                request_id=result.request_id,
                cost=CostUsage(total_cost=float(result.metadata.get("provider_cost_usd")), currency="USD")
                if result.metadata.get("provider_cost_usd") is not None else None,
                metadata={"operation": operation, "billing": billing, "detected_language": result.detected_language, "duration_seconds": result.duration_seconds, "segments": result.segments, **result.metadata},
            )

        tts = TextToSpeechRequest(
            user_id=req.user_id, sub_tool_id=req.model_id,
            conversation_uuid=req.conversation_uuid, user_message=req.user_message,
            provider=provider, model=req.state.model or configured_model,
            voice=params.get("voice"), response_format=params.get("response_format"),
            speed=float(params.get("speed", 1.0)),
            provider_options=params.get("provider_options"), debug=req.debug,
        )
        result = await run_text_to_speech(
            db, tts, request.state.request_id, validate_context=False
        )
        provider_completed = True
        resolved_cost = result.cost
        if resolved_cost is not None:
            billing_metadata = {
                "billing_source": "openrouter_generation",
                "cost_is_estimate": False,
            }
        else:
            resolved_cost, billing_metadata = _estimate_tts_catalog_cost(
                db_model, req.user_message
            )
        billing = capture_charge(
            db, authorization,
            actual_provider_cost_usd=(
                resolved_cost.total_cost if resolved_cost is not None else None
            ),
        )
        return GeneralToolResponse(
            tool="general_audio", provider=result.provider, model=result.model,
            model_tier=db_model.tier, user_id=req.user_id, model_id=req.model_id,
            selected_model_id=req.selected_model_id,
            conversation_uuid=req.conversation_uuid, content=result.message,
            files=result.files, request_id=result.request_id,
            usage=result.usage, cost=resolved_cost,
            metadata={
                "operation": operation,
                "effective_parameters": params,
                "billing": billing,
                **billing_metadata,
                **result.metadata,
            }, debug=result.debug,
        )
    except Exception as exc:
        if authorization.enabled and not provider_completed:
            if provider_started:
                mark_reconciliation_required(
                    db, authorization,
                    reason=f"provider_call_failed: {type(exc).__name__}",
                )
            else:
                release_charge(
                    db, authorization,
                    reason=f"failed_before_provider_call: {type(exc).__name__}",
                )
        raise _error(exc) from exc
