import uuid
import httpx
from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.settings import get_settings
from app.crud import get_conversation_by_uuid_for_user, list_messages
from app.database import get_db
from app.schemas import (
    TaskRequest,
    TaskResponse,
    MessageResponse,
    HeadlineChatRequest,
    HeadlineChatResponse,
    ParaphraserChatRequest,
    ParaphraserChatResponse,
    SocialPostChatRequest,
    SocialPostChatResponse,
    EmailWriterChatRequest,
    EmailWriterChatResponse,
    ScriptGeneratorChatRequest,
    ScriptGeneratorChatResponse,
    ProductDescriptionChatRequest,
    ProductDescriptionChatResponse,
    PromptGeneratorChatRequest,
    PromptGeneratorChatResponse,
    PromptEnhancerChatRequest,
    PromptEnhancerChatResponse,
    IdeaGeneratorChatRequest,
    IdeaGeneratorChatResponse,
    HookGeneratorChatRequest,
    HookGeneratorChatResponse,
    KeywordGeneratorChatRequest,
    KeywordGeneratorChatResponse,
    MetaDescriptionChatRequest,
    MetaDescriptionChatResponse,
    ContentAnalyzerChatRequest,
    ContentAnalyzerChatResponse,
    ContentOptimizerChatRequest,
    ContentOptimizerChatResponse,
    AIDetectorChatRequest,
    AIDetectorChatResponse,
    AIHumanizerChatRequest,
    AIHumanizerChatResponse,
    BusinessNameChatRequest,
    BusinessNameChatResponse,
    ResumeBuilderChatResponse,
    ImageGeneratorRequest,
    MediaToolResponse,
    YouTubeSummarizerRequest,
    YouTubeSummarizerResponse,
    SpeechToTextResponse,
    TextToSpeechRequest,
    MediaUploadRequest,
)
from app.headline_chat import run_headline_chat
from app.paraphraser_chat import run_paraphraser_chat
from app.generic_content_chat import (
    run_social_post_chat,
    run_email_writer_chat,
    run_script_generator_chat,
    run_product_description_chat,
    run_prompt_generator_chat,
    run_prompt_enhancer_chat,
    run_idea_generator_chat,
    run_hook_generator_chat,
    run_keyword_generator_chat,
    run_meta_description_chat,
    run_content_analyzer_chat,
    run_content_optimizer_chat,
    run_ai_detector_chat,
    run_ai_humanizer_chat,
    run_business_name_chat,
)
from app.services import run_task
from app.resume_builder import run_resume_builder_chat_upload, resume_file_path
from app.media_storage import get_media_file
from app.media_tools import (
    run_image_generator,
    run_background_remover,
    run_image_upscaler,
    run_youtube_summarizer,
    run_speech_to_text,
    run_text_to_speech,
)
from app.tasks import TASKS
from app.request_context import reset_request_id, set_request_id
from app.security import verify_internal_api_key
from app.general_tools.router import router as general_tools_router
from app.trends.router import router as trends_router
from app.agents.router import router as agents_router
from app.billing.errors import BillingError
from app.billing.legacy import run_legacy_billable

settings = get_settings()


def _endpoint_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, BillingError):
        return HTTPException(status_code=exc.status_code, detail=exc.as_detail())
    return HTTPException(status_code=502, detail=str(exc))


def _wrap_legacy_request_runner(runner, tool_key: str, operation: str):
    """Wrap legacy `(db, request_model, request_id)` runners once, centrally."""
    async def wrapped(db, req, request_id, *args, **kwargs):
        return await run_legacy_billable(
            db, user_id=req.user_id, sub_tool_id=req.sub_tool_id,
            request_id=request_id, tool_key=tool_key, operation=operation,
            request_payload=req,
            call=lambda: runner(db, req, request_id, *args, **kwargs),
        )
    return wrapped


def _wrap_legacy_keyword_runner(runner, tool_key: str, operation: str):
    """Wrap legacy runners whose identity fields are keyword arguments."""
    async def wrapped(db, *args, **kwargs):
        return await run_legacy_billable(
            db, user_id=int(kwargs["user_id"]),
            sub_tool_id=int(kwargs["sub_tool_id"]),
            request_id=str(kwargs["request_id"]),
            tool_key=tool_key, operation=operation,
            request_payload=kwargs,
            call=lambda: runner(db, *args, **kwargs),
        )
    return wrapped


# These names are local to main.py. General-model routes import the original
# media runners directly, so wrapping here does not double-charge them.
run_headline_chat = _wrap_legacy_request_runner(run_headline_chat, "headline_generator", "text_generation")
run_paraphraser_chat = _wrap_legacy_request_runner(run_paraphraser_chat, "paraphraser", "text_generation")
run_social_post_chat = _wrap_legacy_request_runner(run_social_post_chat, "social_post_generator", "text_generation")
run_email_writer_chat = _wrap_legacy_request_runner(run_email_writer_chat, "email_writer", "text_generation")
run_script_generator_chat = _wrap_legacy_request_runner(run_script_generator_chat, "script_generator", "text_generation")
run_product_description_chat = _wrap_legacy_request_runner(run_product_description_chat, "product_description_generator", "text_generation")
run_prompt_generator_chat = _wrap_legacy_request_runner(run_prompt_generator_chat, "prompt_generator", "text_generation")
run_prompt_enhancer_chat = _wrap_legacy_request_runner(run_prompt_enhancer_chat, "prompt_enhancer", "text_generation")
run_idea_generator_chat = _wrap_legacy_request_runner(run_idea_generator_chat, "idea_generator", "text_generation")
run_hook_generator_chat = _wrap_legacy_request_runner(run_hook_generator_chat, "hook_generator", "text_generation")
run_keyword_generator_chat = _wrap_legacy_request_runner(run_keyword_generator_chat, "keyword_generator", "text_generation")
run_meta_description_chat = _wrap_legacy_request_runner(run_meta_description_chat, "meta_description_generator", "text_generation")
run_content_analyzer_chat = _wrap_legacy_request_runner(run_content_analyzer_chat, "content_analyzer", "text_generation")
run_content_optimizer_chat = _wrap_legacy_request_runner(run_content_optimizer_chat, "content_optimizer", "text_generation")
run_ai_detector_chat = _wrap_legacy_request_runner(run_ai_detector_chat, "ai_detector", "text_generation")
run_ai_humanizer_chat = _wrap_legacy_request_runner(run_ai_humanizer_chat, "ai_humanizer", "text_generation")
run_business_name_chat = _wrap_legacy_request_runner(run_business_name_chat, "business_name_generator", "text_generation")
run_image_generator = _wrap_legacy_request_runner(run_image_generator, "image_generator", "image_generation")
run_youtube_summarizer = _wrap_legacy_request_runner(run_youtube_summarizer, "youtube_summarizer", "summarization")
run_text_to_speech = _wrap_legacy_request_runner(run_text_to_speech, "text_to_speech", "text_to_speech")
run_resume_builder_chat_upload = _wrap_legacy_keyword_runner(run_resume_builder_chat_upload, "resume_builder", "document_generation")
run_background_remover = _wrap_legacy_keyword_runner(run_background_remover, "background_remover", "background_remove")
run_image_upscaler = _wrap_legacy_keyword_runner(run_image_upscaler, "image_upscaler", "image_upscale")
run_speech_to_text = _wrap_legacy_keyword_runner(run_speech_to_text, "speech_to_text", "speech_to_text")

def _resolve_media_upload_request(
    *,
    payload: str | None,
    user_id: int | None,
    sub_tool_id: int | None,
    conversation_uuid: str | None,
    user_message: str | None,
    debug: bool,
) -> MediaUploadRequest:
    """
    Accept the preferred unified JSON envelope in multipart field `payload`,
    while retaining the old individual form fields for backward compatibility.
    """
    if payload:
        try:
            return MediaUploadRequest.model_validate_json(payload)
        except Exception as exc:
            raise ValueError(f"Invalid multipart payload JSON: {exc}") from exc

    if user_id is None or sub_tool_id is None or not str(conversation_uuid or "").strip():
        raise ValueError(
            "Send either the multipart `payload` JSON field, or the legacy "
            "user_id, sub_tool_id, and conversation_uuid form fields"
        )

    return MediaUploadRequest(
        user_id=user_id,
        sub_tool_id=sub_tool_id,
        conversation_uuid=str(conversation_uuid).strip(),
        user_message=(user_message or "Process the uploaded file").strip(),
        state={},
        debug=debug,
    )


def _state_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


app = FastAPI(title=settings.APP_NAME)
app.include_router(general_tools_router)
app.include_router(trends_router)
app.include_router(agents_router)

origins = (
    ["*"]
    if settings.ALLOWED_ORIGINS == "*"
    else [origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",") if origin.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    # Credentialed CORS requires explicit origins; disable credentials for wildcard mode.
    allow_credentials=origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    token = set_request_id(request_id)
    try:
        response = await call_next(request)
    finally:
        reset_request_id(token)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "env": settings.APP_ENV,
        "db_write_mode": (
            "mysql_credits_check_only"
            if settings.BILLING_ENFORCEMENT_ENABLED
            else "disabled_fastapi_read_only"
        ),
    }


@app.get(
    "/users/{user_id}/conversations/{conversation_uuid}/messages",
    response_model=list[MessageResponse],
    tags=["messages"],
)
async def list_conversation_messages_endpoint(
    user_id: int,
    conversation_uuid: str,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
):
    conv = get_conversation_by_uuid_for_user(db, conversation_uuid, user_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    rows = list_messages(db, conv.id, limit=500)
    return [
        MessageResponse(
            id=row.id,
            conversation_id=row.conversation_id,
            role=row.role,
            content=row.content,
            is_error=bool(row.is_error),
            created_at=row.created_at.isoformat() if row.created_at else None,
        )
        for row in rows
    ]


def create_task_endpoint(task_key: str):
    async def endpoint(
        req: TaskRequest,
        request: Request,
        db: Session = Depends(get_db),
        _: None = Depends(verify_internal_api_key),
    ) -> TaskResponse:
        try:
            return await run_legacy_billable(
                db, user_id=req.user_id, sub_tool_id=req.sub_tool_id,
                request_id=request.state.request_id,
                tool_key=task_key, operation="text_generation",
                call=lambda: run_task(
                    db, task_key, req, request.state.request_id
                ),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except httpx.HTTPStatusError as e:
            provider_text = e.response.text if e.response is not None else str(e)
            raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
        except Exception as e:
            raise _endpoint_exception(e)

    return endpoint


@app.post(
    "/tasks/headline-generator/chat",
    response_model=HeadlineChatResponse,
    tags=["tasks"],
)
async def headline_generator_chat_endpoint(
    req: HeadlineChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> HeadlineChatResponse:
    try:
        return await run_headline_chat(db, req, request.state.request_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise _endpoint_exception(e)


@app.post(
    "/tasks/paraphraser/chat",
    response_model=ParaphraserChatResponse,
    tags=["tasks"],
)
async def paraphraser_chat_endpoint(
    req: ParaphraserChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> ParaphraserChatResponse:
    try:
        return await run_paraphraser_chat(db, req, request.state.request_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise _endpoint_exception(e)


@app.post(
    "/tasks/social-post-generator/chat",
    response_model=SocialPostChatResponse,
    tags=["tasks"],
)
async def social_post_generator_chat_endpoint(
    req: SocialPostChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> SocialPostChatResponse:
    try:
        return await run_social_post_chat(db, req, request.state.request_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise _endpoint_exception(e)


@app.post(
    "/tasks/email-writer/chat",
    response_model=EmailWriterChatResponse,
    tags=["tasks"],
)
async def email_writer_chat_endpoint(
    req: EmailWriterChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> EmailWriterChatResponse:
    try:
        return await run_email_writer_chat(db, req, request.state.request_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise _endpoint_exception(e)


@app.post(
    "/tasks/script-generator/chat",
    response_model=ScriptGeneratorChatResponse,
    tags=["tasks"],
)
async def script_generator_chat_endpoint(
    req: ScriptGeneratorChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> ScriptGeneratorChatResponse:
    try:
        return await run_script_generator_chat(db, req, request.state.request_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise _endpoint_exception(e)


@app.post(
    "/tasks/product-description-generator/chat",
    response_model=ProductDescriptionChatResponse,
    tags=["tasks"],
)
async def product_description_generator_chat_endpoint(
    req: ProductDescriptionChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> ProductDescriptionChatResponse:
    try:
        return await run_product_description_chat(db, req, request.state.request_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise _endpoint_exception(e)


@app.post(
    "/tasks/prompt-generator/chat",
    response_model=PromptGeneratorChatResponse,
    tags=["tasks"],
)
async def prompt_generator_chat_endpoint(
    req: PromptGeneratorChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> PromptGeneratorChatResponse:
    try:
        return await run_prompt_generator_chat(db, req, request.state.request_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise _endpoint_exception(e)


@app.post(
    "/tasks/prompt-enhancer/chat",
    response_model=PromptEnhancerChatResponse,
    tags=["tasks"],
)
async def prompt_enhancer_chat_endpoint(
    req: PromptEnhancerChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> PromptEnhancerChatResponse:
    try:
        return await run_prompt_enhancer_chat(db, req, request.state.request_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise _endpoint_exception(e)


@app.post(
    "/tasks/idea-generator/chat",
    response_model=IdeaGeneratorChatResponse,
    tags=["tasks"],
)
async def idea_generator_chat_endpoint(
    req: IdeaGeneratorChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> IdeaGeneratorChatResponse:
    try:
        return await run_idea_generator_chat(db, req, request.state.request_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise _endpoint_exception(e)


@app.post(
    "/tasks/hook-generator/chat",
    response_model=HookGeneratorChatResponse,
    tags=["tasks"],
)
async def hook_generator_chat_endpoint(
    req: HookGeneratorChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> HookGeneratorChatResponse:
    try:
        return await run_hook_generator_chat(db, req, request.state.request_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise _endpoint_exception(e)


@app.post(
    "/tasks/keyword-generator/chat",
    response_model=KeywordGeneratorChatResponse,
    tags=["tasks"],
)
async def keyword_generator_chat_endpoint(
    req: KeywordGeneratorChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> KeywordGeneratorChatResponse:
    try:
        return await run_keyword_generator_chat(db, req, request.state.request_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise _endpoint_exception(e)


@app.post(
    "/tasks/meta-description-generator/chat",
    response_model=MetaDescriptionChatResponse,
    tags=["tasks"],
)
async def meta_description_generator_chat_endpoint(
    req: MetaDescriptionChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> MetaDescriptionChatResponse:
    try:
        return await run_meta_description_chat(db, req, request.state.request_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise _endpoint_exception(e)


@app.post(
    "/tasks/content-analyzer/chat",
    response_model=ContentAnalyzerChatResponse,
    tags=["tasks"],
)
async def content_analyzer_chat_endpoint(
    req: ContentAnalyzerChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> ContentAnalyzerChatResponse:
    try:
        return await run_content_analyzer_chat(db, req, request.state.request_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise _endpoint_exception(e)


@app.post(
    "/tasks/content-optimizer/chat",
    response_model=ContentOptimizerChatResponse,
    tags=["tasks"],
)
async def content_optimizer_chat_endpoint(
    req: ContentOptimizerChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> ContentOptimizerChatResponse:
    try:
        return await run_content_optimizer_chat(db, req, request.state.request_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise _endpoint_exception(e)


@app.post(
    "/tasks/ai-detector/chat",
    response_model=AIDetectorChatResponse,
    tags=["tasks"],
)
async def ai_detector_chat_endpoint(
    req: AIDetectorChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> AIDetectorChatResponse:
    try:
        return await run_ai_detector_chat(db, req, request.state.request_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise _endpoint_exception(e)


@app.post(
    "/tasks/ai-humanizer/chat",
    response_model=AIHumanizerChatResponse,
    tags=["tasks"],
)
async def ai_humanizer_chat_endpoint(
    req: AIHumanizerChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> AIHumanizerChatResponse:
    try:
        return await run_ai_humanizer_chat(db, req, request.state.request_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise _endpoint_exception(e)


@app.post(
    "/tasks/business-name-generator/chat",
    response_model=BusinessNameChatResponse,
    tags=["tasks"],
)
async def business_name_chat_endpoint(
    req: BusinessNameChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> BusinessNameChatResponse:
    try:
        return await run_business_name_chat(db, req, request.state.request_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise _endpoint_exception(e)


@app.post(
    "/tasks/resume-builder/chat",
    response_model=ResumeBuilderChatResponse,
    tags=["tasks"],
)
async def resume_builder_chat_endpoint(
    request: Request,
    user_id: int = Form(...),
    sub_tool_id: int = Form(...),
    conversation_uuid: str = Form(...),
    user_message: str = Form(...),
    state: str | None = Form(default=None),
    debug: bool = Form(default=False),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> ResumeBuilderChatResponse:
    try:
        return await run_resume_builder_chat_upload(
            db,
            user_id=user_id,
            sub_tool_id=sub_tool_id,
            conversation_uuid=conversation_uuid,
            user_message=user_message,
            state_json=state,
            file=file,
            debug=debug,
            request_id=request.state.request_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise _endpoint_exception(e)


@app.get(
    "/tasks/resume-builder/download/{file_id}",
    tags=["tasks"],
)
async def resume_builder_download_endpoint(
    file_id: str,
    _: None = Depends(verify_internal_api_key),
):
    try:
        path = resume_file_path(file_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not path.exists():
        raise HTTPException(status_code=404, detail="Generated resume file not found")
    return FileResponse(
        path=str(path),
        filename=path.name,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@app.post(
    "/tasks/image-generator",
    response_model=MediaToolResponse,
    tags=["media tools"],
)
async def image_generator_endpoint(
    req: ImageGeneratorRequest,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> MediaToolResponse:
    try:
        return await run_image_generator(db, req, request.state.request_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise _endpoint_exception(e)


@app.post(
    "/tasks/background-remover",
    response_model=MediaToolResponse,
    tags=["media tools"],
)
async def background_remover_endpoint(
    request: Request,
    file: UploadFile = File(...),
    payload: str | None = Form(default=None),
    user_id: int | None = Form(default=None),
    sub_tool_id: int | None = Form(default=None),
    conversation_uuid: str | None = Form(default=None),
    user_message: str | None = Form(default=None),
    provider: str | None = Form(default=None),
    debug: bool = Form(default=False),
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> MediaToolResponse:
    try:
        envelope = _resolve_media_upload_request(
            payload=payload,
            user_id=user_id,
            sub_tool_id=sub_tool_id,
            conversation_uuid=conversation_uuid,
            user_message=user_message,
            debug=debug,
        )
        provider_override = provider or envelope.state.get("provider")
        return await run_background_remover(
            db,
            user_id=envelope.user_id,
            sub_tool_id=envelope.sub_tool_id,
            conversation_uuid=envelope.conversation_uuid,
            file=file,
            provider_override=provider_override,
            request_id=request.state.request_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise _endpoint_exception(e)


@app.post(
    "/tasks/image-upscaler",
    response_model=MediaToolResponse,
    tags=["media tools"],
)
async def image_upscaler_endpoint(
    request: Request,
    file: UploadFile = File(...),
    payload: str | None = Form(default=None),
    user_id: int | None = Form(default=None),
    sub_tool_id: int | None = Form(default=None),
    conversation_uuid: str | None = Form(default=None),
    user_message: str | None = Form(default=None),
    provider: str | None = Form(default=None),
    scale: int | None = Form(default=None),
    face_enhance: bool | None = Form(default=None),
    quality_mode: str | None = Form(default=None),
    enhance_details: bool | None = Form(default=None),
    realism: bool | None = Form(default=None),
    debug: bool = Form(default=False),
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> MediaToolResponse:
    try:
        envelope = _resolve_media_upload_request(
            payload=payload,
            user_id=user_id,
            sub_tool_id=sub_tool_id,
            conversation_uuid=conversation_uuid,
            user_message=user_message,
            debug=debug,
        )
        provider_override = provider or envelope.state.get("provider")
        resolved_scale = int(scale if scale is not None else envelope.state.get("scale", 4))
        resolved_face_enhance = (
            face_enhance
            if face_enhance is not None
            else _state_bool(envelope.state.get("face_enhance"), False)
        )
        resolved_quality_mode = quality_mode or envelope.state.get("quality_mode")
        resolved_enhance_details = (
            enhance_details
            if enhance_details is not None
            else (
                _state_bool(envelope.state.get("enhance_details"), False)
                if "enhance_details" in envelope.state
                else None
            )
        )
        resolved_realism = (
            realism
            if realism is not None
            else (
                _state_bool(envelope.state.get("realism"), False)
                if "realism" in envelope.state
                else None
            )
        )
        return await run_image_upscaler(
            db,
            user_id=envelope.user_id,
            sub_tool_id=envelope.sub_tool_id,
            conversation_uuid=envelope.conversation_uuid,
            file=file,
            provider_override=provider_override,
            scale=resolved_scale,
            face_enhance=resolved_face_enhance,
            quality_mode=resolved_quality_mode,
            enhance_details=resolved_enhance_details,
            realism=resolved_realism,
            request_id=request.state.request_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise _endpoint_exception(e)


@app.post(
    "/tasks/youtube-summarizer",
    response_model=YouTubeSummarizerResponse,
    tags=["media tools"],
)
async def youtube_summarizer_endpoint(
    req: YouTubeSummarizerRequest,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> YouTubeSummarizerResponse:
    try:
        return await run_youtube_summarizer(db, req, request.state.request_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise _endpoint_exception(e)


@app.post(
    "/tasks/speech-to-text",
    response_model=SpeechToTextResponse,
    tags=["media tools"],
)
async def speech_to_text_endpoint(
    request: Request,
    file: UploadFile = File(...),
    payload: str | None = Form(default=None),
    user_id: int | None = Form(default=None),
    sub_tool_id: int | None = Form(default=None),
    conversation_uuid: str | None = Form(default=None),
    user_message: str | None = Form(default=None),
    provider: str | None = Form(default=None),
    language: str | None = Form(default=None),
    include_segments: bool | None = Form(default=None),
    debug: bool = Form(default=False),
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> SpeechToTextResponse:
    try:
        envelope = _resolve_media_upload_request(
            payload=payload,
            user_id=user_id,
            sub_tool_id=sub_tool_id,
            conversation_uuid=conversation_uuid,
            user_message=user_message,
            debug=debug,
        )
        provider_override = provider or envelope.state.get("provider")
        resolved_language = language or envelope.state.get("language")
        resolved_include_segments = (
            include_segments
            if include_segments is not None
            else _state_bool(envelope.state.get("include_segments"), False)
        )
        return await run_speech_to_text(
            db,
            user_id=envelope.user_id,
            sub_tool_id=envelope.sub_tool_id,
            conversation_uuid=envelope.conversation_uuid,
            file=file,
            provider_override=provider_override,
            language=resolved_language,
            include_segments=resolved_include_segments,
            request_id=request.state.request_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise _endpoint_exception(e)


@app.post(
    "/tasks/text-to-speech",
    response_model=MediaToolResponse,
    tags=["media tools"],
)
async def text_to_speech_endpoint(
    req: TextToSpeechRequest,
    request: Request,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> MediaToolResponse:
    try:
        return await run_text_to_speech(db, req, request.state.request_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        provider_text = e.response.text if e.response is not None else str(e)
        raise HTTPException(status_code=502, detail=f"Provider HTTP error: {provider_text}")
    except Exception as e:
        raise _endpoint_exception(e)


@app.get(
    "/tasks/generated-files/download/{file_id}",
    tags=["media tools"],
)
async def generated_media_download_endpoint(
    file_id: str,
    _: None = Depends(verify_internal_api_key),
):
    try:
        stored = get_media_file(file_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return FileResponse(
        path=str(stored.path),
        filename=stored.filename,
        media_type=stored.content_type,
    )


for task_key, task in TASKS.items():
    app.post(task["path"], response_model=TaskResponse, tags=["tasks"])(
        create_task_endpoint(task_key)
    )
