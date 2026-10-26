from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "AI Tools"
    APP_ENV: str = "production"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 9000
    ALLOWED_ORIGINS: str = "*"

    # Text-model cost estimates.
    WRITER_INPUT_COST_PER_1M: float = 0.100
    WRITER_OUTPUT_COST_PER_1M: float = 0.500

    WEB_SEARCH_COST_PER_1000_RESULTS: float = 10.0

    # Headline chat defaults
    HEADLINE_DEFAULT_COUNT: int = 5
    HEADLINE_MAX_CONTENT_CHARS: int = 1000

    # Headline cost values per 1M tokens
    HEADLINE_INPUT_COST_PER_1M: float = 0.100
    HEADLINE_OUTPUT_COST_PER_1M: float = 0.500

    # Paraphraser chat defaults
    PARAPHRASER_MAX_CONTENT_CHARS: int = 6000
    PARAPHRASER_MIN_OUTPUT_TOKENS: int = 600
    PARAPHRASER_MAX_OUTPUT_TOKENS: int = 3000

    # Paraphraser cost values per 1M tokens
    PARAPHRASER_INPUT_COST_PER_1M: float = 0.100
    PARAPHRASER_OUTPUT_COST_PER_1M: float = 0.500

    # Use your real production DB here, for example:
    # mysql+pymysql://DB_USER:DB_PASSWORD@127.0.0.1:3306/DB_NAME?charset=utf8mb4
    DATABASE_URL: str = Field(default="")

    OPENROUTER_API_KEY: str = Field(default="")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # Used for OpenRouter headers only.
    SITE_URL: str = "https://pro.aiarabic.com"

    REQUEST_TIMEOUT_SECONDS: float = 180.0

    # OpenRouter reliability and diagnostics.
    # Structured-output requests should only use endpoints that support the
    # parameters being sent; response healing repairs common JSON syntax issues.
    OPENROUTER_REQUIRE_PARAMETERS: bool = True
    OPENROUTER_RESPONSE_HEALING_ENABLED: bool = True
    OPENROUTER_ROUTER_METADATA_ENABLED: bool = False
    # Keep disabled by default: session_id pins chat requests to one provider.
    OPENROUTER_SESSION_STICKINESS_ENABLED: bool = False
    OPENROUTER_TRACE_ENABLED: bool = False
    OPENROUTER_TRACE_CONTENT: bool = False
    OPENROUTER_TRACE_MAX_CHARS: int = 6000

    DEFAULT_TEMPERATURE: float = 0.45
    DEFAULT_MAX_TOKENS: int = 2500
    DEFAULT_HISTORY_LIMIT: int = 6

    ENABLE_DEBUG_RESPONSE: bool = False
    MAX_USER_MESSAGE_LENGTH: int = 20000
    MAX_HISTORY_MESSAGES: int = 6
    # Retained from the existing .env for compatibility with callers/config.
    CHAT_MAX_CONTEXT_TOKENS: int = 3000
    CHAT_RESERVED_RESPONSE_TOKENS: int = 600

    # Resume Builder generated files. Keep this existing variable for resumes only.
    GENERATED_FILES_DIR: str = "/home/oghasahy/api.aiarabic.com/ai_service/storage/generated"

    # Media-tool generated files (images/audio). Separate from resume files to
    # avoid a variable-name/path conflict with the existing resume builder.
    GENERATED_MEDIA_FILES_DIR: str = "/home/oghasahy/api.aiarabic.com/ai_service/storage/generated-media"

    MAX_RESUME_UPLOAD_MB: int = 8

    # ------------------------------------------------------------------
    # Media tools: provider selection and file limits
    # Existing text/resume settings above remain unchanged.
    # ------------------------------------------------------------------
    MEDIA_PROVIDER_TIMEOUT_SECONDS: float = 300.0
    MAX_IMAGE_UPLOAD_MB: int = 12
    MAX_AUDIO_UPLOAD_MB: int = 50
    MAX_IMAGE_PROMPT_LENGTH: int = 8000
    MAX_IMAGE_PIXELS: int = 40_000_000
    MAX_TTS_TEXT_LENGTH: int = 4096

    # OpenAI paid APIs: image generation, speech-to-text, and text-to-speech.
    OPENAI_API_KEY: str = Field(default="")
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_IMAGE_MODEL: str = "gpt-image-2"
    OPENAI_TRANSCRIPTION_MODEL: str = "gpt-4o-mini-transcribe"
    OPENAI_TTS_MODEL: str = "tts-1"
    OPENAI_TTS_VOICE: str = "alloy"
    OPENAI_TTS_FORMAT: str = "mp3"

    # AI Image Generator. Values: runware, openai, huggingface, comfyui.
    IMAGE_GENERATOR_PROVIDER: str = "runware"
    IMAGE_PROMPT_ENHANCEMENT_ENABLED: bool = True
    IMAGE_PROMPT_ENHANCEMENT_MAX_TOKENS: int = 1400
    IMAGE_PROMPT_ENHANCEMENT_TEMPERATURE: float = 0.25
    IMAGE_PROMPT_DEFAULT_NEGATIVE: str = (
        "low resolution, blur, pixelation, compression artifacts, duplicated subjects, "
        "cropped main subject, malformed anatomy, extra limbs, broken perspective, "
        "watermark, signature, interface elements"
    )

    # Runware image services use one REST endpoint and one API key.
    RUNWARE_API_KEY: str = Field(default="")
    RUNWARE_BASE_URL: str = "https://api.runware.ai/v1"
    RUNWARE_IMAGE_MODEL: str = "runware:400@4"
    RUNWARE_IMAGE_STEPS: int = 7
    RUNWARE_IMAGE_CFG_SCALE: float = 4
    RUNWARE_BACKGROUND_REMOVER_MODEL: str = "runware:112@9"
    RUNWARE_UPSCALER_MODEL: str = "prunaai:p-image@upscale"
    # Used automatically for 3x or detail/face-enhanced requests because the
    # low-cost Real-ESRGAN endpoint supports only 2x and 4x.
    RUNWARE_UPSCALER_ADVANCED_MODEL: str = "prunaai:p-image@upscale"
    RUNWARE_CHECK_CONTENT: bool = True
    


    HF_TOKEN: str = Field(default="")
    HF_IMAGE_PROVIDER: str = ""
    HF_IMAGE_MODEL: str = "black-forest-labs/FLUX.1-schnell"
    COMFYUI_BASE_URL: str = "http://127.0.0.1:8188"
    COMFYUI_WORKFLOW_PATH: str = "config/comfyui-text-to-image.json"
    COMFYUI_POLL_TIMEOUT_SECONDS: float = 300.0
    COMFYUI_POLL_INTERVAL_SECONDS: float = 1.0

    # AI Background Remover. Values: runware, rembg, removebg.
    BACKGROUND_REMOVER_PROVIDER: str = "runware"
    REMBG_MODEL: str = "birefnet-general"
    REMBG_ALPHA_MATTING: bool = True
    REMBG_ALPHA_MATTING_FOREGROUND_THRESHOLD: int = 240
    REMBG_ALPHA_MATTING_BACKGROUND_THRESHOLD: int = 10
    REMBG_ALPHA_MATTING_ERODE_SIZE: int = 10
    REMBG_POST_PROCESS_MASK: bool = True
    REMOVEBG_API_KEY: str = Field(default="")
    REMOVEBG_API_URL: str = "https://api.remove.bg/v1.0/removebg"
    REMOVEBG_SIZE: str = "auto"

    # AI Image Upscaler. Values: runware, realesrgan, replicate.
    IMAGE_UPSCALER_PROVIDER: str = "runware"
    IMAGE_UPSCALER_QUALITY_MODE: str = "quality"
    RUNWARE_UPSCALER_ENHANCE_DETAILS: bool = True
    RUNWARE_UPSCALER_REALISM: bool = False
    REALESRGAN_BINARY: str = ""
    REALESRGAN_MODEL: str = "realesrgan-x4plus"
    REPLICATE_API_TOKEN: str = Field(default="")
    REPLICATE_BASE_URL: str = "https://api.replicate.com/v1"
    REPLICATE_UPSCALER_MODEL: str = "nightmareai/real-esrgan"
    REPLICATE_POLL_TIMEOUT_SECONDS: float = 300.0
    REPLICATE_POLL_INTERVAL_SECONDS: float = 1.0

    # Dedicated OpenRouter audio endpoints. These use the same
    # OPENROUTER_API_KEY and OPENROUTER_BASE_URL as the text tools.
    OPENROUTER_STT_MODEL: str = "openai/whisper-large-v3"
    OPENROUTER_STT_TEMPERATURE: float = 0.0
    OPENROUTER_TTS_MODEL: str = "openai/gpt-4o-mini-tts-2025-12-15"
    OPENROUTER_TTS_VOICE: str = "nova"
    OPENROUTER_TTS_FORMAT: str = "mp3"
    YOUTUBE_DEFAULT_LANGUAGES: str = "ar,en"
    YOUTUBE_MAX_TRANSCRIPT_CHARS: int = 120000
    YOUTUBE_SUMMARY_CHUNK_CHARS: int = 16000
    YOUTUBE_CHUNK_SUMMARY_MAX_TOKENS: int = 900
    YOUTUBE_FINAL_SUMMARY_MAX_TOKENS: int = 1800

    # YouTube transcript retrieval fallback chain:
    # 1) direct youtube-transcript-api (free)
    # 2) Supadata API when configured
    # 3) generic/Webshare proxy when configured
    SUPADATA_API_KEY: str = Field(default="")
    SUPADATA_BASE_URL: str = "https://api.supadata.ai/v1"
    SUPADATA_TRANSCRIPT_MODE: str = "auto"
    SUPADATA_TIMEOUT_SECONDS: float = 90.0
    SUPADATA_POLL_TIMEOUT_SECONDS: float = 120.0
    SUPADATA_POLL_INTERVAL_SECONDS: float = 1.0

    # Generic HTTP/HTTPS proxy URL for Bright Data, Oxylabs, Webshare, etc.
    # Example: http://user:password@proxy-host:port
    YOUTUBE_PROXY_URL: str = Field(default="")

    # Existing Webshare-specific credentials remain supported for backward compatibility.
    YOUTUBE_WEBSHARE_PROXY_USERNAME: str = Field(default="")
    YOUTUBE_WEBSHARE_PROXY_PASSWORD: str = Field(default="")

    # Speech-to-text. Values: openrouter, openai, faster-whisper.
    SPEECH_TO_TEXT_PROVIDER: str = "openrouter"
    FASTER_WHISPER_MODEL: str = "small"
    FASTER_WHISPER_DEVICE: str = "cpu"
    FASTER_WHISPER_COMPUTE_TYPE: str = "int8"
    FASTER_WHISPER_BEAM_SIZE: int = 5
    FASTER_WHISPER_VAD_FILTER: bool = True

    # Text-to-speech. Values: openrouter, openai, piper.
    TEXT_TO_SPEECH_PROVIDER: str = "openrouter"
    PIPER_MODEL_PATH: str = ""
    PIPER_USE_CUDA: bool = False

    # Web search defaults. Keep small for cost control.
    WEB_SEARCH_DEFAULT_MAX_RESULTS: int = 3
    WEB_SEARCH_DEFAULT_MAX_TOTAL_RESULTS: int = 5
    WEB_SEARCH_HARD_MAX_RESULTS: int = 5
    WEB_SEARCH_HARD_MAX_TOTAL_RESULTS: int = 10

    INTERNAL_API_KEY: str = Field(default="")
    INTERNAL_API_HEADER_NAME: str = "X-Internal-Api-Key"

    # General tools are isolated from the existing task routes. The model IDs
    # exposed to users are defined in app/general_tools/catalog.py.
    GENERAL_TOOLS_ENABLED: bool = True
    GENERAL_CHAT_DEFAULT_MODEL: str = "qwen/qwen3.5-35b-a3b"
    GENERAL_CODE_DEFAULT_MODEL: str = "qwen/qwen3-coder-next"
    GENERAL_TRANSLATION_DEFAULT_MODEL: str = "qwen/qwen3.5-35b-a3b"
    GENERAL_TOOLS_MAX_OUTPUT_TOKENS: int = 12000
    GENERAL_TOOLS_HISTORY_LIMIT: int = 12

    # Check-only credits guard. The Laravel backend owns the distributed
    # per-user lock and performs the final debit after the response.
    BILLING_ENFORCEMENT_ENABLED: bool = True
    BILLING_CREDITS_PER_USD: int = 1_000_000
    BILLING_SAFETY_MARGIN_PERCENT: float = 10.0
    BILLING_MARKUP_MULTIPLIER: float = 1.0
    BILLING_MIN_REQUIRED_CREDITS: int = 1
    # Code execution remains disabled until an isolated worker/container is
    # configured. Never point this at the API application's own checkout.
    AGENT_SANDBOX_ENABLED: bool = False
    AGENT_SANDBOX_PROVIDER: str = "docker"
    AGENT_SANDBOX_WORKSPACE_ROOT: str = "/tmp/ai-agent-sandboxes"
    AGENT_SANDBOX_TIMEOUT_SECONDS: int = 60
    AGENT_SANDBOX_MEMORY_MB: int = 512
    AGENT_SANDBOX_CPU_LIMIT: float = 1.0
    AGENT_SANDBOX_NETWORK_ENABLED: bool = False
    # Comma-separated trusted Python modules. Each module must expose
    # register_agent_tools(registry) and can add tools independent of site routes.
    AGENT_TOOL_MODULES: str = ""
    # Application conversations for the new model-based tools.
    MODELS_CONVERSATIONS_TABLE: str = "models_conversations"
    # JSON list of Runware models exposed by general-media. Configure exact AIR
    # identifiers from your Runware account; the first enabled item is default.
    GENERAL_RUNWARE_MODELS_JSON: str = (
        '[{"id":"runware:400@4","label":"Runware Fast Image",'
        '"tier":"standard","recommended":true,"enabled":true}]'
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
