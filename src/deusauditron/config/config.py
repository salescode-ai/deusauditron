"""
Configuration module for Deusauditron.
This module centralizes all environment variable access and provides type-safe configuration.
"""

import os
from typing import List, Optional
from phoenix.otel import register
from openinference.instrumentation import OITracer
from loguru import logger
from pydantic import BaseModel
from pydantic import Field
from dotenv import load_dotenv

try:
    load_dotenv()
except ImportError:
    logger.warning("Error loading .env file")
    pass

class TracingConfig(BaseModel):
    enabled: bool = Field(default=False)
    tracer_url: str = Field(default="http://localhost:6006")
    tracer_api_key: str = Field(default="")


class TracingManager:
    _instance: Optional["TracingManager"] = None
    _tracer: Optional[OITracer] = None
    _initialized = False

    def __new__(cls) -> "TracingManager":
        if cls._instance is None:
            cls._instance = super(TracingManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._tracer = None
            self._initialized = True
            self._auto_initialize()

    def _auto_initialize(self) -> None:
        try:
            config = get_config()
            if not config.tracing.enabled:
                return
            
            # Set up Phoenix Cloud authentication
            os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = config.tracing.tracer_url
            if config.tracing.tracer_api_key:
                os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={config.tracing.tracer_api_key}"
            
            # Register Phoenix tracer with proper configuration
            tracer_provider = register(
                project_name="Deusauditron", 
                auto_instrument=True, 
                batch=True
            )
            self._tracer = tracer_provider.get_tracer("Deusauditron")
            
            from openinference.instrumentation.litellm import LiteLLMInstrumentor
            LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
            
            logger.info("Phoenix tracing initialized successfully")
        except Exception as e:
            logger.error(f"Failed to auto-initialize Phoenix tracer: {e}")
            self._tracer = None

    @property
    def tracer(self):
        return self._tracer

    @property
    def is_enabled(self) -> bool:
        return self._tracer is not None

    def get_tracer(self) -> Optional[OITracer]:
        return self._tracer


class RedisConfig(BaseModel):
    use_redis: bool = Field(default=False)
    redis_url: str = Field(default="redis://127.0.0.1:6379/0")


class CORSConfig(BaseModel):
    allowed_origins: List[str] = Field(
        default=[
            "https://app.salescode.ai",
            "https://admin.salescode.ai",
            "http://localhost:5173",
        ]
    )
    allow_credentials: bool = Field(default=True)
    allow_methods: List[str] = Field(default=["*"])
    allow_headers: List[str] = Field(default=["*"])


class LLMConfig(BaseModel):
    max_retries: int = Field(default=3)
    fallback_model: str = Field(default="groq/llama-3.3-70b-versatile")
    providers_without_api_key: List[str] = Field(default=["bedrock"])


class StateConfig(BaseModel):
    wait_timeout: float = Field(default=5.0)
    check_interval: float = Field(default=0.1)


class TranscriptionConfig(BaseModel):
    strategy: str = Field(default="llm")  # "llm", "aws", "elevenlabs" or "deepgram"
    llm_model: str = Field(default="whisper-1") 
    timeout: int = Field(default=60)
    aws_region: str = Field(default="us-east-1")
    language: str = Field(default="en-US")


class DeepgramConfig(BaseModel):
    model: str = Field(default="nova-2")
    api_url: str = Field(default="https://api.deepgram.com/v1/listen")
    smart_format: bool = Field(default=True)
    punctuate: bool = Field(default=True)


class ElevenLabsConfig(BaseModel):
    model: str = Field(default="scribe_v1")
    upload_url: str = Field(default="https://api.elevenlabs.io/v1/speech-to-text")
    transcripts_url: str = Field(default="https://api.elevenlabs.io/v1/speech-to-text/transcripts")
    diarize: bool = Field(default=True)
    max_poll_retries: int = Field(default=60, description="Maximum polling attempts for transcript readiness")
    poll_interval: float = Field(default=1.0, description="Interval between polling attempts in seconds")


class VoiceEvaluationConfig(BaseModel):
    evaluation_model: str = Field(default="openai/gpt-4o-mini", description="Model to use for voice evaluation comparisons")
    temperature: float = Field(default=0.0, description="Temperature for evaluation model")


class Config(BaseModel):
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    cors: CORSConfig = Field(default_factory=CORSConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    deepgram: DeepgramConfig = Field(default_factory=DeepgramConfig)
    elevenlabs: ElevenLabsConfig = Field(default_factory=ElevenLabsConfig)
    voice_evaluation: VoiceEvaluationConfig = Field(default_factory=VoiceEvaluationConfig)
    log_level: str = Field(default="DEBUG")
    mgmt_url: str = Field(default="https://dev-apimgmt.salescode.ai/v1")
    deusmachina_url: str = Field(default="http://localhost:8080/internal/api/v1")

    @classmethod
    def from_env(cls) -> "Config":
        config = cls(
            tracing=TracingConfig(
                enabled=os.getenv("TRACING_ENABLED", "false").lower() == "true",
                tracer_url=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"),
                tracer_api_key=os.getenv("PHOENIX_API_KEY", ""),
            ),
            redis=RedisConfig(
                use_redis=os.getenv("USE_REDIS", "false").lower() == "true",
                redis_url=os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
            ),
            cors=CORSConfig(
                allowed_origins=os.getenv(
                    "CORS_ALLOWED_ORIGINS",
                    "https://app.salescode.ai,https://admin.salescode.ai,http://localhost:5173",
                ).split(","),
                allow_credentials=os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true",
                allow_methods=os.getenv("CORS_ALLOW_METHODS", "*").split(","),
                allow_headers=os.getenv("CORS_ALLOW_HEADERS", "*").split(","),
            ),
            llm=LLMConfig(
                max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
                fallback_model=os.getenv("FALLBACK_MODEL", "groq/llama-3.3-70b-versatile"),
                providers_without_api_key=os.getenv("PROVIDERS_WITHOUT_API_KEY", "bedrock").split(","),
            ),
            state=StateConfig(
                wait_timeout=float(os.getenv("STATE_WAIT_TIMEOUT", "5.0")),
                check_interval=float(os.getenv("STATE_CHECK_INTERVAL", "0.1")),
            ),
            transcription=TranscriptionConfig(
                strategy=os.getenv("TRANSCRIPTION_STRATEGY", "elevenlabs"),
                llm_model=os.getenv("TRANSCRIPTION_LLM_MODEL", "scribe_v1"),
                timeout=int(os.getenv("TRANSCRIPTION_TIMEOUT", "60")),
                aws_region=os.getenv("TRANSCRIPTION_AWS_REGION", "us-east-1"),
                language=os.getenv("TRANSCRIPTION_LANGUAGE", "en-US"),
            ),
            deepgram=DeepgramConfig(
                model=os.getenv("DEEPGRAM_MODEL", "nova-2"),
                api_url=os.getenv("DEEPGRAM_API_URL", "https://api.deepgram.com/v1/listen"),
                smart_format=os.getenv("DEEPGRAM_SMART_FORMAT", "true").lower() == "true",
                punctuate=os.getenv("DEEPGRAM_PUNCTUATE", "true").lower() == "true",
            ),
            elevenlabs=ElevenLabsConfig(
                model=os.getenv("ELEVENLABS_MODEL", "scribe_v1"),
                upload_url=os.getenv("ELEVENLABS_UPLOAD_URL", "https://api.elevenlabs.io/v1/speech-to-text"),
                transcripts_url=os.getenv("ELEVENLABS_TRANSCRIPTS_URL", "https://api.elevenlabs.io/v1/speech-to-text/transcripts"),
                diarize=os.getenv("ELEVENLABS_DIARIZE", "true").lower() == "true",
                max_poll_retries=int(os.getenv("ELEVENLABS_MAX_POLL_RETRIES", "60")),
                poll_interval=float(os.getenv("ELEVENLABS_POLL_INTERVAL", "1.0")),
            ),
            voice_evaluation=VoiceEvaluationConfig(
                evaluation_model=os.getenv("VOICE_EVAL_MODEL", "openai/gpt-4o-mini"),
                temperature=float(os.getenv("VOICE_EVAL_TEMPERATURE", "0.0")),
            ),
            log_level=os.getenv("DEUSAUDITRON_LOG_LEVEL", "DEBUG").upper(),
            mgmt_url=os.getenv("MGMT_URL", "https://dev-apimgmt.salescode.ai/v1"),
            deusmachina_url=os.getenv("DEUSMACHINA_URL", "http://localhost:8080/internal/api/v1"),
        )
        logger.info("Loaded Deusauditron configuration: {}", config.model_dump_json(indent=2))
        return config


_config_instance = None


def get_config() -> Config:
    global _config_instance
    if _config_instance is None:
        _config_instance = Config.from_env()
    return _config_instance

