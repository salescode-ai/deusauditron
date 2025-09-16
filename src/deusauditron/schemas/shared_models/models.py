from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator
from deusauditron.schemas.autogen.references.llm_node_for_conversation_schema import LLMNodeForConversation
from deusauditron.schemas.autogen.references.global_llm_node_for_conversation_schema import GlobalLLMNodeForConversation
from deusauditron.schemas.autogen.references.model_schema import ModelConfig
from deusauditron.config import get_config


class VariableType(str, Enum):
    """Enum for different types of variables that can be used to configure LLM response processing."""

    NUM2WORDS = "num2words"
    VOCAB_REPLACE = "vocab_replace"
    SKIP_INTENT_UPON_INTERRUPTION = "skip_intent_upon_interruption"


class SchemaObject(BaseModel):
    name: str = "response_generic_schema"
    json_schema: Dict[str, Any]


class ResponseFormat(BaseModel):
    type: str = "json_schema"
    json_schema: SchemaObject


class LLMResponse(BaseModel):
    """Represents a response from an LLM, including content and optional metrics."""

    intent: str = ""
    intent_detector: str = ""
    raw_content: str = ""
    content: str = ""
    reasoning_content: str = ""
    error: bool = False
    error_text: str = ""
    node: str = ""
    src: str = ""
    usage: str = ""


class InteractionLog(BaseModel):
    user_message: str = ""
    llm_response: LLMResponse = Field(..., description="The LLM response")


class MiniInteractionLog(BaseModel):
    """Data class for mini interaction log."""

    user_message: str
    llm_response: str
    node: str


class MessageRole(str, Enum):
    """Enum for message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """Base class for all message types."""

    content: Union[str, List[Dict[str, Any]]] = Field(
        ...,
        description="The content of the message - can be string for text-only or list for multimodal content",
    )
    role: MessageRole = Field(
        ..., description="The role of the message sender", serialization_alias="role"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {"timestamp": str(datetime.now(timezone.utc).timestamp())}, 
        description="Metadata for the message including timestamp and optional node information"
    )


class LLMInteraction(BaseModel):
    user_message: str = Field(..., description="The user's input message")
    llm_response: LLMResponse = Field(..., description="The LLM response")


class SystemMessage(Message):
    """System message with predefined role."""

    role: MessageRole = Field(
        default=MessageRole.SYSTEM, description="System message role"
    )


class HumanMessage(Message):
    """Human message with predefined role."""

    role: MessageRole = Field(
        default=MessageRole.USER, description="Human message role"
    )

    content: List[Dict[str, Any]] = Field(  # type: ignore[override]
        ..., description="The content of a human message - must be a list"
    )


class AIMessage(Message):
    """AI message with predefined role."""

    role: MessageRole = Field(
        default=MessageRole.ASSISTANT, description="AI message role"
    )


class BaseRequest(BaseModel):
    """Base class for all request types with common fields."""

    tenant_id: str = Field(..., description="The tenant identifier")
    agent_id: str = Field(..., description="The agent identifier")
    run_id: str = Field(..., description="The run identifier")

    @property
    def composite_key(self) -> str:
        """Generate a unique composite key for the request."""
        return f"{self.tenant_id}:{self.agent_id}:{self.run_id}"


class AgentRunRequest(BaseRequest):
    """Request model for running an agent."""

    internal_request_id: UUID = Field(
        default_factory=UUID, description="Unique request identifier"
    )
    user_input: str = Field(..., description="The user's input message")
    audio_data: Optional[bytes] = Field(
        default=None, 
        description="Binary audio data",
        exclude=True
    )
    audio_format: Optional[str] = Field(default=None, description="Audio format (wav, mp3, m4a, flac, ogg, pcm)")
    streaming: bool = Field(default=False, description="Whether to stream the response")
    reasoning: bool = Field(
        default=False, description="Whether to send the reasoning response"
    )
    prev_ai_response: Optional[str] = Field(
        default=None, description="Previous AI response if any"
    )
    messages: List[Message] = Field(
        default_factory=list, description="Previous conversation turns"
    )
    entry_node_name: Optional[str] = Field(
        default=None, description="Previous node name if any"
    )
    node_overrides: Optional[Dict[str, LLMNodeForConversation | GlobalLLMNodeForConversation]] = Field(
        default=None, description="Override the node schemas only for this call"
    )
    variables: Dict[VariableType, Any] = Field(
        default_factory=dict,
        description="Variables to configure LLM response processing. "
        "Use 'vocab_replace': true/false, 'num2words': true/false, and 'skip_intent_upon_interruption': true/false",
        json_schema_extra={"example": {"vocab_replace": True, "num2words": True}},
    )
    image_url: Optional[str] = Field(
        default=None,
        description="Public URL for multimodal input",
    )
    override_model: Optional[str] = Field(
        default=None,
        description="Override the model for the run",
    )

    @field_validator("internal_request_id", mode="before")
    @classmethod
    def validate_uuid(cls, v):
        """Ensure UUID is properly formatted."""
        if isinstance(v, str):
            return UUID(v)
        return v


class CreatePayload(BaseModel):
    """Payload for agent creation endpoint."""

    metadata: Dict[str, Any] = Field(..., description="Agent metadata")
    blueprint: str = Field(..., description="Agent blueprint identifier")
    api_keys: Dict[str, str] = Field(
        default_factory=dict, description="API keys for the agent"
    )
    fail_on_missing_dynamic_data: bool = Field(
        default=False, 
        description="If true, agent creation will fail if any dynamic data cannot be fetched from external sources"
    )
    entry_node_name: Optional[str] = Field(
        default=None,
        description="Optional override for the initial entry node."
    )
    messages: List[Message] = Field(
        default_factory=list,
        description="Initial conversation history to seed the agent state."
    )


class AgentCreateRequest(BaseRequest):
    """Request model for creating an agent."""

    create_payload: CreatePayload = Field(..., description="Payload for agent creation")


class RunPayload(BaseModel):
    """Payload for agent run endpoint."""

    user_input: Optional[str] = Field(default="", description="The user's text input message")
    streaming: bool = Field(default=False, description="Whether to stream the response")
    reasoning: bool = Field(default=False, description="Whether to stream the response")
    prev_ai_response: Optional[str] = Field(
        default=None, description="Previous AI response if any"
    )
    entry_node_name: Optional[str] = Field(
        default=None, description="Previous node name if any"
    )
    node_overrides: Optional[Dict[str, LLMNodeForConversation | GlobalLLMNodeForConversation]] = Field(
        default=None, description="Override the node schemas only for this call"
    )
    messages: List[Message] = Field(
        default_factory=list, description="Previous conversation turns"
    )
    variables: Dict[VariableType, Any] = Field(
        default_factory=dict,
        description="Variables to configure LLM response processing. Use 'vocab_replace': true/false and 'num2words': true/false",
        json_schema_extra={"example": {"vocab_replace": True, "num2words": True}},
    )
    image_url: Optional[str] = Field(
        default=None,
        description="Public URL for multimodal input",
    )
    override_model: Optional[str] = Field(
        default=None,
        description="Override the model for the run",
    )


class CompletionsPayload(BaseModel):
    """Payload for stateless completions endpoint."""

    model: ModelConfig = Field(..., description="Model configuration including name and temperature")
    api_keys: Dict[str, str] = Field(
        default_factory=dict, description="API keys for LLM providers (e.g., openai, groq, anthropic)"
    )
    user_input: Optional[str] = Field(default="", description="The user's text input message")
    messages: List[Message] = Field(
        default_factory=list, description="Optional prior messages to include in the prompt"
    )
    image_url: Optional[str] = Field(
        default=None,
        description="Public URL or data URI for multimodal input",
    )
    streaming: bool = Field(default=False, description="Whether to stream the response")
    reasoning: bool = Field(default=False, description="Whether to include reasoning stream if supported")

# Validation-related models


class ValidationRule(BaseModel):
    """Model representing a validation rule result."""

    rule_name: str
    rule_description: str
    passed: bool
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


class ValidationResult(BaseModel):
    """Model representing the complete validation result."""

    flow_type: str
    overall_passed: bool
    total_rules_checked: int
    passed_rules: int
    failed_rules: int
    rules: List[ValidationRule]

    def add_rule_result(self, rule: ValidationRule) -> None:
        """Add a rule result to the validation result."""
        self.rules.append(rule)
        self.total_rules_checked += 1
        if rule.passed:
            self.passed_rules += 1
        else:
            self.failed_rules += 1
            self.overall_passed = False


class ValidatePayload(BaseModel):
    """Payload for blueprint validation endpoint."""

    metadata: Dict[str, Any] = Field(..., description="Agent metadata")
    blueprint: str = Field(..., description="Agent blueprint YAML content")
    api_keys: Dict[str, str] = Field(
        default_factory=dict, description="API keys for the agent"
    )


class SchemaResponse(BaseModel):
    """Response model for schema endpoint."""

    schemas: Dict[str, Any] = Field(..., description="The schemas")


class AgentValidateRequest(BaseRequest):
    """Request model for validating an agent blueprint."""

    validate_payload: ValidatePayload = Field(
        ..., description="Payload for blueprint validation"
    )


class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history endpoint."""

    messages: List[Message] = Field(
        ..., description="The conversation history messages"
    )
    total_messages: int = Field(
        ..., description="Total number of messages in the conversation"
    )


class EvaluationPayload(BaseModel):
    """Request model for evaluation endpoint."""

    api_keys: Dict[str, str] = Field(
        default_factory=dict, description="API keys for the agent"
    )
    node_names: List[str] = Field(
        default_factory=list,
        description="The names of the nodes to evaluate, this is optional and if not provided, all nodes will be evaluated",
    )
    auto_refine: bool = Field(
        default=False,
        description="Whether to auto-refine the evaluation results",
    )
    force: Optional[bool] = Field(
        default=None,
        description="Whether to force creation of a new evaluation, even if one already exists",
    )
    persist_path: Optional[str] = Field(
        default=None,
        description="Full S3 path for saving evaluation results. If provided, evaluation results will be saved to S3 at this exact path.",
    )


class AgentEvalRequest(BaseRequest):
    payload: Optional[EvaluationPayload] = Field(
        default=None, description="The payload for the evaluation"
    )


class JourneyResponse(BaseModel):
    """Response model for journey endpoint."""

    journey: Dict[str, Any] = Field(..., description="The journey")


class TranscriberType(str, Enum):
    LLM = "LLM"
    AWS = "AWS"
    DEEPGRAM = "DEEPGRAM"


class MinimalModelConfig(BaseModel):
    """Minimal model config allowing only a name. Useful for providers like Deepgram."""
    name: str


class TranscriberConfigModel(BaseModel):
    type: TranscriberType = Field(..., description="Transcription service type (LLM, AWS, DEEPGRAM)")
    model: Optional[ModelConfig | MinimalModelConfig] = Field(
        default=None, 
        description="Model configuration. For LLM, a model is required; for Deepgram, only 'name' is used if provided."
    )

    @model_validator(mode='after')
    def validate_config(self):
        if self.type == TranscriberType.LLM and not self.model:
            raise ValueError("model configuration is required when type is LLM")
        return self


class TranscriptionPayload(BaseModel):
    """Payload for transcription endpoint (excluding the binary audio file)."""
    
    transcriber_config: TranscriberConfigModel = Field(..., description="Transcriber configuration")
    language: Optional[str] = Field(
        default=None, 
        description="Language code for transcription (e.g., 'en-US', 'es-ES'). If not specified, automatic language detection will be used where supported."
    )
    api_keys: Dict[str, str] = Field(
        default_factory=dict, 
        description="API keys for transcription services (optional for AWS or local services)"
    )


class TranscriptionResponse(BaseModel):
    """Response model for transcription endpoint."""

    transcript: str = Field(..., description="The transcribed text")
    confidence: Optional[float] = Field(default=None, description="Confidence score (0.0-1.0)")
    duration: Optional[float] = Field(default=None, description="Processing duration in seconds")


class SpeechType(str, Enum):
    ELEVENLABS = "ELEVENLABS"
    CARTESIA = "CARTESIA"


class SpeechConfigModel(BaseModel):
    type: SpeechType = Field(..., description="Speech synthesis service type (ELEVENLABS, CARTESIA)")
    voice_id: Optional[str] = Field(
        default=None, 
        description="Voice ID for the speech service."
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Model ID for speech synthesis."
    )
    voice_settings: Optional[Dict[str, float]] = Field(
        default=None,
        description="Voice settings for speech synthesis."
    )
    language: Optional[str] = Field(
        default=None,
        description="Language code for speech synthesis (e.g., 'en', 'es', 'hi')."
    )

class SpeechPayload(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    speech_config: SpeechConfigModel = Field(..., description="Speech synthesis configuration")
    api_keys: Dict[str, str] = Field(
        default_factory=dict, 
        description="API keys for speech synthesis service."
    )

    @model_validator(mode='after')
    def validate_text(self):
        if not self.text.strip():
            raise ValueError("text cannot be empty")
        config = get_config()
        max_length = config.speech.max_text_length
        if len(self.text) > max_length:
            raise ValueError(f"text length cannot exceed {max_length} characters")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Hello! This is a test of the speech synthesis API.",
                "speech_config": {
                    "model_id": "eleven_monolingual_v1",
                    "type": "ELEVENLABS",
                    "voice_id": "21m00Tcm4TlvDq8ikWAM",
                    "language": "en",
                    "voice_settings": {
                        "similarity_boost": 0.5,
                        "stability": 0.5
                    }
                },
                "api_keys": {
                    "elevenlabs": "your_elevenlabs_api_key_here",
                    "cartesia": "your_cartesia_api_key_here"
                }
            }
        }
    }


class MessageType(str, Enum):
    CREATE_AGENT = "Create_Agent"
    DELETE_AGENT = "Delete_Agent"
    RUN_AGENT = "Run_Agent"
    UNKNOWN = "Unknown_Message_Type"
    ERROR = "Error_Processing_Request"