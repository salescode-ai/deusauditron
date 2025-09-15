from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class VariableType(str, Enum):
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
    user_message: str
    llm_response: str
    node: str


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    content: Union[str, List[Dict[str, Any]]] = Field(...)
    role: MessageRole = Field(..., serialization_alias="role")
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "timestamp": str(datetime.now(timezone.utc).timestamp())
        }
    )


class LLMInteraction(BaseModel):
    user_message: str = Field(...)
    llm_response: LLMResponse = Field(...)


class SystemMessage(Message):
    role: MessageRole = Field(default=MessageRole.SYSTEM)


class HumanMessage(Message):
    role: MessageRole = Field(default=MessageRole.USER)
    content: List[Dict[str, Any]] = Field(...)  # type: ignore[override]


class AIMessage(Message):
    role: MessageRole = Field(default=MessageRole.ASSISTANT)


class BaseRequest(BaseModel):
    tenant_id: str = Field(...)
    agent_id: str = Field(...)
    run_id: str = Field(...)

    @property
    def composite_key(self) -> str:
        return f"{self.tenant_id}:{self.agent_id}:{self.run_id}"


class AgentRunRequest(BaseRequest):
    internal_request_id: UUID = Field(default_factory=UUID)
    user_input: str = Field(...)
    audio_data: Optional[bytes] = Field(default=None, exclude=True)
    audio_format: Optional[str] = Field(default=None)
    streaming: bool = Field(default=False)
    reasoning: bool = Field(default=False)
    prev_ai_response: Optional[str] = Field(default=None)
    messages: List[Message] = Field(default_factory=list)
    entry_node_name: Optional[str] = Field(default=None)
    node_overrides: Optional[Dict[str, Any]] = Field(default=None)
    variables: Dict[VariableType, Any] = Field(default_factory=dict)
    image_url: Optional[str] = Field(default=None)

    @field_validator("internal_request_id", mode="before")
    @classmethod
    def validate_uuid(cls, v):
        if isinstance(v, str):
            return UUID(v)
        return v


class CreatePayload(BaseModel):
    metadata: Dict[str, Any] = Field(...)
    blueprint: str = Field(...)
    api_keys: Dict[str, str] = Field(default_factory=dict)
    fail_on_missing_dynamic_data: bool = Field(default=True)
    entry_node_name: Optional[str] = Field(default=None)
    messages: List[Message] = Field(default_factory=list)


class AgentCreateRequest(BaseRequest):
    create_payload: CreatePayload = Field(...)


class RunPayload(BaseModel):
    user_input: Optional[str] = Field(default="")
    streaming: bool = Field(default=False)
    reasoning: bool = Field(default=False)
    prev_ai_response: Optional[str] = Field(default=None)
    entry_node_name: Optional[str] = Field(default=None)
    node_overrides: Optional[Dict[str, Any]] = Field(default=None)
    messages: List[Message] = Field(default_factory=list)
    variables: Dict[VariableType, Any] = Field(default_factory=dict)
    image_url: Optional[str] = Field(default=None)


class CompletionsPayload(BaseModel):
    model: Dict[str, Any] = Field(...)
    api_keys: Dict[str, str] = Field(default_factory=dict)
    user_input: Optional[str] = Field(default="")
    messages: List[Message] = Field(default_factory=list)
    image_url: Optional[str] = Field(default=None)
    streaming: bool = Field(default=False)
    reasoning: bool = Field(default=False)


class ValidationRule(BaseModel):
    rule_name: str
    rule_description: str
    passed: bool
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


class ValidationResult(BaseModel):
    flow_type: str
    overall_passed: bool
    total_rules_checked: int
    passed_rules: int
    failed_rules: int
    rules: List[ValidationRule]

    def add_rule_result(self, rule: ValidationRule) -> None:
        self.rules.append(rule)
        self.total_rules_checked += 1
        if rule.passed:
            self.passed_rules += 1
        else:
            self.failed_rules += 1
            self.overall_passed = False


class ValidatePayload(BaseModel):
    metadata: Dict[str, Any] = Field(...)
    blueprint: str = Field(...)
    api_keys: Dict[str, str] = Field(default_factory=dict)


class SchemaResponse(BaseModel):
    schemas: Dict[str, Any] = Field(...)


class AgentValidateRequest(BaseRequest):
    validate_payload: ValidatePayload = Field(...)


class ConversationHistoryResponse(BaseModel):
    messages: List[Message] = Field(...)
    total_messages: int = Field(...)


class EvaluationPayload(BaseModel):
    api_keys: Dict[str, str] = Field(default_factory=dict)
    node_names: List[str] = Field(default_factory=list)
    auto_refine: bool = Field(default=False)
    force: Optional[bool] = Field(default=None)
    persist_path: Optional[str] = Field(default=None)


class AgentEvalRequest(BaseRequest):
    payload: Optional[EvaluationPayload] = Field(default=None)


class JourneyResponse(BaseModel):
    journey: Dict[str, Any] = Field(...)


class ScenarioPayload(BaseModel):
    metadata: Dict[str, Any] = Field(...)
    blueprint: str = Field(...)
    transcript: List[Message] = Field(...)
    expected_output: str = Field(...)
    dynamic_data: Optional[Dict[str, Any]] = Field(default=None)
    replay: bool = Field(default=False)
