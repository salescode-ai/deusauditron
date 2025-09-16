from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from deusauditron.schemas.autogen.references.eval_rule_schema import EvalRule
from deusauditron.schemas.autogen.references.evaluation_schema import EvaluationConfig
from deusauditron.schemas.autogen.references.llm_node_schema import LLMNodeConfig
from deusauditron.schemas.autogen.references.llm_node_for_conversation_schema import LLMNodeForConversation
from deusauditron.schemas.autogen.references.global_llm_node_for_conversation_schema import GlobalLLMNodeForConversation


class BaseState(BaseModel):
    """
    Minimal agent state required by Deusauditron evaluation.
    Extra fields are allowed to maintain forward compatibility.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    api_keys: Dict[str, str] = Field(default_factory=dict)
    global_llm_nodes: Dict[str, GlobalLLMNodeForConversation] = Field(default_factory=dict)
    flow_data_llm_nodes: Dict[str, LLMNodeForConversation] = Field(default_factory=dict)
    flow_data_valid_intents: Dict[str, List[str]] = Field(default_factory=dict)
    flow_data_intent_instructions: Dict[str, List[str]] = Field(default_factory=dict)
    global_evaluation_rules: List[EvalRule] = Field(default_factory=list)
    evaluation_config: Optional[EvaluationConfig] = Field(default=None)

    def get_evaluation_config(self) -> Optional[EvaluationConfig]:
        return self.evaluation_config

    def get_flow_data_llm_nodes(self) -> Dict[str, LLMNodeForConversation]:
        return self.flow_data_llm_nodes

    def get_global_llm_nodes(self) -> Dict[str, GlobalLLMNodeForConversation]:
        return self.global_llm_nodes

    def get_flow_data_valid_intents(self) -> Dict[str, List[str]]:
        return self.flow_data_valid_intents

    def get_flow_data_intent_instructions(self) -> Dict[str, List[str]]:
        return self.flow_data_intent_instructions

    def get_global_evaluation_rules(self) -> List[EvalRule]:
        return self.global_evaluation_rules

