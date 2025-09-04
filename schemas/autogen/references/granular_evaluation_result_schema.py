from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field

from . import evaluation_rule_result_schema


class Category(Enum):
    Turn = "Turn"
    Intent = "Intent"
    Node = "Node"
    Flow = "Flow"


class GranularEvaluationResults(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Category = Field(...)
    input_text: str = Field(...)
    node: str = Field(...)
    rule_results: List[evaluation_rule_result_schema.EvaluationRuleResult] = Field(...)

