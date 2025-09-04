from typing import List

from pydantic import BaseModel, ConfigDict, Field

from . import eval_rule_schema


class ModelResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str = Field(...)
    result: str = Field(...)
    reasoning: str = Field(...)


class EvaluationRuleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rule: eval_rule_schema.EvalRule
    model_results: List[ModelResult] = Field(...)

