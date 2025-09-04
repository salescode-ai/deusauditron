from typing import List

from pydantic import BaseModel, ConfigDict, Field

from . import model_schema


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    turn_eval_models: List[model_schema.ModelConfig] = Field(..., min_length=1)
    node_eval_models: List[model_schema.ModelConfig] = Field(..., min_length=1)
    conversation_eval_models: List[model_schema.ModelConfig] = Field(..., min_length=1)
    intent_eval_models: List[model_schema.ModelConfig] = Field(..., min_length=1)
    auto_refine_model: model_schema.ModelConfig

