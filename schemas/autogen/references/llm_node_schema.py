from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, PositiveFloat

from . import eval_rule_schema, model_schema


class LLMNodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    structured_output: Optional[bool] = False
    last_n_turns_llm: int = 0
    last_n_turns_intent: int = 0
    system_prompt: str = ""
    overridden_global_prompt: Optional[str] = None
    next_node: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    model: model_schema.ModelConfig
    timeout: Optional[PositiveFloat] = None
    evaluation_rules: Optional[List[eval_rule_schema.EvalRule]] = None

