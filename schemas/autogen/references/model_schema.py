from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ReasoningEffort(Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ModelConfig(BaseModel):
    name: str = Field(...)
    temperature: float
    reasoning_effort: Optional[ReasoningEffort] = None
    response_schema: Optional[str] = Field(None)
    fallback_models: Optional[List[str]] = Field(None)

