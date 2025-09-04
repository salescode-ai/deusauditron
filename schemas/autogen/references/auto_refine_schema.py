from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class NodeRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node: str = Field(...)
    prompt: str = Field(...)


class IntentRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node: str = Field(...)
    intent_instructions: Dict[str, str] = Field(...)


class AutoRefine(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_recommendations: Optional[List[NodeRecommendation]] = Field(default=[])
    global_recommendations: Optional[str] = Field(default="")
    intent_recommendations: Optional[List[IntentRecommendation]] = Field(default=[])

