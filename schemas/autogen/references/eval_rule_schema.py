from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Type(Enum):
    turn = "turn"
    node = "node"
    flow = "flow"
    intent = "intent"


class EvalRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(...)
    instruction: str = Field(...)
    type: Type = Field(...)

