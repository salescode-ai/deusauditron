from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Result(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    N_A_ = "N.A."
    NO_EVAL = "NO_EVAL"


class EvalLLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    result: Result = Field(...)
    reason: str = Field(...)

