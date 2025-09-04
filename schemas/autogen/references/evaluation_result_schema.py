from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from . import auto_refine_schema, granular_evaluation_result_schema


class Status(Enum):
    Requested = "Requested"
    Submitted = "Submitted"
    In_Progress = "In Progress"
    Completed = "Completed"
    Error = "Error"


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Optional[Status] = Field(None)
    progress: Optional[float] = Field(None)
    start_time: Optional[datetime] = Field(None)
    end_time: Optional[datetime] = Field(None)
    evaluated_nodes: Optional[List[str]] = Field(None)
    turn_level_evaluations: Optional[
        List[granular_evaluation_result_schema.GranularEvaluationResults]
    ] = Field(None)
    node_level_evaluations: Optional[
        List[granular_evaluation_result_schema.GranularEvaluationResults]
    ] = Field(None)
    flow_level_evaluations: Optional[
        List[granular_evaluation_result_schema.GranularEvaluationResults]
    ] = Field(None)
    intent_level_evaluations: Optional[
        List[granular_evaluation_result_schema.GranularEvaluationResults]
    ] = Field(None)
    auto_refinements: Optional[List[auto_refine_schema.AutoRefine]] = Field(None)

