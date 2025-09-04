from typing import Optional

from pydantic import BaseModel, Field

from schemas.autogen.references.evaluation_result_schema import EvaluationResult


class EvalState(BaseModel):
    evaluation_result: Optional[EvaluationResult] = Field(default=None)

    def get_evaluation_result(self) -> Optional[EvaluationResult]:
        return self.evaluation_result

    def set_evaluation_result(self, value: EvaluationResult) -> None:
        self.evaluation_result = value

