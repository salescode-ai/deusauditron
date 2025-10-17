from typing import Optional

from pydantic import BaseModel, Field

from deusauditron.schemas.autogen.references.evaluation_result_schema import EvaluationResult
from deusauditron.schemas.autogen.references.voice_evaluation_result_schema import VoiceEvaluationResult


class EvalState(BaseModel):
    evaluation_result: Optional[EvaluationResult] = Field(default=None)

    def get_evaluation_result(self) -> Optional[EvaluationResult]:
        return self.evaluation_result

    def set_evaluation_result(self, value: EvaluationResult) -> None:
        self.evaluation_result = value


class VoiceEvalState(BaseModel):
    voice_evaluation_result: Optional[VoiceEvaluationResult] = Field(default=None)

    def get_voice_evaluation_result(self) -> Optional[VoiceEvaluationResult]:
        return self.voice_evaluation_result

    def set_voice_evaluation_result(self, value: VoiceEvaluationResult) -> None:
        self.voice_evaluation_result = value


# Ensure forward references are resolved
EvalState.model_rebuild()
VoiceEvalState.model_rebuild()

