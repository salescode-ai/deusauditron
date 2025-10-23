# Generated from JSON Schema
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class VoiceEvalStatus(str, Enum):
    Requested = "Requested"
    In_Progress = "In_Progress"
    Completed = "Completed"
    Error = "Error"


class VoiceEvalCategory(str, Enum):
    STT = "STT"  # Speech-to-Text evaluation
    TTS = "TTS"  # Text-to-Speech evaluation


class VoiceComparisonResult(BaseModel):
    """Result of comparing generated text vs actual text"""
    
    accuracy_score: float = Field(
        ..., 
        description="Accuracy score between 0-100 representing how well the generated text matches actual text",
        ge=0,
        le=100
    )
    token_mismatches: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Token-level substitutions/insertions/deletions across aligned lines"
    )


class VoiceEvaluationDetails(BaseModel):
    """Detailed results for either STT or TTS evaluation"""
    
    category: VoiceEvalCategory = Field(
        ..., description="Type of voice evaluation (STT or TTS)"
    )
    generated_text: str = Field(
        ..., description="Text generated from audio conversion"
    )
    generated_text_clean: Optional[str] = Field(
        default=None, description="Cleaned/normalized generated text used for comparison"
    )
    actual_text: str = Field(
        ..., description="Actual text from transcript"
    )
    actual_text_clean: Optional[str] = Field(
        default=None, description="Cleaned/normalized actual text used for comparison"
    )
    comparison_result: VoiceComparisonResult = Field(
        ..., description="Detailed comparison results"
    )
    processing_time_ms: Optional[int] = Field(
        default=None, description="Time taken for this evaluation in milliseconds"
    )


class VoiceEvaluationResult(BaseModel):
    """Complete voice evaluation result containing both STT and TTS evaluations"""
    
    status: VoiceEvalStatus = Field(
        ..., description="Status of the voice evaluation"
    )
    progress: int = Field(
        default=0,
        description="Progress percentage (0-100)",
        ge=0,
        le=100
    )
    start_time: Optional[datetime] = Field(
        default=None, description="When the evaluation started"
    )
    end_time: Optional[datetime] = Field(
        default=None, description="When the evaluation completed"
    )
    stt_evaluation: Optional[VoiceEvaluationDetails] = Field(
        default=None, description="Speech-to-Text evaluation results"
    )
    tts_evaluation: Optional[VoiceEvaluationDetails] = Field(
        default=None, description="Text-to-Speech evaluation results"
    )
    overall_accuracy: Optional[float] = Field(
        default=None,
        description="Overall accuracy score combining STT and TTS results",
        ge=0,
        le=100
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if evaluation failed"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the evaluation"
    )
