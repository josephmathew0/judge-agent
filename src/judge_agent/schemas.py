from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional
from pydantic import field_serializer

OriginLabel = Literal["ai_generated", "human_generated"]


class OriginPrediction(BaseModel):
    label: OriginLabel
    confidence: float = Field(ge=0.0, le=1.0)

    @field_serializer("confidence")
    def _round_confidence(self, v: float):
        return round(v, 3)


class AudienceSegment(BaseModel):
    segment: str
    likelihood: float = Field(ge=0.0, le=1.0)
    why: str


class JudgeOutput(BaseModel):
    origin_prediction: OriginPrediction
    virality_score: int = Field(ge=0, le=100)
    distribution_analysis: List[AudienceSegment]
    explanations: Dict[str, str]
    debug: Optional[Dict[str, Any]] = None
