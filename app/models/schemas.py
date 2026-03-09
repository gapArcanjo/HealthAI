from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    SYMPTOMS = "symptoms"
    SCHEDULING = "scheduling"
    CANCELLATION = "cancellation"
    EXAM_INQUIRY = "exam_inquiry"
    GENERAL = "general"


class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EntityExtraction(BaseModel):
    symptoms: list[str] = Field(default_factory=list)


class NLPAnalysis(BaseModel):
    intent: IntentType = IntentType.GENERAL
    intent_confidence: float = 0.0
    urgency: UrgencyLevel = UrgencyLevel.LOW
    urgency_confidence: float = 0.0
    entities: EntityExtraction = Field(default_factory=EntityExtraction)
    is_emergency: bool = False
    recommended_specialty: Optional[str] = None


class MessageRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class MessageResponse(BaseModel):
    session_id: str
    message: str
    nlp_analysis: NLPAnalysis
    suggested_actions: list[str] = Field(default_factory=list)
    emergency_protocol: bool = False
    emergency_message: Optional[str] = None
