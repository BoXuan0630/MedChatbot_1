from datetime import datetime

from pydantic import BaseModel, field_validator


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    use_reranking: bool = True
    session_id: str = ""

    @field_validator("question")
    @classmethod
    def validate_question(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        if len(v) > 1000:
            raise ValueError("Question must be 1000 characters or less")
        return v

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v):
        if len(v) > 100:
            raise ValueError("session_id must be 100 characters or less")
        return v


class Source(BaseModel):
    title: str
    source: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    detected_lang: str
    translated_query: str
    answer_mode: str
    model_used: str = "unknown"
    context_found: bool
    top_score: float
    session_id: str
    cached: bool = False
    follow_ups: list[str] = []


class HistoryMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]


class TestCase(BaseModel):
    question: str
    ground_truth: str
    relevant_sources: list[str] = []


class FeedbackRequest(BaseModel):
    session_id: str
    question: str
    rating: str  # "up" or "down"
    comment: str = ""

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v):
        if v not in ("up", "down"):
            raise ValueError("Rating must be 'up' or 'down'")
        return v


class EvaluationRequest(BaseModel):
    test_cases: list[TestCase]
    use_reranking: bool = True
