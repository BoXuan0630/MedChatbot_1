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
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


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


class EvaluationRequest(BaseModel):
    test_cases: list[TestCase]
    use_reranking: bool = True
