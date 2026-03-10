from sqlalchemy import Column, String, Text, Integer, Float, Boolean, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class ParentChunk(Base):
    __tablename__ = "parent_chunks"

    parent_id = Column(String, primary_key=True)
    title = Column(Text, nullable=False)
    source = Column(Text, nullable=False)
    type = Column(Text, nullable=False)
    text = Column(Text, nullable=False)
    word_count = Column(Integer)
    chunk_num = Column(Integer)


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())


class QALog(Base):
    __tablename__ = "qa_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    detected_lang = Column(String)
    translated_query = Column(Text)
    answer_mode = Column(String)
    model_used = Column(String)
    context_found = Column(Boolean)
    top_score = Column(Float)
    sources = Column(JSONB)
    latency_ms = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
