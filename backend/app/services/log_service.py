from app.models.db_models import QALog


async def log_qa(
    session_id: str,
    question: str,
    answer: str,
    detected_lang: str,
    translated_query: str,
    answer_mode: str,
    model_used: str,
    context_found: bool,
    top_score: float,
    sources: list[dict],
    latency_ms: float,
    db_session,
    commit: bool = True,
):
    """Log a Q&A interaction to the qa_logs table."""
    log_entry = QALog(
        session_id=session_id,
        question=question,
        answer=answer,
        detected_lang=detected_lang,
        translated_query=translated_query,
        answer_mode=answer_mode,
        model_used=model_used,
        context_found=context_found,
        top_score=top_score,
        sources=[{"title": s["title"], "source": s["source"]} for s in sources],
        latency_ms=latency_ms,
    )
    db_session.add(log_entry)
    if commit:
        await db_session.commit()
