from sqlalchemy import select

from app.models.db_models import ChatHistory


async def save_message(session_id: str, role: str, content: str, db_session, commit: bool = True):
    """Save a single chat message to the chat_history table."""
    msg = ChatHistory(session_id=session_id, role=role, content=content)
    db_session.add(msg)
    if commit:
        await db_session.commit()


async def get_history(session_id: str, db_session) -> list[dict]:
    """Get full chat history for a session, ordered by timestamp ascending."""
    result = await db_session.execute(
        select(ChatHistory)
        .where(ChatHistory.session_id == session_id)
        .order_by(ChatHistory.timestamp.asc())
    )
    rows = result.scalars().all()
    return [
        {
            "role": row.role,
            "content": row.content,
            "timestamp": row.timestamp.isoformat(),
        }
        for row in rows
    ]


async def get_recent_history(session_id: str, db_session, limit: int = 10) -> list[dict]:
    """Get the last N messages for reformulation context."""
    result = await db_session.execute(
        select(ChatHistory)
        .where(ChatHistory.session_id == session_id)
        .order_by(ChatHistory.timestamp.desc())
        .limit(limit)
    )
    rows = result.scalars().all()
    # Reverse to get chronological order
    rows.reverse()
    return [{"role": row.role, "content": row.content} for row in rows]
