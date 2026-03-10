"""One-time script: load parents.json into the parent_chunks PostgreSQL table."""

import asyncio
import json
import os
import sys

# Ensure the backend directory is on the path when running as a module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.config import Settings
from app.database import create_engine, create_session_factory
from app.models.db_models import Base, ParentChunk

from sqlalchemy.dialects.postgresql import insert


async def seed():
    settings = Settings()
    engine = create_engine(settings)

    # Create tables if they don't exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = create_session_factory(engine)

    # Load parents.json
    parents_path = os.path.join(os.path.dirname(__file__), "..", "..", "parents.json")
    with open(parents_path, encoding="utf-8") as f:
        parents = json.load(f)

    print(f"Loaded {len(parents)} parent chunks from parents.json")

    # Upsert in batches
    batch_size = 500
    items = list(parents.values())

    async with session_factory() as session:
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            stmt = insert(ParentChunk).values([
                {
                    "parent_id": item["parent_id"],
                    "title": item["title"],
                    "source": item["source"],
                    "type": item["type"],
                    "text": item["text"],
                    "word_count": item.get("word_count"),
                    "chunk_num": item.get("chunk_num"),
                }
                for item in batch
            ])
            # On conflict, update the text (idempotent)
            stmt = stmt.on_conflict_do_update(
                index_elements=["parent_id"],
                set_={
                    "title": stmt.excluded.title,
                    "source": stmt.excluded.source,
                    "type": stmt.excluded.type,
                    "text": stmt.excluded.text,
                    "word_count": stmt.excluded.word_count,
                    "chunk_num": stmt.excluded.chunk_num,
                },
            )
            await session.execute(stmt)
            print(f"  Upserted {min(i + batch_size, len(items))}/{len(items)}")

        await session.commit()

    await engine.dispose()
    print(f"Done! Seeded {len(parents)} parent chunks into PostgreSQL.")


if __name__ == "__main__":
    asyncio.run(seed())
