from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
import redis.asyncio as redis

from app.config import Settings


def create_engine(settings: Settings):
    return create_async_engine(
        settings.DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=300,
    )


def create_session_factory(engine):
    return async_sessionmaker(engine, expire_on_commit=False)


def create_redis_client(settings: Settings):
    return redis.from_url(settings.REDIS_URL, decode_responses=True)
