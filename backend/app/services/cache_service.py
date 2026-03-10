import hashlib
import json


def cache_key(prefix: str, text: str) -> str:
    normalized = text.strip().lower()
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    return f"{prefix}:{digest}"


async def get_cached_answer(query: str, redis_client) -> dict | None:
    key = cache_key("answer", query)
    data = await redis_client.get(key)
    return json.loads(data) if data else None


async def set_cached_answer(
    query: str, response: dict, redis_client, ttl: int = 3600
):
    key = cache_key("answer", query)
    await redis_client.set(key, json.dumps(response), ex=ttl)


async def get_cached_translation(query: str, redis_client) -> str | None:
    key = cache_key("translate", query)
    return await redis_client.get(key)


async def set_cached_translation(
    query: str, translation: str, redis_client, ttl: int = 86400
):
    key = cache_key("translate", query)
    await redis_client.set(key, translation, ex=ttl)
