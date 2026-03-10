from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX: str

    # Google
    GEMINI_API_KEY: str

    # MedGemma (empty string = not deployed yet, skip to Gemini)
    MEDGEMMA_URL: str = ""

    # Gemini model name
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # PostgreSQL
    DATABASE_URL: str  # postgresql+asyncpg://user:pass@host:5432/medbot

    # Redis
    REDIS_URL: str  # rediss://default:password@host:6380/0

    # Thresholds / constants
    CONTEXT_THRESHOLD: float = 0.70
    TOP_K_RETRIEVAL: int = 20
    TOP_K_RERANK: int = 5
    RRF_K: int = 60
    MEDGEMMA_TIMEOUT: int = 60
    REDIS_ANSWER_TTL: int = 3600      # 1 hour
    REDIS_TRANSLATE_TTL: int = 86400  # 24 hours
    REDIS_SESSION_TTL: int = 1800     # 30 minutes

    model_config = SettingsConfigDict(env_file=".env")
