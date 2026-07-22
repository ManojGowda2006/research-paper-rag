from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str
    gemini_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_embedding_model: str = "models/text-embedding-004"
    chroma_persist_dir: str = "data/chroma"
    upload_dir: str = "uploads"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 4


settings = Settings()

Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
