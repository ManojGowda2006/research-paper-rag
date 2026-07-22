import google.generativeai as genai

from config import settings

genai.configure(api_key=settings.gemini_api_key)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunk texts for storage. Batches internally for large inputs."""
    result = genai.embed_content(
        model=settings.gemini_embedding_model,
        content=texts,
        task_type="retrieval_document",
    )
    return result["embedding"]


def embed_query(text: str) -> list[float]:
    """Embed a user question for similarity search."""
    result = genai.embed_content(
        model=settings.gemini_embedding_model,
        content=text,
        task_type="retrieval_query",
    )
    return result["embedding"]
