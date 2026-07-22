import chromadb
from chromadb.config import Settings

from pdf_processor import Chunk


def _collection_name(doc_id: str) -> str:
    return f"doc_{doc_id}"


class VectorStore:
    def __init__(self, persist_dir: str):
        self.client = chromadb.PersistentClient(
            path=persist_dir, settings=Settings(anonymized_telemetry=False)
        )

    def add_chunks(
        self,
        doc_id: str,
        filename: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        collection = self.client.get_or_create_collection(
            name=_collection_name(doc_id), metadata={"filename": filename}
        )
        collection.add(
            ids=[str(c.chunk_index) for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[{"page_number": c.page_number} for c in chunks],
        )

    def query(
        self, doc_id: str, query_embedding: list[float], k: int
    ) -> list[dict]:
        collection = self.client.get_collection(name=_collection_name(doc_id))
        results = collection.query(query_embeddings=[query_embedding], n_results=k)
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        return [
            {"text": doc, "page_number": meta["page_number"]}
            for doc, meta in zip(documents, metadatas)
        ]

    def document_exists(self, doc_id: str) -> bool:
        try:
            self.client.get_collection(name=_collection_name(doc_id))
            return True
        except ValueError:
            return False

    def list_documents(self) -> list[dict]:
        return [
            {
                "doc_id": c.name.removeprefix("doc_"),
                "filename": c.metadata.get("filename", "unknown"),
                "num_chunks": c.count(),
            }
            for c in self.client.list_collections()
        ]
