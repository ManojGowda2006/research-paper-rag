import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile

from config import settings
from embeddings import embed_documents
from pdf_processor import process_pdf
from schemas import DocumentInfo, UploadResponse
from vector_store import VectorStore

app = FastAPI(title="Research Paper RAG")
vector_store = VectorStore(settings.chroma_persist_dir)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    return vector_store.list_documents()


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    doc_id = uuid.uuid4().hex
    dest_path = Path(settings.upload_dir) / f"{doc_id}.pdf"
    dest_path.write_bytes(await file.read())

    try:
        chunks, num_pages = process_pdf(
            str(dest_path), settings.chunk_size, settings.chunk_overlap
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        dest_path.unlink(missing_ok=True)

    embeddings = embed_documents([c.text for c in chunks])
    vector_store.add_chunks(doc_id, file.filename, chunks, embeddings)

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        num_chunks=len(chunks),
        num_pages=num_pages,
    )
