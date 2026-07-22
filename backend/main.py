import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from config import settings
from embeddings import embed_documents, embed_query
from pdf_processor import process_pdf
from rag import generate_answer
from schemas import AskRequest, AskResponse, DocumentInfo, SourceChunk, UploadResponse
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

    try:
        embeddings = embed_documents([c.text for c in chunks])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {e}")

    vector_store.add_chunks(doc_id, file.filename, chunks, embeddings)

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        num_chunks=len(chunks),
        num_pages=num_pages,
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if not vector_store.document_exists(request.doc_id):
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        query_embedding = embed_query(request.question)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {e}")

    retrieved = vector_store.query(
        request.doc_id, query_embedding, settings.retrieval_k
    )
    if not retrieved:
        raise HTTPException(
            status_code=404, detail="No indexed content for this document"
        )

    try:
        answer = generate_answer(request.question, retrieved)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Answer generation failed: {e}")
    sources = [
        SourceChunk(page_number=r["page_number"], text=r["text"]) for r in retrieved
    ]
    return AskResponse(answer=answer, sources=sources)


# Mounted last so it doesn't shadow the API routes above.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
