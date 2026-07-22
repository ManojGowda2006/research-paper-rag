# research-paper-rag

Upload a PDF research paper, ask questions about it, get answers grounded in
the paper with page citations.

`PDF -> chunk -> Gemini embeddings -> ChromaDB -> retrieve -> Groq LLM -> cited answer`

## Stack

- **Backend**: FastAPI
- **PDF parsing**: PyMuPDF
- **Chunking**: LangChain `RecursiveCharacterTextSplitter`
- **Embeddings**: Gemini `text-embedding-004`
- **Vector store**: ChromaDB (persistent, local)
- **Generation**: Groq (`llama-3.3-70b-versatile`)
- **Frontend**: static HTML/JS served by FastAPI

## Local setup

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env   # fill in GROQ_API_KEY and GEMINI_API_KEY
uvicorn main:app --reload
```

## Project status

Being built incrementally, one feature branch/PR per stage:

1. `chore/project-scaffold` — project layout, config, schemas
2. `feat/pdf-ingestion` — PDF parsing + chunking
3. `feat/vector-store` — ChromaDB wrapper
4. `feat/gemini-embeddings-upload` — embeddings + `/upload`
5. `feat/groq-rag-ask` — retrieval + `/ask`
6. `feat/frontend-ui` — upload/chat UI
7. `feat/deployment-config` — Docker/Render, error handling
