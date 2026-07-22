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

## Deployment (Render)

The app is a single Docker service — FastAPI serves both the JSON API and
the static frontend, so there's only one thing to deploy.

1. Push this repo to GitHub (already done) and create a new **Web Service**
   on Render pointing at it — it will pick up `render.yaml` automatically
   (or use `Dockerfile` directly if you create the service manually).
2. Set `GROQ_API_KEY` and `GEMINI_API_KEY` in the Render dashboard's
   environment variables (marked `sync: false` in `render.yaml` so they're
   never committed).
3. Render health-checks `/health`.

**Note:** on Render's free tier there's no persistent disk, so the local
ChromaDB store under `data/chroma` is wiped on every deploy/restart —
uploaded papers need to be re-uploaded after a redeploy. Fine for a demo;
for durability, point `CHROMA_PERSIST_DIR` at a mounted disk on a paid plan
or swap in a hosted vector DB.

To build/run the same image locally:

```bash
docker build -t research-paper-rag .
docker run -p 8000:8000 --env-file .env research-paper-rag
```

## Project status

Built incrementally, one feature branch/PR per stage — all merged:

1. `chore/project-scaffold` — project layout, config, schemas
2. `feat/pdf-ingestion` — PDF parsing + chunking
3. `feat/vector-store` — ChromaDB wrapper
4. `feat/gemini-embeddings-upload` — embeddings + `/upload`
5. `feat/groq-rag-ask` — retrieval + `/ask`
6. `feat/frontend-ui` — upload/chat UI
7. `feat/deployment-config` — Docker/Render, error handling
