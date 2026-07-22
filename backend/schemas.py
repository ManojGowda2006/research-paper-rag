from pydantic import BaseModel


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int
    num_pages: int


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int


class AskRequest(BaseModel):
    doc_id: str
    question: str


class SourceChunk(BaseModel):
    page_number: int
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
