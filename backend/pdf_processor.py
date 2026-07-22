from dataclasses import dataclass

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    text: str
    page_number: int
    chunk_index: int


def extract_pages(file_path: str) -> list[tuple[int, str]]:
    """Return (1-indexed page number, page text) for every non-empty page."""
    pages = []
    with fitz.open(file_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages.append((i + 1, text))
    return pages


def chunk_pages(
    pages: list[tuple[int, str]], chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Split page text into overlapping chunks, keeping each chunk tagged
    with the page it came from so answers can cite a page number."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks: list[Chunk] = []
    for page_number, text in pages:
        for piece in splitter.split_text(text):
            chunks.append(
                Chunk(text=piece, page_number=page_number, chunk_index=len(chunks))
            )
    return chunks


def process_pdf(
    file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> tuple[list[Chunk], int]:
    """Parse a PDF into citation-tagged chunks. Returns (chunks, page_count)."""
    pages = extract_pages(file_path)
    if not pages:
        raise ValueError("No extractable text found in PDF")
    chunks = chunk_pages(pages, chunk_size, chunk_overlap)
    return chunks, len(pages)
