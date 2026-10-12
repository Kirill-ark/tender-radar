"""Tender ingestion: download PDF -> parse -> chunk -> embed -> write to Chroma.

Parsing and chunking are taken from the Chat-With-Your-Docs project; new here
are the download step with size limits and the tender_id in chunk metadata.
"""

import threading

import chromadb
import fitz  # PyMuPDF
import httpx
from sentence_transformers import SentenceTransformer

from app.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DOWNLOAD_TIMEOUT_SECONDS,
    EMBEDDING_MODEL,
    MAX_PDF_BYTES,
    MAX_PDF_PAGES,
)

_embedder = None
_embedder_lock = threading.Lock()
_client = None
_client_lock = threading.Lock()


class DocumentRejected(ValueError):
    """The document breaks a guardrail limit (too big, too many pages, not a PDF)."""


def get_embedder() -> SentenceTransformer:
    """Load the embedding model once per process and reuse it.

    Ingest and question answering must use the SAME model, otherwise document
    and query vectors live in different spaces and search returns garbage.
    """
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def get_collection() -> chromadb.Collection:
    """Open (or create) the persisted Chroma collection, cosine similarity."""
    global _client
    with _client_lock:
        if _client is None:
            _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def download_pdf(url: str) -> bytes:
    """Download a PDF, aborting as soon as it exceeds MAX_PDF_BYTES.

    Streaming lets us stop early instead of first loading a huge file into memory.
    """
    with httpx.stream(
        "GET", url, timeout=DOWNLOAD_TIMEOUT_SECONDS, follow_redirects=True
    ) as response:
        response.raise_for_status()
        data = bytearray()
        for block in response.iter_bytes():
            data.extend(block)
            if len(data) > MAX_PDF_BYTES:
                raise DocumentRejected(f"PDF is larger than {MAX_PDF_BYTES} bytes")
    if not data.startswith(b"%PDF"):
        raise DocumentRejected("downloaded file is not a PDF")
    return bytes(data)


def parse_pdf(pdf_bytes: bytes, filename: str) -> list[dict]:
    """Extract text page by page, keeping filename + page for citations."""
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.page_count > MAX_PDF_PAGES:
            raise DocumentRejected(
                f"PDF has {doc.page_count} pages, limit is {MAX_PDF_PAGES}"
            )
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:  # skip blank / image-only pages
                pages.append({"text": text, "file": filename, "page": page_number})
    return pages


def chunk_text(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping chunks with a sliding window.

    Neighbouring chunks share `overlap` characters, so a sentence cut at one
    chunk's boundary survives intact at the start of the next one.
    """
    if not 0 <= overlap < size:
        raise ValueError("overlap must be non-negative and smaller than chunk size")
    step = size - overlap
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start : start + size]
        if chunk.strip():
            chunks.append(chunk)
        if start + size >= len(text):
            break  # the window already covered the tail; avoid tiny duplicates
    return chunks


def ingest_tender(tender_id: str, pdf_url: str) -> int:
    """Download and index one tender PDF. Returns the number of stored chunks.

    Re-ingesting a tender REPLACES its chunks: old ones are deleted first, so a
    corrected notice leaves no stale text behind.
    """
    filename = f"{tender_id}.pdf"
    pages = parse_pdf(download_pdf(pdf_url), filename)

    documents, metadatas, ids = [], [], []
    for page in pages:
        for i, chunk in enumerate(chunk_text(page["text"])):
            documents.append(chunk)
            metadatas.append(
                {"tender_id": tender_id, "file": page["file"], "page": page["page"]}
            )
            ids.append(f"{tender_id}_p{page['page']}_c{i}")

    collection = get_collection()
    collection.delete(where={"tender_id": tender_id})
    if not documents:
        return 0

    embeddings = get_embedder().encode(documents, show_progress_bar=False).tolist()
    collection.upsert(
        ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )
    return len(documents)
