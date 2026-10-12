"""FastAPI service that n8n calls for the heavy lifting (parsing, embeddings, LLM)."""

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl

from app.ingest import DocumentRejected, ingest_tender

app = FastAPI(title="Tender Radar service")


class IngestRequest(BaseModel):
    # TED publication numbers look like "674261-2026"; the strict pattern also
    # keeps arbitrary strings out of chunk ids and metadata filters.
    tender_id: str = Field(pattern=r"^\d{1,10}-\d{4}$")
    pdf_url: HttpUrl


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict:
    try:
        chunks = ingest_tender(request.tender_id, str(request.pdf_url))
    except DocumentRejected as error:
        raise HTTPException(status_code=422, detail=str(error))
    except httpx.HTTPError as error:
        raise HTTPException(status_code=502, detail=f"PDF download failed: {error}")
    return {"tender_id": request.tender_id, "chunks": chunks}
