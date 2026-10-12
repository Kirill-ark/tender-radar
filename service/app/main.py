"""FastAPI service that n8n calls for the heavy lifting (parsing, embeddings, LLM)."""

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl

from app.card import TenderNotFound, generate_card
from app.ingest import DocumentRejected, ingest_tender
from app.llm import LLMUnavailable

app = FastAPI(title="Tender Radar service")


# TED publication numbers look like "674261-2026"; the strict pattern also
# keeps arbitrary strings out of chunk ids and metadata filters.
TENDER_ID = Field(pattern=r"^\d{1,10}-\d{4}$")


class IngestRequest(BaseModel):
    tender_id: str = TENDER_ID
    pdf_url: HttpUrl


class CardRequest(BaseModel):
    tender_id: str = TENDER_ID


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


@app.post("/card")
def card(request: CardRequest) -> dict:
    try:
        return generate_card(request.tender_id)
    except TenderNotFound:
        raise HTTPException(status_code=404, detail="tender not ingested")
    except LLMUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error))
