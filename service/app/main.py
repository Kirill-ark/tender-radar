"""FastAPI service that n8n calls for the heavy lifting (parsing, embeddings, LLM)."""

import time

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl

from app.ask import answer_question
from app.card import TenderNotFound, generate_card
from app.config import MAX_QUESTION_CHARS
from app.ingest import DocumentRejected, ingest_tender
from app.llm import LLMUnavailable
from app.metrics import record, summary

app = FastAPI(title="Tender Radar service")


# TED publication numbers look like "674261-2026"; the strict pattern also
# keeps arbitrary strings out of chunk ids and metadata filters.
TENDER_ID = Field(pattern=r"^\d{1,10}-\d{4}$")


class IngestRequest(BaseModel):
    tender_id: str = TENDER_ID
    pdf_url: HttpUrl


class CardRequest(BaseModel):
    tender_id: str = TENDER_ID


class AskRequest(BaseModel):
    tender_id: str = TENDER_ID
    question: str = Field(min_length=3, max_length=MAX_QUESTION_CHARS)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict:
    return summary()


# Each endpoint logs its outcome (status, duration, tokens) for the metrics,
# failures included: the failure rate is one of the README numbers.


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict:
    started = time.monotonic()
    try:
        chunks = ingest_tender(request.tender_id, str(request.pdf_url))
    except DocumentRejected as error:
        record("ingest", request.tender_id, 422, started)
        raise HTTPException(status_code=422, detail=str(error))
    except httpx.HTTPError as error:
        record("ingest", request.tender_id, 502, started)
        raise HTTPException(status_code=502, detail=f"PDF download failed: {error}")
    record("ingest", request.tender_id, 200, started)
    return {"tender_id": request.tender_id, "chunks": chunks}


@app.post("/card")
def card(request: CardRequest) -> dict:
    started = time.monotonic()
    try:
        result = generate_card(request.tender_id)
    except TenderNotFound:
        record("card", request.tender_id, 404, started)
        raise HTTPException(status_code=404, detail="tender not ingested")
    except LLMUnavailable as error:
        record("card", request.tender_id, 503, started)
        raise HTTPException(status_code=503, detail=str(error))
    record("card", request.tender_id, 200, started, result["usage"])
    return result


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    started = time.monotonic()
    try:
        result = answer_question(request.tender_id, request.question)
    except LLMUnavailable as error:
        record("ask", request.tender_id, 503, started)
        raise HTTPException(status_code=503, detail=str(error))
    record("ask", request.tender_id, 200, started, result["usage"])
    return result
