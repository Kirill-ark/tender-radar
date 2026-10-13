"""Parsing, chunking and the document limits (guardrails on untrusted PDFs)."""

import httpx
import pytest

from app import ingest
from app.ingest import DocumentRejected, chunk_text, download_pdf, parse_pdf


# --- chunking ---


def test_chunks_respect_size_and_overlap():
    text = "".join(str(i % 10) for i in range(1000))
    chunks = chunk_text(text, size=400, overlap=50)
    assert all(len(chunk) <= 400 for chunk in chunks)
    # Neighbouring chunks share exactly `overlap` characters.
    assert chunks[0][-50:] == chunks[1][:50]
    # Nothing is lost: the chunks cover the whole text.
    assert chunks[-1].endswith(text[-10:])


def test_short_text_is_one_chunk():
    assert chunk_text("short text", size=400, overlap=50) == ["short text"]


def test_no_tiny_duplicate_tail():
    text = "x" * 800
    chunks = chunk_text(text, size=400, overlap=50)
    # The window that already reached the end must not spawn an extra chunk.
    assert len(chunks) == 3


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("abc", size=100, overlap=100)


# --- parsing ---


def test_parse_keeps_page_numbers_and_skips_blank_pages(make_pdf):
    pages = parse_pdf(make_pdf("Seite eins", "", "Seite drei"), "t.pdf")
    assert [p["page"] for p in pages] == [1, 3]
    assert pages[0]["text"] == "Seite eins"
    assert all(p["file"] == "t.pdf" for p in pages)


def test_parse_rejects_too_many_pages(make_pdf, monkeypatch):
    monkeypatch.setattr(ingest, "MAX_PDF_PAGES", 2)
    with pytest.raises(DocumentRejected, match="3 pages"):
        parse_pdf(make_pdf("a", "b", "c"), "t.pdf")


# --- download limits ---


def fake_download(monkeypatch, body: bytes) -> None:
    """Serve `body` for any URL through httpx's mock transport, without the network."""
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body)))
    monkeypatch.setattr(ingest.httpx, "stream", client.stream)


def test_download_accepts_a_pdf(monkeypatch, make_pdf):
    pdf = make_pdf("hello")
    fake_download(monkeypatch, pdf)
    assert download_pdf("https://example.org/t.pdf") == pdf


def test_download_rejects_non_pdf(monkeypatch):
    fake_download(monkeypatch, b"<html>not a pdf</html>")
    with pytest.raises(DocumentRejected, match="not a PDF"):
        download_pdf("https://example.org/page")


def test_download_rejects_oversized_file(monkeypatch):
    monkeypatch.setattr(ingest, "MAX_PDF_BYTES", 1000)
    fake_download(monkeypatch, b"%PDF" + b"0" * 5000)
    with pytest.raises(DocumentRejected, match="larger than"):
        download_pdf("https://example.org/big.pdf")
