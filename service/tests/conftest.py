"""Shared fixtures. Tests never call Groq, TED or the network: those are faked."""

import fitz  # PyMuPDF
import pytest


@pytest.fixture
def make_pdf():
    """Build an in-memory PDF; each string is one page ("" makes a blank page)."""

    def build(*pages: str) -> bytes:
        doc = fitz.open()
        for text in pages:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
        data = doc.tobytes()
        doc.close()
        return data

    return build
